# distill_cbam.py
"""
Student = EfficientNet-B2 + Multi-Kernel CBAM, trained with teacher (EfficientNet-B5) distillation.
"""

import os, random, warnings
import numpy as np, pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt, seaborn as sns

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import timm

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Config
# -------------------------
config = {
    "csv_path": "train.csv",
    "img_root": "./diabetes/colored_images",
    "student_backbone": "efficientnet_b2",
    "teacher_backbone": "efficientnet_b5",
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    "early_stop_patience": 5,
    "gradcam_out_dir": "distill_cbam_gradcam",
    "image_size": 224,
    "num_workers": 4,
    "seed": 42
}
random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", device)

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([transforms.Resize((config["image_size"], config["image_size"])), transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# -------------------------
# Dataset (same as baseline)
# -------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png",".jpg",".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root; self.transform = transform
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]; self.label_col = self.data.columns[1]
        self.img_exts = img_exts
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root,f))]) if os.path.isdir(img_root) else []
        self.numeric_to_folder = {0:"No_DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferate_DR"}
    def __len__(self): return len(self.data)
    def _find_image(self, folder, img_id):
        for ext in self.img_exts:
            p = os.path.join(self.img_root, folder, f"{img_id}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_root, folder, img_id)
        if os.path.exists(p): return p
        p2 = os.path.join(self.img_root, img_id)
        if os.path.exists(p2): return p2
        raise FileNotFoundError(f"{img_id} not found in {folder}")
    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx][self.image_col]); label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val,(int,float)) or str(label_val).isdigit():
            label_val=int(label_val); folder=self.numeric_to_folder.get(label_val,str(label_val)); lbl=label_val
        else:
            folder=str(label_val); lbl=self.folder_names.index(folder) if folder in self.folder_names else 0
        p = self._find_image(folder, img_id); img=Image.open(p).convert("RGB")
        if self.transform: img=self.transform(img)
        return img,lbl

# load dataset
dataset=None; use_imagefolder=False
if os.path.exists(config["csv_path"]) and os.path.isdir(config["img_root"]):
    try:
        dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=train_transform); print("Loaded CSV dataset")
    except Exception as e:
        print("CSV load failed:", e); dataset=None

if dataset is None:
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(config["img_root"], transform=train_transform); use_imagefolder=True; print("Loaded ImageFolder")
    else:
        raise FileNotFoundError("Dataset missing")

class_names = dataset.classes if use_imagefolder else ["No_DR","Mild","Moderate","Severe","Proliferate_DR"]
total = len(dataset); train_n=int(0.8*total); val_n=total-train_n
train_ds, val_ds = random_split(dataset,[train_n,val_n], generator=torch.Generator().manual_seed(config["seed"]))

if use_imagefolder:
    class Wrapped(torch.utils.data.Dataset):
        def __init__(self, base, indices, transform): self.base=base; self.indices=indices; self.transform=transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            real_idx=self.indices[idx]; path,label=self.base.samples[real_idx]
            img=Image.open(path).convert("RGB"); 
            if self.transform: img=self.transform(img)
            return img,label
    train_ds = Wrapped(dataset, train_ds.indices, train_transform)
    val_ds = Wrapped(dataset, val_ds.indices, val_transform)
else:
    train_ds.dataset.transform = train_transform; val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
print(f"Dataset: {total} images -> train {len(train_ds)} val {len(val_ds)} classes {class_names}")

# -------------------------
# CBAM module
# -------------------------
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels//reduction)
        self.mlp = nn.Sequential(nn.Linear(channels,mid,bias=False), nn.ReLU(inplace=True), nn.Linear(mid,channels,bias=False))
    def forward(self,x):
        b,c,h,w = x.size()
        avg = torch.mean(x, dim=(2,3)); mx,_ = torch.max(x.view(b,c,-1), dim=2)
        out = self.mlp(avg) + self.mlp(mx); return torch.sigmoid(out).view(b,c,1,1)

class SpatialGateMultiK(nn.Module):
    def __init__(self, kernel_sizes=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(2,1,k,padding=(k-1)//2,bias=False) for k in kernel_sizes])
        self.bns = nn.ModuleList([nn.BatchNorm2d(1) for _ in kernel_sizes])
    def forward(self,x):
        avg = torch.mean(x, dim=1, keepdim=True); mx,_ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg,mx], dim=1); outs=[torch.sigmoid(bn(conv(cat))) for conv,bn in zip(self.convs,self.bns)]
        return sum(outs)/len(outs)

class CBAM_MultiK(nn.Module):
    def __init__(self, channels): super().__init__(); self.cg=ChannelGate(channels); self.sg=SpatialGateMultiK()
    def forward(self,x): x = x * self.cg(x); x = x * self.sg(x); return x

# -------------------------
# Models: teacher and student with CBAM
# -------------------------
teacher = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
teacher.eval()
for p in teacher.parameters(): p.requires_grad=False

class StudentCBAM(nn.Module):
    def __init__(self, backbone='efficientnet_b2', num_classes=5, pretrained=True, use_cbam=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        out_ch = self.backbone.feature_info[-1]['num_chs']
        self.cbam = CBAM_MultiK(out_ch) if use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self,x):
        feats = self.backbone(x)[-1]
        if self.cbam: feats = self.cbam(feats)
        pooled = self.pool(feats).flatten(1)
        return self.fc(pooled)

student = StudentCBAM(config["student_backbone"], config["num_classes"], pretrained=True, use_cbam=True).to(device)

# -------------------------
# Distillation loss
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher_model=None, alpha=0.5, temperature=4.0):
        super().__init__(); self.base_loss=base_loss; self.teacher=teacher_model; self.alpha=alpha; self.T=temperature
        if self.teacher is not None:
            self.teacher.eval(); 
            for p in self.teacher.parameters(): p.requires_grad=False
    def forward(self, student_logits, inputs, labels):
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None: return hard
        with torch.no_grad(): t_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits/self.T, dim=1)
        t_prob = nn.functional.softmax(t_logits/self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T*self.T)
        return self.alpha * soft + (1.0 - self.alpha) * hard

base_loss = nn.CrossEntropyLoss()
criterion = DistillationLoss(base_loss, teacher, alpha=0.5, temperature=4.0)

optimizer = optim.AdamW(student.parameters(), lr=config["lr"])
if len(train_loader)>0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"]*10, steps_per_epoch=len(train_loader), epochs=config["epochs"])
else:
    scheduler = None

# -------------------------
# Trainer and loop (same style)
# -------------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model=model; self.criterion=criterion; self.optimizer=optimizer; self.scheduler=scheduler; self.device=device
    def train_epoch(self, loader):
        self.model.train(); running_loss=0.0; n=0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, imgs, labels)
            loss.backward(); self.optimizer.step()
            if self.scheduler is not None:
                try: self.scheduler.step()
                except: pass
            running_loss += float(loss.item()) * imgs.size(0); n += imgs.size(0)
        return running_loss / max(1,n)
    def evaluate(self, loader):
        self.model.eval(); preds=[]; trues=[]
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.numpy().tolist())
        return preds, trues

trainer = Trainer(student, criterion, optimizer, scheduler, device)

os.makedirs(config["gradcam_out_dir"], exist_ok=True)
best_val_loss=float("inf"); patience_counter=0; patience=config["early_stop_patience"]; train_losses=[]

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader); train_losses.append(train_loss)
    # validation
    student.eval(); val_loss=0.0; total=0; preds=[]; trues=[]
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = student(imgs)
            loss = criterion(outputs, imgs, labels)
            val_loss += float(loss.item()) * imgs.size(0); total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist()); trues.extend(labels.cpu().numpy().tolist())
    val_loss /= max(1,total)
    print(f"Epoch {epoch+1}/{config['epochs']} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss; patience_counter=0
        torch.save(student.state_dict(), "best_model.pth"); print("Saved best_model.pth")
    else:
        patience_counter += 1; print(f"No improvement for {patience_counter}")
        if patience_counter >= patience: print("Early stopping"); break
    try: print(classification_report(trues, preds, digits=4, target_names=class_names))
    except: pass

# -------------------------
# Grad-CAM (same method as baseline)
# -------------------------
def find_last_conv_module(net):
    last_mod=None
    for name,module in net.named_modules():
        if isinstance(module, nn.Conv2d): last_mod=module
    return last_mod

def generate_gradcam(model, input_tensor, target_class, device):
    model.eval(); last_conv=find_last_conv_module(model)
    if last_conv is None: raise RuntimeError("No Conv2d found")
    acts={}; grads={}
    def fwd(m,i,o): acts['val']=o.detach()
    def bwd(m,gi,go): grads['val']=go[0].detach()
    fh = last_conv.register_forward_hook(fwd); bh = last_conv.register_backward_hook(bwd)
    outputs = model(input_tensor)
    score = outputs[0, target_class]
    model.zero_grad(); score.backward(retain_graph=True)
    act = acts['val']; grad = grads['val']
    weights = torch.mean(grad, dim=(2,3), keepdim=True)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)
    cam_map = torch.relu(cam_map).squeeze().cpu().numpy()
    cam_map -= cam_map.min(); 
    if cam_map.max()!=0: cam_map/=cam_map.max()
    fh.remove(); bh.remove()
    return cam_map

# build per-class indices
indices_per_class = {i:[] for i in range(len(class_names))}
if isinstance(val_ds, Subset):
    underlying = val_ds.dataset
    for local_idx, global_idx in enumerate(val_ds.indices):
        try: img,lbl = val_ds[local_idx]
        except: img,lbl = underlying[global_idx]
        indices_per_class[lbl].append(local_idx)
else:
    for i in range(len(val_ds)):
        _, lbl = val_ds[i]; indices_per_class[lbl].append(i)

inv_mean = np.array([0.485,0.456,0.406]); inv_std=np.array([0.229,0.224,0.225])
print("\nGenerating Grad-CAMs ..."); saved=0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx,[]); 
    if not idxs: print(f"No samples for {cls_name}"); continue
    sel = random.choice(idxs); img_tensor,lbl = val_ds[sel]
    if isinstance(img_tensor, torch.Tensor): input_tensor = img_tensor.unsqueeze(0).to(device)
    else:
        arr=np.array(img_tensor)
        if arr.ndim==3 and arr.shape[2]==3: input_tensor=torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(device).float()
        else: continue
    try: cam = generate_gradcam(student, input_tensor, int(lbl), device)
    except Exception as e: print("Grad-CAM failed:", e); continue
    cam_resized = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heatmap = np.uint8(255*cam_resized); heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().permute(1,2,0).numpy()
        img_np = (img_np * inv_std) + inv_mean; img_np = np.clip(img_np*255,0,255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        arr = img_tensor
        if arr.max()<=1.0: arr=(arr*255).astype(np.uint8)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr,0.6,heatmap,0.4,0)
    out_path = os.path.join(config["gradcam_out_dir"], f"{cls_name}_gradcam.jpg"); os.makedirs(config["gradcam_out_dir"], exist_ok=True)
    cv2.imwrite(out_path, overlay); print("Saved", out_path); saved+=1

print(f"Saved {saved} Grad-CAM(s) in {config['gradcam_out_dir']}")
