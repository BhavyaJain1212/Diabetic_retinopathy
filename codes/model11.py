# baseline_distill.py
"""
Baseline student (EfficientNet-B2) trained with teacher distillation (EfficientNet-B5).
No CBAM / refine / focal / TTA — pure baseline + distillation.
Saves best_model.pth and Grad-CAMs to config["gradcam_out_dir"].
"""

import os, random, warnings
import numpy as np, pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
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
    "teacher_backbone": "efficientnet_b5",
    "student_backbone": "efficientnet_b2",
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    "use_distill": True,
    "alpha": 0.5,            # distillation alpha
    "temperature": 4.0,      # distillation temperature
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2B5_baseline_distill_gradcam",
    "image_size": 224,
    "num_workers": 4,
    "seed": 42
}

os.makedirs(config["gradcam_out_dir"], exist_ok=True)

# reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Dataset loader (CSV first, else ImageFolder)
# -------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png",".jpg",".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root,f))]) if os.path.isdir(img_root) else []
        self.numeric_to_folder = {0:"No_DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferate_DR"}
        self.img_exts = img_exts

    def __len__(self): return len(self.data)

    def _find_image(self, folder, img_id):
        for ext in self.img_exts:
            p = os.path.join(self.img_root, folder, f"{img_id}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_root, folder, img_id)
        if os.path.exists(p): return p
        p2 = os.path.join(self.img_root, img_id)
        if os.path.exists(p2): return p2
        raise FileNotFoundError(f"Image {img_id} not found in {folder}")

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx][self.image_col])
        label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val,(int,float)) or str(label_val).isdigit():
            label_val = int(label_val)
            folder_name = self.numeric_to_folder.get(label_val, str(label_val))
            label_idx = int(label_val)
        else:
            folder_name = str(label_val)
            label_idx = self.folder_names.index(folder_name) if folder_name in self.folder_names else 0
        img_path = self._find_image(folder_name, img_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label_idx

# build dataset
dataset = None
use_imagefolder = False
if os.path.exists(config["csv_path"]) and os.path.isdir(config["img_root"]):
    try:
        dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=train_transform)
        print("Loaded CSV dataset.")
    except Exception as e:
        print("CSV load failed:", e)
        dataset = None

if dataset is None:
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(root=config["img_root"], transform=train_transform)
        use_imagefolder = True
        print("Loaded ImageFolder dataset.")
    else:
        raise FileNotFoundError("No dataset found.")

class_names = dataset.classes if use_imagefolder else ["No_DR","Mild","Moderate","Severe","Proliferate_DR"]

# split
total = len(dataset)
train_n = int(0.8 * total)
val_n = total - train_n
train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(config["seed"]))

# if ImageFolder, wrap to set transform separately for val
if use_imagefolder:
    class WrappedImageFolder(torch.utils.data.Dataset):
        def __init__(self, base, indices, transform):
            self.base, self.indices, self.transform = base, indices, transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            real = self.indices[idx]
            path, lbl = self.base.samples[real]
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, lbl
    train_ds = WrappedImageFolder(dataset, train_ds.indices, transform=train_transform)
    val_ds = WrappedImageFolder(dataset, val_ds.indices, transform=val_transform)
else:
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

print(f"Dataset: {total} images → train:{len(train_ds)} val:{len(val_ds)}")

# -------------------------
# Models: teacher + student
# -------------------------
teacher = None
if config["use_distill"]:
    teacher = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    print("Teacher loaded:", config["teacher_backbone"])

student = timm.create_model(config["student_backbone"], pretrained=True, features_only=True)
out_ch = student.feature_info[-1]['num_chs']
class StudentBaseline(nn.Module):
    def __init__(self, feat_model, out_ch, num_classes):
        super().__init__()
        self.features = feat_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        f = self.features(x)
        if isinstance(f,(list,tuple)): f = f[-1]
        p = self.pool(f).flatten(1)
        return self.fc(p)

student_model = StudentBaseline(student, out_ch, config["num_classes"]).to(device)

# -------------------------
# Distillation loss
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher, alpha=0.5, T=4.0):
        super().__init__()
        self.base = base_loss
        self.teacher = teacher
        self.alpha = alpha
        self.T = T
    def forward(self, student_logits, inputs, labels):
        hard = self.base(student_logits, labels)
        if self.teacher is None:
            return hard
        with torch.no_grad():
            t_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits / self.T, dim=1)
        t_prob = nn.functional.softmax(t_logits / self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T * self.T)
        return self.alpha * soft + (1.0 - self.alpha) * hard

base_loss = nn.CrossEntropyLoss()
criterion = DistillationLoss(base_loss, teacher, alpha=config["alpha"], T=config["temperature"]) if config["use_distill"] else base_loss

optimizer = optim.AdamW(student_model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"]*10, epochs=config["epochs"], steps_per_epoch=len(train_loader)) if len(train_loader)>0 else None

# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, model, criterion, opt, sched, device):
        self.model, self.criterion, self.opt, self.sched, self.device = model, criterion, opt, sched, device

    def train_epoch(self, loader):
        self.model.train()
        running, n = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, imgs, labels) if isinstance(self.criterion, DistillationLoss) else self.criterion(outputs, labels)
            loss.backward()
            self.opt.step()
            if self.sched is not None:
                try: self.sched.step()
                except: pass
            running += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        return running / max(1,n)

    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outs = self.model(imgs)
                preds.extend(outs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        return preds, trues

trainer = Trainer(student_model, criterion, optimizer, scheduler, device)

# -------------------------
# Training loop + early stopping
# -------------------------
best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader)
    # validation
    student_model.eval()
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = student_model(imgs)
            loss = criterion(outs, imgs, labels) if isinstance(criterion, DistillationLoss) else criterion(outs, labels)
            val_loss += float(loss.item()) * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    val_loss /= max(1, total)
    print(f"Epoch {epoch+1}/{config['epochs']} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(student_model.state_dict(), "best_model.pth")
        print("✅ Saved best_model.pth")
    else:
        patience_counter += 1
        print(f"⚠ No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print("🛑 Early stopping")
            break

    try:
        print(classification_report(trues, preds, digits=4, target_names=class_names))
    except Exception:
        pass

# -------------------------
# Final evaluation & confusion matrix
# -------------------------
preds, trues = trainer.evaluate(val_loader)
print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (baseline distill)")
plt.savefig(os.path.join(config["gradcam_out_dir"], "confusion_matrix.png"))

# -------------------------
# Grad-CAM (last conv)
# -------------------------
def find_last_conv(net):
    last = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def generate_gradcam(model, input_tensor, target_class, device):
    model.eval()
    last_conv = find_last_conv(model)
    if last_conv is None:
        raise RuntimeError("No conv layer found")
    acts, grads = {}, {}
    def fwd(m,i,o): acts['v']=o.detach()
    def bwd(m,gi,go): grads['v']=go[0].detach()
    fh = last_conv.register_forward_hook(fwd)
    bh = last_conv.register_backward_hook(bwd)
    outs = model(input_tensor)
    score = outs[0, target_class]
    model.zero_grad(); score.backward(retain_graph=True)
    act = acts['v']; grad = grads['v']
    weights = torch.mean(grad, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = torch.relu(cam).squeeze().cpu().numpy()
    cam -= cam.min()
    if cam.max()!=0: cam/=cam.max()
    fh.remove(); bh.remove()
    return cam

os.makedirs(config["gradcam_out_dir"], exist_ok=True)
student_model.to(device); student_model.eval()
# map class->val indices
indices_per_class = {i:[] for i in range(len(class_names))}
if isinstance(val_ds, Subset):
    underlying = val_ds.dataset
    for local_idx, global_idx in enumerate(val_ds.indices):
        try:
            img,lbl = val_ds[local_idx]
        except:
            img,lbl = underlying[global_idx]
        indices_per_class[lbl].append(local_idx)
else:
    for i in range(len(val_ds)):
        _, lbl = val_ds[i]
        indices_per_class[lbl].append(i)

inv_mean = np.array([0.485,0.456,0.406]); inv_std = np.array([0.229,0.224,0.225])
saved = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs: continue
    sel = random.choice(idxs)
    img_tensor, lbl = val_ds[sel]
    if isinstance(img_tensor, torch.Tensor):
        inp = img_tensor.unsqueeze(0).to(device)
    else:
        arr = np.array(img_tensor); inp = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(device).float()
    try:
        cam = generate_gradcam(student_model, inp, int(lbl), device)
    except Exception as e:
        print("Grad-CAM failed:", e); continue
    cam_resized = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heat = np.uint8(255 * cam_resized); heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().permute(1,2,0).numpy(); img_np = (img_np * inv_std) + inv_mean
        img_np = np.clip(img_np*255,0,255).astype(np.uint8); img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        arr = img_tensor; arr = (arr*255).astype(np.uint8); img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    over = cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
    out_path = os.path.join(config["gradcam_out_dir"], f"{cls_name}_gradcam.jpg")
    cv2.imwrite(out_path, over)
    print("Saved Grad-CAM:", out_path); saved += 1

print(f"\nSaved {saved} Grad-CAM(s) to {config['gradcam_out_dir']}")
