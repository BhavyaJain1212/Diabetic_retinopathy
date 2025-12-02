import os
import random
import warnings
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import classification_report, confusion_matrix
import timm

# --- TDA Import ---
try:
    import gudhi as gd
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("⚠ 'gudhi' library not found. TDA features will be zeros. Run: pip install gudhi")

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Config
# -------------------------
config = {
    "csv_path": "train.csv",
    "img_root": "../dataset/india_dataset/colored_images",
    "backbone": "efficientnet_b2",
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 25,
    "lr": 1e-4,
    "use_cbam": True,
    "use_distill": True,
    "teacher_backbone": "efficientnet_b5",
    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_B5_Distill_TDA",
    "image_size": 224,
    
    # Teacher Training Config
    "train_teacher": True,          # Set to True to train teacher first
    "teacher_epochs": 5,            # Epochs for teacher
    
    # Distillation Config
    "distill_alpha_start": 0.2,
    "distill_alpha_end": 0.8,
    "distill_temperature": 4.0,
    
    # Regularization
    "consistency_weight": 0.1,
    "use_class_weights": True,
    "label_smoothing": 0.1,
    "max_grad_norm": 5.0,
    "rotation_angles": [0, 90, 180, 270],
    
    # TDA Config
    "use_tda": True,
    "tda_img_size": 32,             # Small size for fast TDA computation
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -------------------------
# Device Setup
# -------------------------
device = torch.device(config["device"])
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ No GPU found. Using CPU.")

# -------------------------
# Data Transforms & Dataset
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png", ".jpg", ".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))])
        self.numeric_to_folder = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR"}
        self.img_exts = img_exts

    def __len__(self):
        return len(self.data)

    def _find_image(self, folder, img_id):
        for ext in self.img_exts:
            p = os.path.join(self.img_root, folder, f"{img_id}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_root, folder, img_id)
        if os.path.exists(p): return p
        raise FileNotFoundError(f"Image for id {img_id} not found in {folder}")

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx][self.image_col])
        label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val, (int, float)) or str(label_val).isdigit():
            label_val = int(label_val)
            folder_name = self.numeric_to_folder[label_val]
            label_idx = label_val
        else:
            folder_name = str(label_val)
            label_idx = self.folder_names.index(folder_name)
        img_path = self._find_image(folder_name, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# Load Dataset
dataset = None
use_imagefolder = False
if os.path.exists(config["csv_path"]) and os.path.isdir(config["img_root"]):
    try:
        dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=train_transform)
        print("✅ Loaded dataset from CSV + folders.")
    except Exception as e:
        print(f"⚠ Failed to load CSV-based dataset: {e}")
        dataset = None

if dataset is None:
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(root=config["img_root"], transform=train_transform)
        use_imagefolder = True
        print("✅ Loaded dataset with ImageFolder.")
    else:
        raise FileNotFoundError(f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist.")

if use_imagefolder:
    class_names = dataset.classes
else:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

if use_imagefolder:
    val_dataset_full = datasets.ImageFolder(root=config["img_root"], transform=val_transform)
    val_ds = Subset(val_dataset_full, val_ds.indices)
else:
    val_ds = Subset(RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform), val_ds.indices)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

print(f"Dataset: {len(dataset)} images → train: {len(train_ds)}, val: {len(val_ds)}")

# -------------------------
# TDA Utilities
# -------------------------
def compute_persistence_entropy(persistence, dim):
    pts = [p[1] for p in persistence if p[0] == dim]
    if not pts: return 0.0, 0.0
    pts = np.array(pts)
    births = pts[:, 0]
    deaths = pts[:, 1]
    max_val = np.max(deaths[deaths != np.inf]) if np.any(deaths != np.inf) else 1.0
    deaths[deaths == np.inf] = max_val + 0.1
    lifetimes = deaths - births
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0: return 0.0, 0.0
    sum_lifetime = np.sum(lifetimes)
    if sum_lifetime == 0: return 0.0, 0.0
    probs = lifetimes / sum_lifetime
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    return entropy, sum_lifetime

def extract_tda_features_batch(images_tensor, tda_size=32):
    if not TDA_AVAILABLE:
        return torch.zeros((images_tensor.size(0), 4), device=images_tensor.device)
    
    # Downsample and convert to grayscale for speed
    small_imgs = torch.nn.functional.interpolate(images_tensor, size=(tda_size, tda_size), mode='bilinear', align_corners=False)
    gray_imgs = 0.299 * small_imgs[:, 0, :, :] + 0.587 * small_imgs[:, 1, :, :] + 0.114 * small_imgs[:, 2, :, :]
    gray_imgs_np = gray_imgs.cpu().detach().numpy()
    
    batch_features = []
    for i in range(gray_imgs_np.shape[0]):
        img_arr = gray_imgs_np[i]
        cc = gd.CubicalComplex(dimensions=img_arr.shape, top_dimensional_cells=img_arr.flatten())
        persistence = cc.persistence()
        h0_ent, h0_sum = compute_persistence_entropy(persistence, 0)
        h1_ent, h1_sum = compute_persistence_entropy(persistence, 1)
        batch_features.append([h0_ent, h0_sum, h1_ent, h1_sum])
        
    return torch.tensor(batch_features, dtype=torch.float32, device=images_tensor.device)

# -------------------------
# CBAM & Student Model
# -------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, max(1, channels//reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels//reduction), channels, bias=False)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(1)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        # Channel
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        c_out = torch.sigmoid(self.channel_gate(avg_pool) + self.channel_gate(max_pool)).view(b, c, 1, 1)
        x = x * c_out
        # Spatial
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        s_out = torch.sigmoid(self.spatial_gate(torch.cat([avg_s, max_s], dim=1)))
        return x * s_out

class EfficientNetWithCBAM(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=5, use_cbam=True, pretrained=True, use_tda=False):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAMBlock(out_channels) if use_cbam else None
        self.use_tda = use_tda
        self.tda_dim = 4 if use_tda else 0
        self.classifier = nn.Linear(out_channels + self.tda_dim, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)): feats = feats[-1]
        if self.cbam: feats = self.cbam(feats)
        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        
        if self.use_tda:
            tda_feats = extract_tda_features_batch(x, tda_size=config["tda_img_size"])
            tda_feats = torch.log1p(tda_feats) # Normalize
            combined = torch.cat((pooled, tda_feats), dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(pooled)
        return logits

# -------------------------
# Loss & Trainer
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher_model=None, alpha_start=0.5, alpha_end=0.5, temperature=4.0):
        super().__init__()
        self.base_loss = base_loss
        self.teacher = teacher_model
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.T = temperature
        self.current_epoch = 0
        self.max_epochs = 1
        # Freeze teacher immediately
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def set_epoch(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def get_alpha(self):
        if self.max_epochs <= 1: return self.alpha_end
        frac = min(1.0, max(0.0, self.current_epoch / float(self.max_epochs - 1)))
        return self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def forward(self, student_logits, inputs, labels):
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None: return hard
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits / self.T, dim=1)
        t_prob = nn.functional.softmax(teacher_logits / self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T * self.T)
        alpha = self.get_alpha()
        return alpha * soft + (1.0 - alpha) * hard

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

    def train_epoch(self, loader, epoch_idx=0, total_epochs=1):
        self.model.train()
        running_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            if isinstance(self.criterion, DistillationLoss):
                if hasattr(self.criterion, "set_epoch"):
                    self.criterion.set_epoch(epoch_idx, total_epochs)
                loss = self.criterion(outputs, imgs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            if self.config.get("consistency_weight", 0.0) > 0.0:
                angle = random.choice(self.config["rotation_angles"])
                if angle % 360 != 0:
                    rotated_imgs = torch.stack([transforms.functional.rotate(img.cpu(), angle) for img in imgs]).to(self.device)
                    with torch.no_grad():
                        out_orig = nn.functional.log_softmax(outputs, dim=1)
                    out_rot = self.model(rotated_imgs)
                    out_rot_logp = nn.functional.log_softmax(out_rot, dim=1)
                    kl_loss = nn.functional.kl_div(out_orig, nn.functional.softmax(out_rot.detach(), dim=1), reduction='batchmean')
                    loss += self.config["consistency_weight"] * kl_loss

            loss.backward()
            if self.config.get("max_grad_norm", None):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return running_loss / max(1, n)

    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.numpy().tolist())
        return preds, trues

# -------------------------
# EXECUTION FLOW
# -------------------------

# 1. Setup Class Weights & Base Loss
class_weights = None
if config["use_class_weights"]:
    label_counts = np.zeros(config["num_classes"], dtype=np.int64)
    for i in range(len(train_ds)):
        _, lbl = train_ds[i]
        label_counts[lbl] += 1
    freq = label_counts / float(label_counts.sum())
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", weights)

if config["loss_fn"].lower() == "crossentropyloss":
    base_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
else:
    base_loss = nn.CrossEntropyLoss(weight=class_weights)

# 2. Teacher Setup & Training
teacher_model = None
if config["use_distill"]:
    teacher_model = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
    print(f"✅ Teacher initialized: {config['teacher_backbone']}")

    if config["train_teacher"]:
        print("\n🎓 Starting Teacher Training...")
        # Ensure params are trainable
        for p in teacher_model.parameters(): p.requires_grad = True
        
        t_opt = optim.AdamW(teacher_model.parameters(), lr=config["lr"]*0.5)
        t_sched = torch.optim.lr_scheduler.OneCycleLR(t_opt, max_lr=config["lr"], steps_per_epoch=len(train_loader), epochs=config["teacher_epochs"])
        t_trainer = Trainer(teacher_model, base_loss, t_opt, t_sched, device, config)
        
        best_t_loss = float("inf")
        for ep in range(config["teacher_epochs"]):
            tl = t_trainer.train_epoch(train_loader, ep, config["teacher_epochs"])
            # Val
            teacher_model.eval()
            vl, tot = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    vl += base_loss(teacher_model(imgs), labels).item() * imgs.size(0)
                    tot += imgs.size(0)
            vl /= max(1, tot)
            print(f"Teacher Epoch {ep+1}/{config['teacher_epochs']} - Train: {tl:.4f} - Val: {vl:.4f}")
            if vl < best_t_loss:
                best_t_loss = vl
                torch.save(teacher_model.state_dict(), "best_teacher.pth")
        
        # Reload best teacher
        if os.path.exists("best_teacher.pth"):
            teacher_model.load_state_dict(torch.load("best_teacher.pth", map_location=device))
        print("🎓 Teacher Training Complete.\n")

# 3. Initialize Distillation Loss (freezes teacher)
if config["use_distill"]:
    criterion = DistillationLoss(
        base_loss, 
        teacher_model=teacher_model, 
        alpha_start=config["distill_alpha_start"], 
        alpha_end=config["distill_alpha_end"]
    )
else:
    criterion = base_loss

# 4. Student Setup
model = EfficientNetWithCBAM(
    config["backbone"], 
    config["num_classes"], 
    config["use_cbam"], 
    pretrained=True, 
    use_tda=config["use_tda"]
).to(device)
print(f"✅ Student initialized: {config['backbone']} (TDA Fusion: {config['use_tda']})")

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"]*5, steps_per_epoch=len(train_loader), epochs=config["epochs"])
trainer = Trainer(model, criterion, optimizer, scheduler, device, config)

# 5. Student Training Loop
train_losses = []
best_val_loss = float("inf")
patience = config["early_stop_patience"]
counter = 0

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader, epoch, config["epochs"])
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss, total = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(criterion, DistillationLoss):
                criterion.set_epoch(epoch, config["epochs"])
                loss = criterion(outputs, imgs, labels)
            else:
                loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    val_loss /= max(1, total)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model saved.")
    else:
        counter += 1
        print(f"⚠ No improvement ({counter}/{patience})")
        if counter >= patience:
            print("🛑 Early stopping.")
            break

# 6. Evaluation
plt.plot(train_losses); plt.title("Training Loss"); plt.show()
preds, trues = trainer.evaluate(val_loader)
print(classification_report(trues, preds, target_names=class_names))
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.show()

# 7. Grad-CAM (Simplified for brevity)
def get_cam(model, x, cls_idx):
    model.eval()
    # Find last conv
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d): last_conv = m
    if not last_conv: return None
    
    acts, grads = {}, {}
    def f_hook(m, i, o): acts['v'] = o.detach()
    def b_hook(m, gi, go): grads['v'] = go[0].detach()
    h1 = last_conv.register_forward_hook(f_hook)
    h2 = last_conv.register_full_backward_hook(b_hook)
    
    out = model(x)
    model.zero_grad()
    out[0, cls_idx].backward()
    
    w = torch.mean(grads['v'], dim=(2,3), keepdim=True)
    cam = torch.relu(torch.sum(w * acts['v'], dim=1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    
    h1.remove(); h2.remove()
    return cam

os.makedirs(config["gradcam_out_dir"], exist_ok=True)
if os.path.exists("best_model.pth"): model.load_state_dict(torch.load("best_model.pth", map_location=device))
print("\nGenerating Grad-CAMs...")
# Pick one image per class from val
cls_indices = {i:[] for i in range(len(class_names))}
dset = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
indices = val_ds.indices if isinstance(val_ds, Subset) else range(len(val_ds))
for i, idx in enumerate(indices):
    _, l = dset[idx]
    if len(cls_indices[l]) < 1: cls_indices[l].append(i)

for cls_id, idxs in cls_indices.items():
    if not idxs: continue
    img, lbl = val_ds[idxs[0]]
    x = img.unsqueeze(0).to(device)
    cam = get_cam(model, x, lbl)
    if cam is not None:
        cam = cv2.resize(cam, (config["image_size"], config["image_size"]))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        orig = ((img.permute(1,2,0).cpu().numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255).astype(np.uint8)
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        out = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(f"{config['gradcam_out_dir']}/class_{class_names[lbl]}.jpg", out)
print("Done.")