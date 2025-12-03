"""
Complete DR Training Script with Hybrid Model
Combines:
- Hybrid EfficientNet + CBAM (vanilla/multi-kernel) + Dual-Head Refinement from both prior scripts
- CSV/ImageFolder dataset loading
- Focal Loss or CrossEntropy with class weights / label smoothing
- Optional distillation (with alpha scheduling)
- Rotation consistency loss (optional)
- WeightedRandomSampler for imbalance
- Rotation TTA for evaluation
- OneCycleLR scheduler + Early Stopping
- Mixed precision (optional)
- MONAI transforms support (optional)
- Grad-CAM visualization (last conv hook)
- Plots: loss curve, confusion matrix

Config-driven; set use_distill=False by default for simplicity.
Dataset: Assumes colored_images/ with class folders or train.csv (id, label).
Classes: No_DR, Mild, Moderate, Severe, Proliferate_DR (5-class ICDR).
"""

import os
import random
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import timm
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Config (change only here)
# -------------------------
config = {
    "csv_path": "train.csv",                    # if missing, fallback to ImageFolder on img_root
    "img_root": "./diabetes/colored_images",    # folder root OR ImageFolder root (class subfolders)
    "backbone": "efficientnet_b2",
    "teacher_backbone": "efficientnet_b5",      # if use_distill True
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 25,
    "lr": 1e-4,
    "use_cbam": True,                           # enable CBAM (vanilla or multi-kernel)
    "cbam_type": "multi",                       # "vanilla" or "multi"
    "use_distill": False,                       # set True for teacher distillation
    "train_teacher": True,                      # pretrain/fine-tune teacher if use_distill
    "teacher_epochs": 5,                        # teacher epochs if train_teacher True
    "distill_alpha_start": 0.2,                 # initial distill loss weight
    "distill_alpha_end": 0.8,                   # final distill loss weight (linear schedule)
    "distill_temperature": 4.0,
    "loss_fn": "focal",                         # "crossentropy" or "focal"
    "use_class_weights": True,                  # for imbalance in loss
    "label_smoothing": 0.1,                     # for CrossEntropy (0 to disable)
    "consistency_weight": 0.1,                  # rotation consistency loss weight (0 to disable)
    "rotation_angles": [0, 90, 180, 270],       # for consistency and TTA
    "early_stop_patience": 5,
    "gradcam_out_dir": "Hybrid_B2_B5_GradCAM",
    "image_size": 224,
    "tta_rotations": [0, 90, 180, 270],         # TTA angles
    "use_mixed_precision": False,               # AMP for speed/memory
    "num_workers": 4,
    "use_moniotransforms": False,               # MONAI if installed (optional)
    "imbalanced_sampler": True,                 # WeightedRandomSampler
    "fusion": "avg",                            # "sum" | "avg" | "concat" | "refine_only" | "pool_only"
    "dropout": 0.2,                             # head dropout
    "seed": 42,
    "max_grad_norm": 5.0,                       # gradient clipping
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

# -------------------------
# Device
# -------------------------
device = torch.device(config["device"])
if device.type == "cuda":
    try:
        device_name = torch.cuda.get_device_name(0)
        if "NVIDIA" in device_name.upper():
            print(f"✅ Using NVIDIA GPU: {device_name}")
        else:
            print(f"⚠ Using GPU (not clearly NVIDIA): {device_name}")
    except Exception:
        print("✅ Using GPU")
else:
    print("⚠ No GPU found. Using CPU.")

# -------------------------
# Try MONAI if requested
# -------------------------
use_monai = False
if config["use_moniotransforms"]:
    try:
        import monai
        from monai.transforms import (
            Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityD,
            ResizeD, RandFlipD, RandRotateD, RandZoomD, ToTensorD
        )
        use_monai = True
        print("✅ MONAI available and will be used for transforms.")
    except Exception as e:
        print("⚠ MONAI requested but not available. Falling back to torchvision transforms.")
        use_monai = False

# -------------------------
# Transforms
# -------------------------
if use_monai:
    train_transform = Compose([
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        ScaleIntensityD(keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(config["image_size"], config["image_size"])),
        RandFlipD(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipD(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotateD(keys=["image"], range_x=np.pi/6, prob=0.5),
        RandZoomD(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
        ToTensorD(keys=["image"])
    ])
    val_transform = Compose([
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        ScaleIntensityD(keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(config["image_size"], config["image_size"])),
        ToTensorD(keys=["image"])
    ])
else:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# -------------------------
# Dataset Class (CSV or ImageFolder fallback)
# -------------------------
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
            if os.path.exists(p):
                return p
        p = os.path.join(self.img_root, folder, img_id)
        if os.path.exists(p):
            return p
        p2 = os.path.join(self.img_root, img_id)
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"Image for id {img_id} not found in {folder} (tried {self.img_exts})")

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx][self.image_col])
        label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val, (int, float)) or str(label_val).isdigit():
            label_val = int(label_val)
            folder_name = self.numeric_to_folder.get(label_val, str(label_val))
            label_idx = label_val
        else:
            folder_name = str(label_val)
            if folder_name in self.folder_names:
                label_idx = self.folder_names.index(folder_name)
            else:
                label_idx = 0
        img_path = self._find_image(folder_name, img_id)
        if use_monai:
            return {"image": img_path, "label": label_idx}  # MONAI dict
        else:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label_idx

# Build dataset
dataset = None
use_imagefolder = False
if os.path.exists(config["csv_path"]) and os.path.isdir(config["img_root"]):
    try:
        dataset = RetinopathyDataset(config["csv_path"], config["img_root"])
        print("✅ Loaded dataset from CSV + folders.")
    except Exception as e:
        print(f"⚠ Failed to load CSV-based dataset: {e}")
        dataset = None

if dataset is None:
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(root=config["img_root"])
        use_imagefolder = True
        print("✅ Loaded dataset with ImageFolder (class-subfolders).")
    else:
        raise FileNotFoundError(f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist.")

# Class names
if use_imagefolder:
    class_names = dataset.classes
else:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# Train/val split
total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size = total_len - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config["seed"]))

# Adjust transforms for splits
if use_imagefolder:
    class WrappedImageFolder(Dataset):
        def __init__(self, base_ds, indices, transform):
            self.base = base_ds
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            path, label = self.base.samples[real_idx]
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    train_ds = WrappedImageFolder(dataset, train_ds.indices, transform=train_transform)
    val_ds = WrappedImageFolder(dataset, val_ds.indices, transform=val_transform)
else:
    if use_monai:
        # MONAI: transforms applied in loop via data dict
        pass
    else:
        # For CSV: recreate with transforms
        train_ds.dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=train_transform)
        val_ds.dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform)

# -------------------------
# Sampler for imbalance
# -------------------------
def make_sampler(ds):
    labels = []
    if isinstance(ds, Subset):
        underlying = ds.dataset
        for i in ds.indices:
            if use_imagefolder:
                _, lbl = underlying[i]
            else:
                _, lbl = underlying[i]
            labels.append(int(lbl))
    else:
        for i in range(min(1000, len(ds))):  # Sample for large sets
            _, lbl = ds[i]
            labels.append(int(lbl))
    class_sample_count = np.bincount(np.array(labels), minlength=config["num_classes"])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[l] for l in labels])
    return WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

# DataLoaders
if config["imbalanced_sampler"]:
    try:
        sampler = make_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"], pin_memory=True)
        print("✅ Using WeightedRandomSampler. Class counts:", np.bincount(labels, minlength=config["num_classes"]))
    except Exception as e:
        print(f"⚠ Sampler failed: {e}. Using shuffle.")
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
else:
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

print(f"Dataset: {total_len} images → train: {len(train_ds)}, val: {len(val_ds)}")
print("Classes:", class_names)

# -------------------------
# Hybrid Model Components
# -------------------------
# CBAM
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = torch.mean(x, dim=(2,3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(out).view(b, c, 1, 1)

class SpatialGateVanilla(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        out = self.bn(self.conv(cat))
        return torch.sigmoid(out)

class SpatialGateMultiK(nn.Module):
    def __init__(self, kernel_sizes=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k-1)//2
            self.convs.append(nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False))
            self.bns.append(nn.BatchNorm2d(1))
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        outs = [torch.sigmoid(bn(conv(cat))) for conv, bn in zip(self.convs, self.bns)]
        return sum(outs) / len(outs)

class CBAM_Vanilla(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.cg = ChannelGate(channels, reduction)
        self.sg = SpatialGateVanilla(kernel_size)
    def forward(self, x):
        x = x * self.cg(x)
        x = x * self.sg(x)
        return x

class CBAM_MultiK(nn.Module):
    def __init__(self, channels, reduction=16, kernel_sizes=(3,5,7)):
        super().__init__()
        self.cg = ChannelGate(channels, reduction)
        self.sg = SpatialGateMultiK(kernel_sizes)
    def forward(self, x):
        x = x * self.cg(x)
        x = x * self.sg(x)
        return x

# Refinement Head
class RefinementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256, num_classes=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(mid_channels//2, num_classes)
        self.last_conv = self.conv2  # For Grad-CAM

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        vec = self.pool(x).flatten(1)
        vec = self.drop(vec)
        logits = self.fc(vec)
        return logits, vec

# Hybrid Model
class HybridEfficientNetCBAMRefine(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_b2',
                 num_classes=5,
                 pretrained=True,
                 use_cbam=True,
                 cbam_type='multi',
                 cbam_reduction=16,
                 cbam_kernel_sizes=(3,5,7),
                 refine_mid_channels=256,
                 fusion='avg',
                 dropout=0.2):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=True
            )
        out_ch = self.feature_extractor.feature_info[-1]['num_chs']

        self.use_cbam = use_cbam
        if use_cbam:
            if cbam_type == 'multi':
                self.cbam = CBAM_MultiK(out_ch, reduction=cbam_reduction, kernel_sizes=cbam_kernel_sizes)
            else:
                self.cbam = CBAM_Vanilla(out_ch, reduction=cbam_reduction, kernel_size=7)
        else:
            self.cbam = nn.Identity()

        self.refine = RefinementHead(out_ch, mid_channels=refine_mid_channels, num_classes=num_classes, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc_pool = nn.Linear(out_ch, num_classes)

        self.fusion = fusion
        if fusion == 'concat':
            concat_in = (refine_mid_channels // 2) + out_ch
            self.fc_concat = nn.Linear(concat_in, num_classes)

    def forward(self, x, return_dict=False):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.cbam(feats)

        logits_r, vec_r = self.refine(feats)
        vec_p = self.pool(feats).flatten(1)
        vec_p = self.drop(vec_p)
        logits_p = self.fc_pool(vec_p)

        if self.fusion == 'sum':
            logits = logits_r + logits_p
        elif self.fusion == 'avg':
            logits = 0.5 * (logits_r + logits_p)
        elif self.fusion == 'concat':
            z = torch.cat([vec_r, vec_p], dim=1)
            logits = self.fc_concat(z)
        elif self.fusion == 'refine_only':
            logits = logits_r
        elif self.fusion == 'pool_only':
            logits = logits_p
        else:
            raise ValueError(f"Unsupported fusion: {self.fusion}")

        if return_dict:
            return {
                "logits": logits, "logits_refine": logits_r, "logits_pool": logits_p,
                "vec_refine": vec_r, "vec_pool": vec_p, "feats": feats
            }
        return logits

# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-9):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt_true = logpt[range(inputs.size(0)), targets]
        pt_true = pt[range(inputs.size(0)), targets]
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            loss = -at * ((1 - pt_true) ** self.gamma) * logpt_true
        else:
            loss = -((1 - pt_true) ** self.gamma) * logpt_true
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher_model=None, alpha_start=0.5, alpha_end=0.5, temperature=4.0):
        super().__init__()
        self.base_loss = base_loss
        self.teacher = teacher_model
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.T = temperature
        self.current_epoch = 0
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def set_epoch(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def get_alpha(self):
        if self.max_epochs <= 1:
            return self.alpha_end
        frac = min(1.0, max(0.0, self.current_epoch / (self.max_epochs - 1)))
        return self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def forward(self, student_logits, inputs, labels):
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None:
            return hard
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        s_logp = F.log_softmax(student_logits / self.T, dim=1)
        t_prob = F.softmax(teacher_logits / self.T, dim=1)
        soft = F.kl_div(s_logp, t_prob, reduction='batchmean') * (self.T * self.T)
        alpha = self.get_alpha()
        return alpha * soft + (1 - alpha) * hard

# Compute class weights
def compute_class_counts(ds):
    labels = []
    if isinstance(ds, Subset):
        for i in ds.indices:
            _, lbl = ds.dataset[i] if hasattr(ds.dataset, '__getitem__') else ds[i]
            labels.append(int(lbl))
    else:
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(int(lbl))
    counts = np.bincount(np.array(labels), minlength=config["num_classes"])
    return np.where(counts == 0, 1, counts)

class_counts = compute_class_counts(train_ds)
class_weights = 1.0 / (class_counts / class_counts.sum() + 1e-6)
class_weights = class_weights / class_weights.mean()  # Normalize
print("Class counts:", class_counts)
print("Class weights:", class_weights)

# Base loss
loss_name = config["loss_fn"].lower()
if loss_name in ("crossentropyloss", "crossentropy"):
    if config["label_smoothing"] > 0:
        base_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device), label_smoothing=config["label_smoothing"])
    else:
        base_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
elif loss_name in ("focal", "focalloss"):
    alpha_focal = (class_weights / class_weights.sum()).tolist()
    base_loss = FocalLoss(gamma=2.0, alpha=alpha_focal)
else:
    raise ValueError(f"Unsupported loss: {config['loss_fn']}")

# Final criterion
teacher_model = None
if config["use_distill"]:
    teacher_model = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
    print(f"✅ Teacher created: {config['teacher_backbone']}")
criterion = DistillationLoss(base_loss, teacher_model, config["distill_alpha_start"], config["distill_alpha_end"], config["distill_temperature"]) if config["use_distill"] else base_loss

# Student model (hybrid)
model = HybridEfficientNetCBAMRefine(
    backbone_name=config["backbone"],
    num_classes=config["num_classes"],
    pretrained=True,
    use_cbam=config["use_cbam"],
    cbam_type=config["cbam_type"],
    fusion=config["fusion"],
    dropout=config["dropout"]
).to(device)
print(f"✅ Hybrid student created: {config['backbone']} + {config['cbam_type']} CBAM + {config['fusion']} fusion")

# Teacher training (if enabled)
if config["use_distill"] and config["train_teacher"]:
    print("\n🎓 Training teacher...")
    teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=config["lr"] * 0.5)
    teacher_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        teacher_optimizer, max_lr=config["lr"], steps_per_epoch=len(train_loader), epochs=config["teacher_epochs"]
    )
    teacher_trainer = Trainer(teacher_model, base_loss, teacher_optimizer, teacher_scheduler, device, config)
    best_teacher_loss = float("inf")
    for t_epoch in range(config["teacher_epochs"]):
        t_loss = teacher_trainer.train_epoch(train_loader, t_epoch, config["teacher_epochs"])
        # Val loss
        vloss = 0.0
        total = 0
        teacher_model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = teacher_model(imgs)
                loss = base_loss(outputs, labels)
                vloss += loss.item() * imgs.size(0)
                total += imgs.size(0)
        vloss /= max(1, total)
        print(f"Teacher Epoch {t_epoch+1}/{config['teacher_epochs']} - Train: {t_loss:.4f}, Val: {vloss:.4f}")
        if vloss < best_teacher_loss:
            best_teacher_loss = vloss
            torch.save(teacher_model.state_dict(), "best_teacher.pth")
    if os.path.exists("best_teacher.pth"):
        teacher_model.load_state_dict(torch.load("best_teacher.pth", map_location=device))
        criterion.teacher = teacher_model
    print("🎓 Teacher complete.")

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=config["lr"] * 5, steps_per_epoch=len(train_loader), epochs=config["epochs"]
) if len(train_loader) > 0 else None

# -------------------------
# Trainer Class
# -------------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config, use_amp=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader, epoch_idx=0, total_epochs=1):
        self.model.train()
        running_loss = 0.0
        n = 0
        for batch in loader:
            if use_monai:
                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
            else:
                imgs, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    if isinstance(self.criterion, DistillationLoss):
                        self.criterion.set_epoch(epoch_idx, total_epochs)
                        loss = self.criterion(outputs, imgs, labels)
                    else:
                        loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.config.get("max_grad_norm"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs)
                if isinstance(self.criterion, DistillationLoss):
                    self.criterion.set_epoch(epoch_idx, total_epochs)
                    loss = self.criterion(outputs, imgs, labels)
                else:
                    loss = self.criterion(outputs, labels)

                # Consistency loss
                if self.config.get("consistency_weight", 0.0) > 0.0:
                    angle = random.choice(self.config["rotation_angles"])
                    if angle % 360 != 0:
                        rotated_imgs = torch.stack([transforms.functional.rotate(img.cpu(), angle) for img in imgs]).to(self.device)
                    else:
                        rotated_imgs = imgs
                    with torch.no_grad():
                        out_orig = F.log_softmax(outputs, dim=1)
                    out_rot = self.model(rotated_imgs)
                    out_rot_logp = F.log_softmax(out_rot, dim=1)
                    p_orig = F.softmax(outputs.detach(), dim=1)
                    p_rot = F.softmax(out_rot.detach(), dim=1)
                    kl1 = F.kl_div(out_orig, p_rot, reduction='batchmean')
                    kl2 = F.kl_div(out_rot_logp, p_orig, reduction='batchmean')
                    consistency_loss = 0.5 * (kl1 + kl2)
                    loss += self.config["consistency_weight"] * consistency_loss

                loss.backward()
                if self.config.get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return running_loss / max(1, n)

    def predict_batch_tta(self, imgs):
        self.model.eval()
        logits_sum = None
        for ang in self.config["tta_rotations"]:
            k = (ang // 90) % 4
            if k != 0:
                imgs_rot = torch.rot90(imgs, k=k, dims=[2,3])
            else:
                imgs_rot = imgs
            out = self.model(imgs_rot)
            logits_sum = out if logits_sum is None else logits_sum + out
        return logits_sum / len(self.config["tta_rotations"])

    def evaluate(self, loader, tta=True):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                if use_monai:
                    imgs = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                else:
                    imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
                if tta:
                    outputs = self.predict_batch_tta(imgs)
                else:
                    outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        return preds, trues

# Instantiate trainer
trainer = Trainer(
    model, criterion, optimizer, scheduler, device, config,
    use_amp=config["use_mixed_precision"]
)

# -------------------------
# Training Loop
# -------------------------
train_losses = []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader, epoch, config["epochs"])
    train_losses.append(train_loss)

    # Validation
    preds, trues = trainer.evaluate(val_loader, tta=True)
    val_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if use_monai:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                imgs, labels = batch[0].to(device), batch[1].to(device)
            outputs = trainer.predict_batch_tta(imgs)
            if isinstance(criterion, DistillationLoss):
                criterion.set_epoch(epoch, config["epochs"])
                loss = criterion(outputs, imgs, labels)
            else:
                loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
    val_loss /= max(1, total)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    print(classification_report(trues, preds, digits=4, target_names=class_names))

    # Early stopping & save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved best model.")
    else:
        patience_counter += 1
        print(f"⚠ No improvement ({patience_counter}/{config['early_stop_patience']})")
        if patience_counter >= config["early_stop_patience"]:
            print("🛑 Early stopping.")
            break

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Final eval
preds, trues = trainer.evaluate(val_loader, tta=True)
print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------------
# Grad-CAM
# -------------------------
def find_last_conv_module(net):
    last_name = None
    last_mod = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_mod = module
    return last_name, last_mod

def generate_gradcam(model, input_tensor, target_class, device):
    model.eval()
    last_name, last_conv = find_last_conv_module(model)
    if last_conv is None:
        raise RuntimeError("No Conv2d found for Grad-CAM.")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    outputs = model(input_tensor)
    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations['value']
    grad = gradients['value']
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)
    cam_map = F.relu(cam_map)
    cam_map = cam_map.squeeze().cpu().numpy()

    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    fh.remove()
    bh.remove()
    return cam_map

# Generate Grad-CAMs
os.makedirs(config["gradcam_out_dir"], exist_ok=True)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("✅ Loaded best model for Grad-CAM.")

model.eval()

# Indices per class
indices_per_class = {i: [] for i in range(config["num_classes"])}
for i in range(len(val_ds)):
    _, lbl = val_ds[i]
    indices_per_class[int(lbl)].append(i)

inv_mean = np.array([0.485, 0.456, 0.406])
inv_std = np.array([0.229, 0.224, 0.225])

print("\n🎯 Generating Grad-CAMs...")
saved_count = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs:
        print(f"⚠ Skipping {cls_name}: no val samples.")
        continue
    sel_idx = random.choice(idxs)
    if use_monai:
        sample = val_transform({"image": val_ds[sel_idx]["image"]})
        img_tensor = sample["image"].unsqueeze(0).to(device)
        lbl = sample["label"]
    else:
        img_tensor, lbl = val_ds[sel_idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

    target_class = int(lbl)
    try:
        cam = generate_gradcam(model, input_tensor, target_class, device)
    except Exception as e:
        print(f"⚠ Grad-CAM failed for {cls_name}: {e}")
        continue

    cam_resized = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * inv_std) + inv_mean
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(config["gradcam_out_dir"], f"{cls_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Saved {cls_name} Grad-CAM: {out_path}")
    saved_count += 1

print(f"\n🎉 Saved {saved_count} Grad-CAMs in {config['gradcam_out_dir']}.")

