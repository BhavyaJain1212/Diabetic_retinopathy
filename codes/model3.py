# train_dr_full.py
"""
Complete training script combining:
- Rotation augmentations + Rotation TTA
- MONAI transforms support (optional)
- Focal Loss + class weights
- Multi-kernel CBAM
- Refinement head
- WeightedRandomSampler for imbalance
- OneCycleLR scheduler + EarlyStopping
- Grad-CAM visualization saving to gradcam_out_dir

Keeps dataset paths and outputs as in original config:
    config["csv_path"], config["img_root"], config["gradcam_out_dir"]
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

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Config (change only here)
# -------------------------
config = {
    "csv_path": "train.csv",          # if missing, code will fallback to ImageFolder on img_root
    "img_root": "./diabetes/colored_images",     # folder root OR ImageFolder root (class subfolders)
    "backbone": "efficientnet_b2",
    "teacher_backbone": "efficientnet_b5",  # if use_distill True
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 25,
    "lr": 1e-4,
    "use_cbam": True,
    "use_distill": False,             # set True if you want teacher distillation
    "loss_fn": "focal",               # options: "crossentropy", "focal"
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_B5_Distill",
    "image_size": 224,
    "tta_rotations": [0, 90, 180, 270],  # TTA rotations (degrees)
    "use_mixed_precision": False,     # optional amp
    "num_workers": 4,
    "use_moniotransforms": False,     # set True to prefer MONAI if installed
    "imbalanced_sampler": True,       # use WeightedRandomSampler
    "seed": 42
}

# reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    if "NVIDIA" in device_name.upper():
        print(f"✅ Using NVIDIA GPU: {device_name}")
    else:
        print(f"⚠ Using GPU (not clearly NVIDIA): {device_name}")
else:
    print("⚠ No GPU found. Using CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Try import MONAI if user wants it
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
    # CSV dataset later will provide full file paths per sample
    train_transform = Compose([
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        ScaleIntensityD(keys=["image"]),  # scales to 0-1
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
    # torchvision transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),  # rotation augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# -------------------------
# CSV-based dataset class (fixed)
# -------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png", ".jpg", ".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        # normalize column names and pick first two columns as image and label if names unspecified
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))])
        # default mapping comes from your script
        self.numeric_to_folder = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR"}
        self.img_exts = img_exts

    def __len__(self):
        return len(self.data)

    def _find_image(self, folder, img_id):
        # try different extensions and direct path
        for ext in self.img_exts:
            p = os.path.join(self.img_root, folder, f"{img_id}{ext}")
            if os.path.exists(p):
                return p
        p = os.path.join(self.img_root, folder, img_id)
        if os.path.exists(p):
            return p
        # try if csv stored relative paths already
        p2 = os.path.join(self.img_root, img_id)
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"Image for id {img_id} not found in {folder} (tried {self.img_exts})")

    def __getitem__(self, idx):
        # read image id and label from CSV
        img_id = str(self.data.iloc[idx][self.image_col])
        label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val, (int, float)) or str(label_val).isdigit():
            label_val = int(label_val)
            # map numeric to folder name if folder mapping exists
            folder_name = self.numeric_to_folder.get(label_val, str(label_val))
            label_idx = int(label_val)
        else:
            folder_name = str(label_val)
            # fallback to folder index if folder names present
            if folder_name in self.folder_names:
                label_idx = self.folder_names.index(folder_name)
            else:
                # try parse last part of path
                label_idx = 0
        img_path = self._find_image(folder_name, img_id)
        image = Image.open(img_path).convert("RGB")
        if use_monai:
            # monai transforms expect dict-like with keys; we will forward through transform when used
            data = {"image": img_path, "label": label_idx}
            # the dataset consumer (train loop) will apply transform if necessary; but for simplicity we load and transform image here using torchvision transforms
            if self.transform:
                image = self.transform(image)
        else:
            if self.transform:
                image = self.transform(image)
        return image, label_idx

# -------------------------
# Build dataset: try CSV, else ImageFolder
# -------------------------
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
    # fallback to ImageFolder
    if os.path.isdir(config["img_root"]):
        # using torchvision ImageFolder. We'll set its transform later per split.
        dataset = datasets.ImageFolder(root=config["img_root"])
        use_imagefolder = True
        print("✅ Loaded dataset with ImageFolder (class-subfolders).")
    else:
        raise FileNotFoundError(f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist.")

# class names
if use_imagefolder:
    class_names = dataset.classes
else:
    # use the numeric_to_folder mapping order (ensures 5 classes)
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# -------------------------
# train/val split
# -------------------------
total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size = total_len - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config["seed"]))

# if using ImageFolder fallback, set transforms on underlying dataset
if use_imagefolder:
    # random_split returns Subset; set transform on the underlying dataset
    dataset.transform = None  # remove old
    # create two new ImageFolder-likes by wrapping underlying dataset with transform
    class WrappedImageFolder(torch.utils.data.Dataset):
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
    # original CSV dataset had transform baked in; adjust val_ds transform to val_transform
    # random_split returns Subset of RetinopathyDataset; set underlying dataset transform appropriately
    if isinstance(train_ds, Subset):
        train_ds.dataset.transform = train_transform
    if isinstance(val_ds, Subset):
        val_ds.dataset.transform = val_transform

# data loaders: possibly use WeightedRandomSampler to handle imbalance
def make_sampler_from_dataset(ds):
    # extract labels from subset/dataset
    labels = []
    if isinstance(ds, Subset):
        underlying = ds.dataset
        for i in ds.indices:
            try:
                if use_imagefolder:
                    # no CSV
                    _, lbl = underlying.samples[i]
                else:
                    lbl = underlying.data.iloc[i][underlying.label_col]
            except Exception:
                # fallback: try _getitem_
                item = ds._getitem_(0)
                lbl = item[1]
            labels.append(int(lbl))
    else:
        # ds is dataset object; iterate (careful with large datasets)
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(int(lbl))
    labels = np.array(labels)
    class_sample_count = np.array([ (labels == t).sum() for t in range(config["num_classes"]) ])
    # handle zero-count classes
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1.0 / class_sample_count
    samples_weight = weight[labels]
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    return sampler, class_sample_count

if config["imbalanced_sampler"]:
    try:
        sampler, class_counts = make_sampler_from_dataset(train_ds)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                                  num_workers=config["num_workers"])
        print("✅ Using WeightedRandomSampler for imbalance handling. Class counts (train subset):", class_counts)
    except Exception as e:
        print("⚠ Could not build sampler, falling back to shuffle dataloader:", e)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
else:
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

print(f"Dataset: {total_len} images → train: {len(train_ds)}, val: {len(val_ds)}")
print("Classes:", class_names)

# -------------------------
# CBAM Multi-kernel and RefinementHead
# -------------------------
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = torch.mean(x, dim=(2,3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(out).view(b, c, 1, 1)

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
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            outs.append(torch.sigmoid(bn(conv(cat))))
        mask = sum(outs) / len(outs)
        return mask

class CBAM_MultiK(nn.Module):
    def __init__(self, channels, reduction=16, kernel_sizes=(3,5,7)):
        super().__init__()
        self.cg = ChannelGate(channels, reduction)
        self.sg = SpatialGateMultiK(kernel_sizes)
    def forward(self, x):
        x = x * self.cg(x)
        x = x * self.sg(x)
        return x

class RefinementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels//2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid_channels//2, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        p = self.pool(x).flatten(1)
        return self.fc(p)

# -------------------------
# Model combining EfficientNet features + CBAM + Refinement head
# -------------------------
class EfficientNetWithRefineCBAM(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2', num_classes=5, use_cbam=True, pretrained=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAM_MultiK(out_channels) if use_cbam else None
        self.refine = RefinementHead(out_channels, mid_channels=256, num_classes=num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if self.cbam:
            feats = self.cbam(feats)
        logits = self.refine(feats)
        return logits

# -------------------------
# Focal Loss implementation
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-9):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (N,C), targets: (N,)
        logpt = torch.nn.functional.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt_true = logpt[range(inputs.size(0)), targets]
        pt_true = pt[range(inputs.size(0)), targets]
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            loss = - at * ((1 - pt_true) ** self.gamma) * logpt_true
        else:
            loss = - ((1 - pt_true) ** self.gamma) * logpt_true
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -------------------------
# Optional: Distillation Loss (kept simple)
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher_model=None, alpha=0.5, temperature=4.0):
        super().__init__()
        self.base_loss = base_loss
        self.teacher = teacher_model
        self.alpha = alpha
        self.T = temperature
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, student_logits, inputs, labels):
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None:
            return hard
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits / self.T, dim=1)
        t_prob = nn.functional.softmax(teacher_logits / self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T * self.T)
        return self.alpha * soft + (1.0 - self.alpha) * hard

# -------------------------
# Initialize model, loss, optimizer, scheduler
# -------------------------
teacher_model = None
if config["use_distill"]:
    print("Loading teacher model for distillation...")
    teacher_model = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device).eval()

# compute class weights from train_ds to optionally pass to loss
def compute_class_counts(ds):
    labels = []
    if isinstance(ds, Subset):
        underlying = ds.dataset
        for i in ds.indices:
            _, lbl = underlying.getitem_(i) if not use_imagefolder else underlying[i]
            labels.append(int(lbl))
    else:
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(int(lbl))
    counts = np.bincount(np.array(labels), minlength=config["num_classes"])
    counts = np.where(counts == 0, 1, counts)
    return counts

try:
    class_counts = compute_class_counts(train_ds)
    class_weights = class_counts.max() / class_counts
    class_weights = class_weights.astype(float)
    print("Class counts (train subset):", class_counts)
    print("Computed class_weights (inverse freq normalized by max):", class_weights)
except Exception as e:
    print("⚠ Could not compute class counts:", e)
    class_weights = np.ones(config["num_classes"], dtype=float)

# choose base loss
loss_name = config["loss_fn"].lower()
if loss_name in ("crossentropyloss", "crossentropy", "cross"):
    base_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
elif loss_name in ("focal", "focalloss"):
    alpha = (class_weights / class_weights.sum()).tolist()  # normalized alpha for focal
    base_loss = FocalLoss(gamma=2.0, alpha=alpha)
else:
    raise ValueError(f"Unsupported loss function: {config['loss_fn']}")

criterion = DistillationLoss(base_loss, teacher_model) if config["use_distill"] else base_loss

model = EfficientNetWithRefineCBAM(config["backbone"], config["num_classes"], use_cbam=config["use_cbam"], pretrained=True).to(device)

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
if len(train_loader) > 0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"] * 10,
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"]
    )
else:
    scheduler = None

# Trainer helper
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, tta_rots=None, use_amp=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tta_rots = tta_rots if tta_rots is not None else [0]
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train_epoch(self, loader, epoch_idx=0):
        self.model.train()
        running_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, imgs, labels) if isinstance(self.criterion, DistillationLoss) else self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, imgs, labels) if isinstance(self.criterion, DistillationLoss) else self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # one-cycle step
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            running_loss += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        return running_loss / max(1, n)

    def predict_batch_tta(self, imgs):
        # imgs: (B,C,H,W) on device
        logits_sum = None
        for ang in self.tta_rots:
            k = (ang // 90) % 4
            if k != 0:
                imgs_rot = torch.rot90(imgs, k=k, dims=[2,3])
            else:
                imgs_rot = imgs
            out = self.model(imgs_rot)
            if logits_sum is None:
                logits_sum = out
            else:
                logits_sum = logits_sum + out
        logits_avg = logits_sum / len(self.tta_rots)
        return logits_avg

    def evaluate(self, loader, tta=True):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if tta:
                    outputs = self.predict_batch_tta(imgs)
                else:
                    outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        return preds, trues

trainer = Trainer(model, criterion, optimizer, scheduler, device, tta_rots=config["tta_rotations"], use_amp=config["use_mixed_precision"])

# -------------------------
# Training loop + early stopping
# -------------------------
train_losses = []
best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader, epoch)
    train_losses.append(train_loss)

    # validation
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = trainer.predict_batch_tta(imgs)
            loss = criterion(outputs, imgs, labels) if isinstance(criterion, DistillationLoss) else criterion(outputs, labels)
            val_loss += float(loss.item()) * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    val_loss /= max(1, total)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # early stopping + save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Validation loss improved — model saved.")
    else:
        patience_counter += 1
        print(f"⚠ No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered!")
            break

    # classification report each epoch
    try:
        print(classification_report(trues, preds, digits=4, target_names=class_names))
    except Exception:
        pass

# -------------------------
# Plot training loss
# -------------------------
try:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
except Exception:
    pass

# -------------------------
# Final evaluation and confusion matrix
# -------------------------
preds, trues = trainer.evaluate(val_loader, tta=True)
print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))

cm = confusion_matrix(trues, preds)
try:
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
except Exception:
    pass

# -------------------------
# Self-contained Grad-CAM implementation (tries to find last Conv2d)
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
    """
    Returns normalized heatmap (H x W) in range [0,1]
    input_tensor: (1,C,H,W) on device
    """
    model.eval()
    # find last conv
    last_name, last_conv = find_last_conv_module(model)
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model for Grad-CAM.")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    # forward
    outputs = model(input_tensor)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    # get grad and activation
    act = activations['value']          # shape (1, C, H, W)
    grad = gradients['value']           # shape (1, C, H, W)
    # global-average-pool the gradients to get weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)  # (1,1,H,W)
    cam_map = torch.relu(cam_map)
    cam_map = cam_map.squeeze().cpu().numpy()  # (H, W)

    # normalize
    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    # remove hooks
    fh.remove()
    bh.remove()
    return cam_map  # numpy HxW in [0,1]

# -------------------------
# Save Grad-CAMs: pick one random val image per class
# -------------------------
os.makedirs(config["gradcam_out_dir"], exist_ok=True)
if os.path.exists("best_model.pth"):
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("✅ Loaded best_model.pth for Grad-CAM generation.")
    except Exception as e:
        print("⚠ Could not load best_model.pth:", e)
else:
    print("⚠ No best_model.pth found — using current model weights for Grad-CAM.")

model.to(device)
model.eval()

# build mapping of class->indices in validation set
indices_per_class = {i: [] for i in range(len(class_names))}
# val_ds may be Subset or custom wrapper
if isinstance(val_ds, Subset):
    underlying_dataset = val_ds.dataset
    for local_idx, global_idx in enumerate(val_ds.indices):
        try:
            img, lbl = val_ds[local_idx]
        except Exception:
            img, lbl = underlying_dataset[global_idx]
        indices_per_class[lbl].append(local_idx)
else:
    for i in range(len(val_ds)):
        _, lbl = val_ds[i]
        indices_per_class[lbl].append(i)

# inverse normalization helper
inv_mean = np.array([0.485, 0.456, 0.406])
inv_std = np.array([0.229, 0.224, 0.225])

print("\n🎯 Generating Grad-CAMs for one random validation image per class...")
saved_count = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs:
        print(f"⚠ No samples for class {cls_name} in validation set; skipping.")
        continue
    sel_local_idx = random.choice(idxs)
    img_tensor, lbl = val_ds[sel_local_idx]  # returns normalized tensor
    # ensure CPU tensor for preprocessing if it's minibatched
    if isinstance(img_tensor, torch.Tensor):
        input_tensor = img_tensor.unsqueeze(0).to(device)
    else:
        # if using MONAI transforms that produced numpy arrays
        input_arr = np.array(img_tensor)
        if input_arr.ndim == 3 and input_arr.shape[0] == 3:
            input_tensor = torch.tensor(input_arr).unsqueeze(0).to(device).float()
        elif input_arr.ndim == 3 and input_arr.shape[2] == 3:
            # HWC
            input_tensor = torch.tensor(input_arr).permute(2,0,1).unsqueeze(0).to(device).float()
        else:
            raise RuntimeError("Unsupported image shape for Grad-CAM.")
    target_class = int(lbl)
    try:
        cam = generate_gradcam(model, input_tensor, target_class, device)  # HxW float [0,1]
    except Exception as e:
        print(f"⚠ Grad-CAM failed for class {cls_name}: {e}")
        continue

    # resize cam to image size
    cam_resized = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR

    # reconstruct original image (unnormalize) if tensor
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * inv_std) + inv_mean  # un-normalize
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        # if img_tensor was numpy HWC in 0-1
        arr = img_tensor
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # overlay
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(config["gradcam_out_dir"], f"{cls_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Saved Grad-CAM for class '{cls_name}' → {out_path}")
    saved_count += 1

if saved_count == 0:
    print("⚠ No Grad-CAMs were saved.")
else:
    print(f"\n🎉 Saved {saved_count} Grad-CAM(s) in '{config['gradcam_out_dir']}'")

# End of script
