# train_with_cbam.py
"""
CBAM-only training script
- EfficientNet-B2 backbone (timm)
- Inserts Multi-Kernel CBAM before classifier
- Rest of pipeline mirrors train_baseline.py (transforms, split, scheduler, early stopping, Grad-CAM)
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
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
# Config (edit here)
# -------------------------
config = {
    "csv_path": "train.csv",
    "img_root": "./diabetes/colored_images",
    "backbone": "efficientnet_b2",
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_CBAM_GradCAM",
    "image_size": 224,
    "tta_rotations": [0],
    "num_workers": 4,
    "seed": 42
}

# reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {name}")
    except Exception:
        print("✅ Using GPU")
else:
    print("⚠ Using CPU")

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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
# CSV-based dataset class
# -------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png", ".jpg", ".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))]) \
            if os.path.isdir(img_root) else []
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
        raise FileNotFoundError(f"Image for id {img_id} not found (tried folders and extensions)")

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx][self.image_col])
        label_val = self.data.iloc[idx][self.label_col]
        if isinstance(label_val, (int, float)) or str(label_val).isdigit():
            label_val = int(label_val)
            folder_name = self.numeric_to_folder.get(label_val, str(label_val))
            label_idx = int(label_val)
        else:
            folder_name = str(label_val)
            label_idx = self.folder_names.index(folder_name) if folder_name in self.folder_names else 0
        img_path = self._find_image(folder_name, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# -------------------------
# Build dataset
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
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(root=config["img_root"], transform=train_transform)
        use_imagefolder = True
        print("✅ Loaded dataset with ImageFolder (class-subfolders).")
    else:
        raise FileNotFoundError(f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist.")

# class names
if use_imagefolder:
    class_names = dataset.classes
else:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# -------------------------
# Train/Val split
# -------------------------
total_len = len(dataset)
train_len = int(0.8 * total_len)
val_len = total_len - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config["seed"]))

# if ImageFolder, wrap subsets to apply transforms correctly
if use_imagefolder:
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
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

print(f"Dataset: {total_len} images -> train: {len(train_ds)}, val: {len(val_ds)}")
print("Classes:", class_names)

# -------------------------
# CBAM Multi-kernel implementation
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

# -------------------------
# Model combining EfficientNet + CBAM + classifier
# -------------------------
class EfficientNetWithCBAM(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2', num_classes=5, use_cbam=True, pretrained=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAM_MultiK(out_channels) if use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if self.cbam:
            feats = self.cbam(feats)
        pooled = self.pool(feats).flatten(1)
        logits = self.classifier(pooled)
        return logits

model = EfficientNetWithCBAM(config["backbone"], config["num_classes"], use_cbam=True, pretrained=True).to(device)

# -------------------------
# Loss, optimizer, scheduler
# -------------------------
def compute_class_counts(ds):
    labels = []
    for i in range(len(ds)):
        _, lbl = ds[i]
        labels.append(int(lbl))
    counts = np.bincount(np.array(labels), minlength=config["num_classes"])
    counts = np.where(counts == 0, 1, counts)
    return counts

try:
    counts = compute_class_counts(train_ds)
    print("Train class counts:", counts)
    class_weights = torch.tensor(counts.max() / counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using class-weighted CrossEntropyLoss.")
except Exception:
    criterion = nn.CrossEntropyLoss()
    print("Using plain CrossEntropyLoss.")

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
if len(train_loader) > 0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"]*10, epochs=config["epochs"], steps_per_epoch=len(train_loader))
else:
    scheduler = None

# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, tta_rots=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tta_rots = tta_rots if tta_rots is not None else [0]

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            running_loss += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        return running_loss / max(1, n)

    def predict_batch(self, imgs):
        return self.model(imgs)

    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.predict_batch(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        return preds, trues

trainer = Trainer(model, criterion, optimizer, scheduler, device, tta_rots=config["tta_rotations"])

# -------------------------
# Training loop + early stopping
# -------------------------
best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]
train_losses = []

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = trainer.predict_batch(imgs)
            loss = criterion(outputs, labels)
            val_loss += float(loss.item()) * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    val_loss = val_loss / max(1, total)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Validation loss improved — saved best_model.pth")
    else:
        patience_counter += 1
        print(f"⚠ No improvement for {patience_counter} epoch(s)")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    try:
        print(classification_report(trues, preds, digits=4, target_names=class_names))
    except Exception:
        pass

# -------------------------
# Plot training loss
# -------------------------
try:
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_cbam.png")
    plt.close()
except Exception:
    pass

# -------------------------
# Final evaluation and confusion matrix
# -------------------------
preds, trues = trainer.evaluate(val_loader)
print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))

cm = confusion_matrix(trues, preds)
try:
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_cbam.png")
    plt.close()
except Exception:
    pass

# -------------------------
# Grad-CAM (same method as baseline)
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
        raise RuntimeError("No Conv2d layer for Grad-CAM.")
    activations = {}
    gradients = {}
    def fwd_hook(m, i, o):
        activations['value'] = o.detach()
    def bwd_hook(m, gi, go):
        gradients['value'] = go[0].detach()
    fh = last_conv.register_forward_hook(fwd_hook)
    bh = last_conv.register_backward_hook(bwd_hook)
    outputs = model(input_tensor)
    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    act = activations['value']
    grad = gradients['value']
    weights = torch.mean(grad, dim=(2,3), keepdim=True)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)
    cam_map = torch.relu(cam_map).squeeze().cpu().numpy()
    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()
    fh.remove()
    bh.remove()
    return cam_map

os.makedirs(config["gradcam_out_dir"], exist_ok=True)
if os.path.exists("best_model.pth"):
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("✅ Loaded best_model.pth for Grad-CAM")
    except Exception as e:
        print("⚠ Could not load best_model.pth:", e)
else:
    print("⚠ No best_model.pth found — using current model weights for Grad-CAM generation")

model.to(device)
model.eval()

# mapping class->indices in validation set
indices_per_class = {i: [] for i in range(len(class_names))}
if isinstance(val_ds, Subset):
    underlying = val_ds.dataset
    for local_idx, global_idx in enumerate(val_ds.indices):
        try:
            img, lbl = val_ds[local_idx]
        except Exception:
            img, lbl = underlying[global_idx]
        indices_per_class[lbl].append(local_idx)
else:
    for i in range(len(val_ds)):
        _, lbl = val_ds[i]
        indices_per_class[lbl].append(i)

inv_mean = np.array([0.485, 0.456, 0.406])
inv_std = np.array([0.229, 0.224, 0.225])

print("\nGenerating Grad-CAMs for one random validation image per class...")
saved = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs:
        print(f"⚠ No samples for class {cls_name}; skipping.")
        continue
    sel_idx = random.choice(idxs)
    img_tensor, lbl = val_ds[sel_idx]
    if isinstance(img_tensor, torch.Tensor):
        input_tensor = img_tensor.unsqueeze(0).to(device)
    else:
        arr = np.array(img_tensor)
        if arr.ndim==3 and arr.shape[2]==3:
            input_tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(device).float()
        else:
            raise RuntimeError("Unsupported image format for Grad-CAM")
    try:
        cam = generate_gradcam(model, input_tensor, int(lbl), device)
    except Exception as e:
        print(f"⚠ Grad-CAM failed for {cls_name}: {e}")
        continue
    cam_resized = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().permute(1,2,0).numpy()
        img_np = (img_np * inv_std) + inv_mean
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        arr = img_tensor
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(config["gradcam_out_dir"], f"{cls_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"Saved Grad-CAM for class {cls_name} -> {out_path}")
    saved += 1

print(f"\nSaved {saved} Grad-CAM(s) to '{config['gradcam_out_dir']}'")
