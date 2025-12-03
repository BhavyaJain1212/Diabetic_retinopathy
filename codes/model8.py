# train_with_focal.py
"""
Focal loss + WeightedRandomSampler training script
- EfficientNet-B2 backbone (timm)
- Focal Loss implementation (with alpha from inverse class frequency)
- WeightedRandomSampler for balanced mini-batches
- Training/validation, early stopping, save best model
- Grad-CAM outputs saved to B2_B5_Distill/gradcam_focal/
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
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
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
    "backbone": "efficientnet_b2",
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_B5_Distill",
    "gradcam_subdir": "gradcam_focal",
    "image_size": 224,
    "num_workers": 4,
    "seed": 42
}

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
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Dataset loader (CSV-first, fallback to ImageFolder)
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
        raise FileNotFoundError(f"Image {img_id} not found in {folder}")

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

# build dataset
dataset = None
use_imagefolder = False
if os.path.exists(config["csv_path"]) and os.path.isdir(config["img_root"]):
    try:
        dataset = RetinopathyDataset(config["csv_path"], config["img_root"], transform=train_transform)
        print("Loaded CSV dataset.")
    except Exception as e:
        print("Failed reading CSV dataset:", e)
        dataset = None

if dataset is None:
    if os.path.isdir(config["img_root"]):
        dataset = datasets.ImageFolder(root=config["img_root"], transform=train_transform)
        use_imagefolder = True
        print("Loaded ImageFolder dataset.")
    else:
        raise FileNotFoundError("Dataset not found.")

# class names
if use_imagefolder:
    class_names = dataset.classes
else:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# train/val split
total = len(dataset)
train_n = int(0.8 * total)
val_n = total - train_n
train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(config["seed"]))

# ensure transforms for val
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

# -------------------------
# WeightedRandomSampler: build sampler from labels in train_ds
# -------------------------
def build_sampler(ds, num_classes):
    labels = []
    if isinstance(ds, Subset):
        underlying = ds.dataset
        for i in ds.indices:
            _, lbl = underlying[i] if hasattr(underlying, 'getitem_') else (None, 0)
            labels.append(int(lbl))
    else:
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(int(lbl))
    labels = np.array(labels)
    counts = np.array([ (labels == i).sum() for i in range(num_classes) ])
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    samples_weight = weights[labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), num_samples=len(samples_weight), replacement=True)
    return sampler, counts

sampler, class_counts = build_sampler(train_ds, config["num_classes"])
print("Train class counts:", class_counts)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"])
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

# -------------------------
# Model (EfficientNet features_only + simple head)
# -------------------------
class SimpleEff(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        out_ch = self.backbone.feature_info[-1]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        f = self.backbone(x)[-1]
        p = self.pool(f).flatten(1)
        return self.fc(p)

model = SimpleEff(config["backbone"], config["num_classes"], pretrained=True).to(device)

# -------------------------
# Focal Loss
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
        logpt = nn.functional.log_softmax(inputs, dim=1)
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

# compute alpha from class_counts (inverse freq normalized)
alpha = (class_counts.max() / class_counts).astype(float)
alpha = alpha / alpha.sum()
print("Focal alpha (normalized):", alpha)
criterion = FocalLoss(gamma=2.0, alpha=alpha)

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
if len(train_loader) > 0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"] * 10,
                                                    steps_per_epoch=len(train_loader), epochs=config["epochs"])
else:
    scheduler = None

# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

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

    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        return preds, trues

trainer = Trainer(model, criterion, optimizer, scheduler, device)

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
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
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
        torch.save(model.state_dict(), os.path.join(config["gradcam_out_dir"], "focal_best.pth"))
        print("✅ Validation loss improved — model saved.")
    else:
        patience_counter += 1
        print(f"⚠ No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered!")
            break

# -------------------------
# Final evaluation and confusion matrix
# -------------------------
preds, trues = trainer.evaluate(val_loader)
print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Focal Loss")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(config["gradcam_out_dir"], "confusion_matrix_focal.png"))

# -------------------------
# Grad-CAM generation
# -------------------------
os.makedirs(os.path.join(config["gradcam_out_dir"], config["gradcam_subdir"]), exist_ok=True)

def find_last_conv_module(net):
    last_mod = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            last_mod = module
    return last_mod

def generate_gradcam(model, input_tensor, target_class, device):
    model.eval()
    last_conv = find_last_conv_module(model)
    if last_conv is None:
        raise RuntimeError("No Conv2d found for Grad-CAM.")
    activations = {}
    gradients = {}
    def fwd(m, i, o):
        activations['val'] = o.detach()
    def bwd(m, gi, go):
        gradients['val'] = go[0].detach()
    fh = last_conv.register_forward_hook(fwd)
    bh = last_conv.register_backward_hook(bwd)
    outputs = model(input_tensor)
    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    act = activations['val']
    grad = gradients['val']
    weights = torch.mean(grad, dim=(2,3), keepdim=True)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)
    cam_map = torch.relu(cam_map).squeeze().cpu().numpy()
    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()
    fh.remove()
    bh.remove()
    return cam_map

# build mapping of class->indices in validation set
indices_per_class = {i: [] for i in range(len(class_names))}
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

inv_mean = np.array([0.485, 0.456, 0.406])
inv_std = np.array([0.229, 0.224, 0.225])

print("\nGenerating Grad-CAMs for one random validation image per class...")
saved_count = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs:
        print(f"⚠ No samples for class {cls_name}; skipping.")
        continue
    sel_local_idx = random.choice(idxs)
    img_tensor, lbl = val_ds[sel_local_idx]
    if isinstance(img_tensor, torch.Tensor):
        input_tensor = img_tensor.unsqueeze(0).to(device)
    else:
        arr = np.array(img_tensor)
        if arr.ndim == 3 and arr.shape[2] == 3:
            input_tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(device).float()
        else:
            raise RuntimeError("Unsupported image shape for Grad-CAM.")
    target_class = int(lbl)
    try:
        cam = generate_gradcam(model, input_tensor, target_class, device)
    except Exception as e:
        print(f"⚠ Grad-CAM failed for class {cls_name}: {e}")
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
    out_path = os.path.join(config["gradcam_out_dir"], config["gradcam_subdir"], f"{cls_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Saved Grad-CAM for class '{cls_name}' → {out_path}")
    saved_count += 1

if saved_count == 0:
    print("⚠ No Grad-CAMs were saved.")
else:
    print(f"\n🎉 Saved {saved_count} Grad-CAM(s) in '{os.path.join(config['gradcam_out_dir'], config['gradcam_subdir'])}'")
