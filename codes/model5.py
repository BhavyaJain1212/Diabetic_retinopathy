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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import timm

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Config (change only here)
# -------------------------
config = {
    "csv_path": "train.csv",
    "img_root": "./diabetes/colored_images",
    "backbone": "efficientnet_b2",  # or "swin_base_patch4_window7_224"
    "use_swin": False,  # Set True to use Swin Transformer
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 25,
    "lr": 1e-4,
    "use_cbam": True,
    "use_distill": True,
    "teacher_backbone": "efficientnet_b5",
    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_B5_Distill",
    "image_size": 224,
    "train_teacher": True,
    "teacher_epochs": 5,
    "distill_alpha_start": 0.2,
    "distill_alpha_end": 0.8,
    "distill_temperature": 4.0,
    "consistency_weight": 0.1,
    "use_class_weights": True,
    "label_smoothing": 0.1,
    "max_grad_norm": 5.0,
    "rotation_angles": [0, 90, 180, 270],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Advanced augmentation flags
    "use_clahe": True,
    "use_green_channel": False,  # Set True to use only green channel
    "use_pixel_amplification": True,
}

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
# Advanced Preprocessing Functions
# -------------------------
class CLAHETransform:
    """Apply CLAHE to enhance local contrast"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)

class GreenChannelExtract:
    """Extract green channel only"""
    def __call__(self, img):
        img_np = np.array(img)
        green = img_np[:, :, 1]
        # Convert back to 3-channel
        green_3ch = np.stack([green, green, green], axis=2)
        return Image.fromarray(green_3ch)

class PixelAmplification:
    """Amplify pixel intensities to enhance features"""
    def __init__(self, alpha=1.5, beta=10):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, img):
        img_np = np.array(img, dtype=np.float32)
        amplified = cv2.convertScaleAbs(img_np, alpha=self.alpha, beta=self.beta)
        return Image.fromarray(amplified)

class AddGaussianNoise:
    """Add Gaussian noise to image"""
    def __init__(self, mean=0., std=10.):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        img_np = np.array(img, dtype=np.float32)
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

# -------------------------
# Enhanced Data Augmentation Pipeline
# -------------------------
# Build preprocessing steps
preprocess_steps = []
if config["use_clahe"]:
    preprocess_steps.append(CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)))
if config["use_green_channel"]:
    preprocess_steps.append(GreenChannelExtract())
if config["use_pixel_amplification"]:
    preprocess_steps.append(PixelAmplification(alpha=1.3, beta=5))

# Training augmentation with advanced techniques
train_transform = transforms.Compose(
    preprocess_steps + [
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([AddGaussianNoise(mean=0, std=15)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Validation transform with preprocessing only
val_transform = transforms.Compose(
    preprocess_steps + [
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# -------------------------
# Dataset (fixed __init__, __len__, __getitem__)
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

# Build dataset
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
print("Classes:", class_names)

# -------------------------
# CBAM (fixed __init__)
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
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(out).view(b, c, 1, 1)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        out = self.bn(self.conv(cat))
        return torch.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x

# -------------------------
# Model with Swin Transformer support
# -------------------------
class EfficientNetWithCBAM(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=5, use_cbam=True, pretrained=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAMBlock(out_channels) if use_cbam else None
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if self.cbam:
            feats = self.cbam(feats)
        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits

class SwinWithCBAM(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', num_classes=5, use_cbam=True, pretrained=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAMBlock(out_channels) if use_cbam else None
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if self.cbam:
            feats = self.cbam(feats)
        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits

# -------------------------
# Distillation Loss (fixed __init__)
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
        frac = min(1.0, max(0.0, self.current_epoch / float(self.max_epochs - 1)))
        return self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def forward(self, student_logits, inputs, labels):
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None:
            return hard
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits / self.T, dim=1)
        t_prob = nn.functional.softmax(teacher_logits / self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T * self.T)
        alpha = self.get_alpha()
        return alpha * soft + (1.0 - alpha) * hard

# -------------------------
# Trainer (fixed __init__, added accuracy tracking)
# -------------------------
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
        correct = 0
        total = 0
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
                angle = random.choice(self.config.get("rotation_angles", [0, 90, 180, 270]))
                if angle % 360 != 0:
                    rotated_imgs = torch.stack([transforms.functional.rotate(img.cpu(), angle) for img in imgs]).to(self.device)
                else:
                    rotated_imgs = imgs
                with torch.no_grad():
                    out_orig = nn.functional.log_softmax(outputs / 1.0, dim=1)
                out_rot = self.model(rotated_imgs)
                out_rot_logp = nn.functional.log_softmax(out_rot / 1.0, dim=1)
                p_orig = nn.functional.softmax(outputs.detach(), dim=1)
                p_rot = nn.functional.softmax(out_rot.detach(), dim=1)
                kl1 = nn.functional.kl_div(out_orig, p_rot, reduction='batchmean')
                kl2 = nn.functional.kl_div(out_rot_logp, p_orig, reduction='batchmean')
                consistency_loss = 0.5 * (kl1 + kl2)
                loss = loss + self.config["consistency_weight"] * consistency_loss

            loss.backward()
            if self.config.get("max_grad_norm", None):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        
        avg_loss = running_loss / max(1, total)
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy().tolist())
                trues.extend(labels.numpy().tolist())
        accuracy = 100.0 * accuracy_score(trues, preds)
        return preds, trues, accuracy

# -------------------------
# Initialize models
# -------------------------
teacher_model = None
if config["use_distill"]:
    teacher_model = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
    print("✅ Teacher model created:", config["teacher_backbone"])

# Class weights
class_weights = None
if config["use_class_weights"]:
    label_counts = np.zeros(config["num_classes"], dtype=np.int64)
    for i in range(len(train_ds)):
        _, lbl = train_ds[i]
        label_counts[lbl] += 1
    freq = label_counts / float(label_counts.sum())
    eps = 1e-6
    weights = 1.0 / (freq + eps)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", weights)

# Loss
loss_name = config["loss_fn"].lower()
if loss_name in ("crossentropyloss", "crossentropy"):
    if config.get("label_smoothing", 0.0) and config["label_smoothing"] > 0.0:
        base_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
    else:
        base_loss = nn.CrossEntropyLoss(weight=class_weights)
elif loss_name == "mse":
    base_loss = nn.MSELoss()
elif loss_name in ("l1loss", "l1"):
    base_loss = nn.L1Loss()
elif loss_name == "huberloss":
    base_loss = nn.SmoothL1Loss()
else:
    raise ValueError(f"Unsupported loss function: {config['loss_fn']}")

if config["use_distill"]:
    criterion = DistillationLoss(
        base_loss=base_loss,
        teacher_model=teacher_model,
        alpha_start=config["distill_alpha_start"],
        alpha_end=config["distill_alpha_end"],
        temperature=config["distill_temperature"]
    )
else:
    criterion = base_loss

# Student model - choose architecture
if config["use_swin"]:
    model = SwinWithCBAM(config["backbone"], config["num_classes"], config["use_cbam"], pretrained=True).to(device)
    print("✅ Student model created with Swin Transformer:", config["backbone"])
else:
    model = EfficientNetWithCBAM(config["backbone"], config["num_classes"], config["use_cbam"], pretrained=True).to(device)
    print("✅ Student model created:", config["backbone"])

# Teacher training
if config["use_distill"] and config["train_teacher"]:
    print("\n🎓 Starting teacher warm-up / fine-tune stage...")
    # Ensure teacher parameters are trainable
    for p in teacher_model.parameters():
        p.requires_grad = True
    teacher_model.train()
    
    teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=config["lr"] * 0.5)
    teacher_scheduler = None
    if len(train_loader) > 0:
        teacher_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            teacher_optimizer,
            max_lr=config["lr"],
            steps_per_epoch=len(train_loader),
            epochs=config["teacher_epochs"]
        )
    teacher_criterion = base_loss
    teacher_trainer = Trainer(teacher_model, teacher_criterion, teacher_optimizer, teacher_scheduler, device, config)

    best_teacher_loss = float("inf")
    for t_epoch in range(config["teacher_epochs"]):
        t_loss, t_acc = teacher_trainer.train_epoch(train_loader, epoch_idx=t_epoch, total_epochs=config["teacher_epochs"])
        teacher_model.eval()
        vloss = 0.0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = teacher_model(imgs)
                loss = teacher_criterion(outputs, labels)
                vloss += loss.item() * imgs.size(0)
                total += imgs.size(0)
        vloss /= max(1, total)
        print(f"Teacher Epoch {t_epoch+1}/{config['teacher_epochs']} - Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.2f}% - Val Loss: {vloss:.4f}")
        if vloss < best_teacher_loss:
            best_teacher_loss = vloss
            torch.save(teacher_model.state_dict(), "best_teacher.pth")
            print("✅ Saved best teacher weights.")
    if os.path.exists("best_teacher.pth"):
        teacher_model.load_state_dict(torch.load("best_teacher.pth", map_location=device))
    teacher_model.eval()
    if isinstance(criterion, DistillationLoss):
        criterion.teacher = teacher_model
        for p in criterion.teacher.parameters():
            p.requires_grad = False
    print("🎓 Teacher training complete.\n")

# Student optimizer & scheduler
optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
if len(train_loader) > 0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"] * 5,
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"]
    )
else:
    scheduler = None

trainer = Trainer(model, criterion, optimizer, scheduler, device, config)

# -------------------------
# Training loop with accuracy tracking
# -------------------------
train_losses = []
train_accs = []
val_accs = []
best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]

for epoch in range(config["epochs"]):
    train_loss, train_acc = trainer.train_epoch(train_loader, epoch_idx=epoch, total_epochs=config["epochs"])
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(criterion, DistillationLoss):
                if hasattr(criterion, "set_epoch"):
                    criterion.set_epoch(epoch, config["epochs"])
                loss = criterion(outputs, imgs, labels)
            else:
                loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    val_loss /= max(1, total)
    val_acc = 100.0 * accuracy_score(trues, preds)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Validation loss improved — model saved.")
    else:
        patience_counter += 1
        print(f"⚠ No improvement for {patience_counter} epochs.")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered!")
            break

# -------------------------
# Plot training metrics
# -------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

ax2.plot(range(1, len(train_accs) + 1), train_accs, marker='o', label='Train Accuracy')
ax2.plot(range(1, len(val_accs) + 1), val_accs, marker='s', label='Val Accuracy')
ax2.set_title("Train & Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# -------------------------
# Final evaluation
# -------------------------
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("✅ Loaded best model weights for final evaluation.")

preds, trues, final_acc = trainer.evaluate(val_loader)
print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT")
print("="*60)
print(classification_report(trues, preds, digits=4, target_names=class_names))
print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")

# -------------------------
# Save confusion matrix
# -------------------------
os.makedirs(config["gradcam_out_dir"], exist_ok=True)
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
cm_path = os.path.join(config["gradcam_out_dir"], "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {cm_path}")
plt.show()

# -------------------------
# Grad-CAM implementation
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
        raise RuntimeError("No Conv2d layer found in the model for Grad-CAM.")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    outputs = model(input_tensor)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations['value']
    grad = gradients['value']
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam_map = torch.sum(weights * act, dim=1, keepdim=True)
    cam_map = torch.relu(cam_map)
    cam_map = cam_map.squeeze().cpu().numpy()

    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    fh.remove()
    bh.remove()
    return cam_map

# -------------------------
# Generate and save Grad-CAMs
# -------------------------
model.to(device)
model.eval()

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

print("\n🎯 Generating Grad-CAMs for one random validation image per class...")
saved_count = 0
for cls_idx, cls_name in enumerate(class_names):
    idxs = indices_per_class.get(cls_idx, [])
    if not idxs:
        print(f"⚠ No samples for class {cls_name} in validation set; skipping.")
        continue
    sel_local_idx = random.choice(idxs)
    img_tensor, lbl = val_ds[sel_local_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    target_class = int(lbl)
    try:
        cam = generate_gradcam(model, input_tensor, target_class, device)
    except Exception as e:
        print(f"⚠ Grad-CAM failed for class {cls_name}: {e}")
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
    print(f"✅ Saved Grad-CAM for class '{cls_name}' → {out_path}")
    saved_count += 1

if saved_count == 0:
    print("⚠ No Grad-CAMs were saved.")
else:
    print(f"\n🎉 Saved {saved_count} Grad-CAM(s) in '{config['gradcam_out_dir']}'")
    
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
