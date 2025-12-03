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

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Config (change only here)
# -------------------------
config = {
    "csv_path": "train.csv",          # if missing, code will fallback to ImageFolder on img_root
    "img_root": "./diabetes/colored_images",     # folder root OR ImageFolder root (class subfolders)
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
    "gradcam_out_dir": "B2_B5_Distill",
    "image_size": 224,
    # New options
    "train_teacher": True,            # whether to pretrain/fine-tune teacher before student
    "teacher_epochs": 5,              # teacher training epochs (if train_teacher True)
    "distill_alpha_start": 0.3,       # initial weight for soft (distill) loss
    "distill_alpha_end": 0.7,         # final weight for soft (distill) loss (linearly scheduled)
    "distill_temperature": 4.0,
    "consistency_weight": 0.1,        # rotation consistency loss weight
    "use_class_weights": True,        # use class weighting for hard loss
    "label_smoothing": 0.1,           # label smoothing factor for hard loss (0 to disable)
    "max_grad_norm": 5.0,             # gradient clipping
    "rotation_angles": [0, 90, 180, 270], # angles for rotation consistency and C4 pooling
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
# Dataset loader: CSV-based or ImageFolder fallback
# -------------------------
# stronger augmentation for training (rotation also used in consistency)
train_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),  # general rotation augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# validation/test transforms: deterministic
val_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class RetinopathyDataset(Dataset):
    # corrected magic method names
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png", ".jpg", ".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]
        self.folder_names = sorted([f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))])
        # numeric map (matches your earlier map)
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

# Build dataset: try CSV, else ImageFolder
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
        # For ImageFolder we will apply train transform to training subset and val_transform to val subset
        dataset = datasets.ImageFolder(root=config["img_root"], transform=train_transform)
        use_imagefolder = True
        print("✅ Loaded dataset with ImageFolder (class-subfolders).")
    else:
        raise FileNotFoundError(f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist.")

# class names
if use_imagefolder:
    class_names = dataset.classes
else:
    # use the numeric_to_folder mapping order
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Ensure val uses deterministic transform
# If underlying dataset is ImageFolder we need to set transform for val subset items
if use_imagefolder:
    # ImageFolder stores samples in dataset.samples and transforms globally, so easiest is to create a copy dataset for val with val_transform
    val_dataset_full = datasets.ImageFolder(root=config["img_root"], transform=val_transform)
    # val_ds is a Subset of the original ImageFolder; we rebuild subset indices into this new val_dataset_full
    val_ds = Subset(val_dataset_full, val_ds.indices)
else:
    # For CSV-based RetinopathyDataset, create a new instance with val_transform
    val_ds = Subset(RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform), val_ds.indices)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

print(f"Dataset: {len(dataset)} images → train: {len(train_ds)}, val: {len(val_ds)}")
print("Classes:", class_names)

# -------------------------
# CBAM (Channel + Spatial) - same design but fixed init names
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
# Model (EfficientNet features_only)
# -------------------------
class EfficientNetWithCBAM(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=5, use_cbam=True, pretrained=True):
        super().__init__()
        # create a features-only model and keep the last feature map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        # feature_info list -> last element channels
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAMBlock(out_channels) if use_cbam else None
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # last feature map (B, C, H, W)
        if self.cbam:
            feats = self.cbam(feats)
        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits

# -------------------------
# C4 rotation-equivariant wrapper (0/90/180/270)
# -------------------------
class C4EquivariantWrapper(nn.Module):
    def __init__(self, base_model, angles=(0, 90, 180, 270)):
        super().__init__()
        # reuse components from EfficientNetWithCBAM
        self.feature_extractor = base_model.feature_extractor
        self.cbam = getattr(base_model, 'cbam', None)
        self.classifier = base_model.classifier
        self.angles = tuple(angles)

    def _rot90(self, x, k):
        if k % 4 == 0:
            return x
        return torch.rot90(x, k=k, dims=(2, 3))

    def forward(self, x):
        feats_sum = None
        count = 0
        # process non-zero angles first so angle 0 runs last (helps Grad-CAM hooks without changing outputs)
        angles_order = [a for a in self.angles if (a % 360) != 0] + [a for a in self.angles if (a % 360) == 0]
        for ang in angles_order:
            k = (ang // 90) % 4
            xr = self._rot90(x, k)
            feats = self.feature_extractor(xr)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]  # last feature map (B, C, H, W)
            if self.cbam is not None:
                feats = self.cbam(feats)
            # inverse-rotate features back to canonical orientation
            feats_back = self._rot90(feats, (-k) % 4)
            feats_sum = feats_back if feats_sum is None else feats_sum + feats_back
            count += 1
        feats_avg = feats_sum / count
        pooled = nn.functional.adaptive_avg_pool2d(feats_avg, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits

# -------------------------
# DistillationLoss (improved): supports alpha schedule via set_epoch
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
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def set_epoch(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def get_alpha(self):
        # linear schedule for alpha (weight of distillation / soft loss)
        if self.max_epochs <= 1:
            return self.alpha_end
        frac = min(1.0, max(0.0, self.current_epoch / float(self.max_epochs - 1)))
        return self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def forward(self, student_logits, inputs, labels):
        # hard loss (base)
        hard = self.base_loss(student_logits, labels)
        if self.teacher is None:
            return hard
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        s_logp = nn.functional.log_softmax(student_logits / self.T, dim=1)
        t_prob = nn.functional.softmax(teacher_logits / self.T, dim=1)
        soft = nn.KLDivLoss(reduction='batchmean')(s_logp, t_prob) * (self.T * self.T)
        alpha = self.get_alpha()
        # alpha is weight for soft (distill) loss; (1-alpha) for hard
        return alpha * soft + (1.0 - alpha) * hard

# -------------------------
# Trainer (modified to support distillation + rotation consistency + epoch passing)
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
        n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(imgs)
            if isinstance(self.criterion, DistillationLoss):
                # distillation expects (student_logits, inputs, labels)
                if hasattr(self.criterion, "set_epoch"):
                    self.criterion.set_epoch(epoch_idx, total_epochs)
                loss = self.criterion(outputs, imgs, labels)
            else:
                loss = self.criterion(outputs, labels)

            # rotation-consistency loss (simple): pick random rotation angle, compute prediction on rotated images and enforce consistency
            if self.config.get("consistency_weight", 0.0) > 0.0:
                angle = random.choice(self.config.get("rotation_angles", [0, 90, 180, 270]))
                if angle % 360 != 0:
                    # imgs are tensors normalized; rotate using torchvision.functional.rotate
                    rotated_imgs = torch.stack([transforms.functional.rotate(img.cpu(), angle) for img in imgs]).to(self.device)
                else:
                    rotated_imgs = imgs
                with torch.no_grad():
                    out_orig = nn.functional.log_softmax(outputs / 1.0, dim=1)  # (log-probs)
                out_rot = self.model(rotated_imgs)
                out_rot_logp = nn.functional.log_softmax(out_rot / 1.0, dim=1)
                # consistency: symmetric KL between out_orig and out_rot
                p_orig = nn.functional.softmax(outputs.detach(), dim=1)
                p_rot = nn.functional.softmax(out_rot.detach(), dim=1)
                kl1 = nn.functional.kl_div(out_orig, p_rot, reduction='batchmean')
                kl2 = nn.functional.kl_div(out_rot_logp, p_orig, reduction='batchmean')
                consistency_loss = 0.5 * (kl1 + kl2)
                loss = loss + self.config["consistency_weight"] * consistency_loss

            loss.backward()
            # gradient clipping
            if self.config.get("max_grad_norm", None):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            # scheduler stepping (safe)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

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
# Initialize teacher (optional) and student models, losses, optimizers
# -------------------------
teacher_model = None
if config["use_distill"]:
    # teacher: pretrained model but with classifier head matching num_classes
    teacher_model = timm.create_model(config["teacher_backbone"], pretrained=True, num_classes=config["num_classes"]).to(device)
    # Optionally fine-tune last layers only or whole teacher — we will allow full finetune in teacher training block
    print("✅ Teacher model created:", config["teacher_backbone"])

# Compute class weights if requested (from train_ds)
class_weights = None
if config["use_class_weights"]:
    # build label histogram from training subset
    label_counts = np.zeros(config["num_classes"], dtype=np.int64)
    for i in range(len(train_ds)):
        _, lbl = train_ds[i]
        label_counts[lbl] += 1
    # inverse frequency (with smoothing)
    freq = label_counts / float(label_counts.sum())
    # weight = 1 / (freq + eps)
    eps = 1e-6
    weights = 1.0 / (freq + eps)
    # normalize to mean=1
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", weights)

# choose base loss from config, support label smoothing if CrossEntropy requested
loss_name = config["loss_fn"].lower()
if loss_name in ("crossentropyloss", "crossentropy"):
    # use label_smoothing if > 0
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

# assemble final criterion: if using distillation, wrap
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

# student model (wrapped with C4 rotation-equivariant pooling)
base_model = EfficientNetWithCBAM(config["backbone"], config["num_classes"], config["use_cbam"], pretrained=True).to(device)
model = C4EquivariantWrapper(base_model, angles=config["rotation_angles"]).to(device)
print("✅ Student model created (C4-equivariant):", config["backbone"])

# teacher training stage (optional)
if config["use_distill"] and config["train_teacher"]:
    print("\n🎓 Starting teacher warm-up / fine-tune stage...")
    teacher_model.train()
    for p in teacher_model.parameters():
      p.requires_grad = True
    teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=config["lr"] * 0.5)
    teacher_scheduler = None
    if len(train_loader) > 0:
        teacher_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            teacher_optimizer,
            max_lr=config["lr"],
            steps_per_epoch=len(train_loader),
            epochs=config["teacher_epochs"]
        )
    teacher_criterion = base_loss  # teacher trained on hard labels only (with smoothing/weights)
    teacher_trainer = Trainer(teacher_model, teacher_criterion, teacher_optimizer, teacher_scheduler, device, config)

    best_teacher_loss = float("inf")
    for t_epoch in range(config["teacher_epochs"]):
        t_loss = teacher_trainer.train_epoch(train_loader, epoch_idx=t_epoch, total_epochs=config["teacher_epochs"])
        # evaluate on val
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
        print(f"Teacher Epoch {t_epoch+1}/{config['teacher_epochs']} - Train Loss: {t_loss:.4f} - Val Loss: {vloss:.4f}")
        # save best teacher
        if vloss < best_teacher_loss:
            best_teacher_loss = vloss
            torch.save(teacher_model.state_dict(), "best_teacher.pth")
            print("✅ Saved best teacher weights.")
    # reload best teacher weights into teacher_model
    if os.path.exists("best_teacher.pth"):
        teacher_model.load_state_dict(torch.load("best_teacher.pth", map_location=device))
    teacher_model.eval()
    # make sure teacher in distillation wrapper is updated
    if isinstance(criterion, DistillationLoss):
        criterion.teacher = teacher_model
        for p in criterion.teacher.parameters():
            p.requires_grad = False
    print("🎓 Teacher training complete.\n")

# -------------------------
# Student optimizer & scheduler
# -------------------------
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
# Training loop + validation + early stopping
# -------------------------
train_losses = []
best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader, epoch_idx=epoch, total_epochs=config["epochs"])
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(criterion, DistillationLoss):
                # use current criterion (this will use teacher if present)
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

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # early stopping + save best
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

    # classification report each epoch (try)
    try:
        print(classification_report(trues, preds, digits=4, target_names=class_names))
    except Exception:
        pass

# -------------------------
# Plot training loss
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

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
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------------
# Self-contained Grad-CAM implementation (fixed hooks)
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
        # grad_out is a tuple; grad_out[0] is gradient wrt output
        gradients['value'] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

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
if not os.path.exists("best_model.pth"):
    print("⚠ No best_model.pth found — using current model weights for Grad-CAM.")
else:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.to(device)
model.eval()

# build mapping of class->indices in validation set
indices_per_class = {i: [] for i in range(len(class_names))}
# val_ds is a Subset, iterate via its indices
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
    input_tensor = img_tensor.unsqueeze(0).to(device)

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

    # reconstruct original image (unnormalize)
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * inv_std) + inv_mean  # un-normalize
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

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

