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

# --- TDA Imports ---
try:
    import gudhi as gd
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("⚠ 'gudhi' library not found. TDA features will be zeros. Run: pip install gudhi")

try:
    from persim import PersistenceImager
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    print("⚠ 'persim' library not found. Persistence images will be zeros. Run: pip install persim")

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
    "epochs": 40,
    "lr": 1e-4,
    "use_cbam": True,
    "use_distill": True,
    "teacher_backbone": "efficientnet_b5",
    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_B5_Distill_TDA_PI",
    "image_size": 224,

    # Teacher Training Config
    "train_teacher": True,
    "teacher_epochs": 15,

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

    # TDA Config (persistence images)
    "use_tda": True,
    "tda_img_size": 32,
    "tda_mlp_dims": [512, 128],

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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, img_exts=(".png", ".jpg", ".jpeg")):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

        # Normalize column names
        self.data.columns = [c.strip().lower() for c in self.data.columns]
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]

        self.folder_names = sorted(
            [f for f in os.listdir(img_root)
             if os.path.isdir(os.path.join(img_root, f))]
        )
        self.numeric_to_folder = {
            0: "No_DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferate_DR"
        }
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

# -------------------------
# Load Dataset & Create Train/Val/Test Split (70/15/15) - ✅ FIXED
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
        print("✅ Loaded dataset with ImageFolder.")
    else:
        raise FileNotFoundError(
            f"Neither CSV ({config['csv_path']}) nor image root ({config['img_root']}) exist."
        )

if use_imagefolder:
    class_names = dataset.classes
else:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

print(f"Total dataset size: {len(dataset)}")

# ✅ FIXED: PROPER TRAIN/VAL/TEST SPLIT (70/15/15)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"✅ Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# ✅ FIXED: Save split indices for reproducibility
split_indices = {
    "train": list(train_ds.indices),
    "val": list(val_ds.indices),
    "test": list(test_ds.indices)
}
torch.save(split_indices, "split_indices.pth")
print("💾 Split indices saved to 'split_indices.pth'")

# Create proper validation and test datasets with val_transform
if use_imagefolder:
    val_dataset_full = datasets.ImageFolder(root=config["img_root"], transform=val_transform)
    test_dataset_full = datasets.ImageFolder(root=config["img_root"], transform=val_transform)
else:
    val_dataset_full = RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform)
    test_dataset_full = RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform)

val_ds = Subset(val_dataset_full, val_ds.indices)
test_ds = Subset(test_dataset_full, test_ds.indices)

# Create DataLoaders
train_loader = DataLoader(
    train_ds, batch_size=config["batch_size"],
    shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=config["batch_size"],
    shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=config["batch_size"],
    shuffle=False, num_workers=2, pin_memory=True
)

print("✅ All DataLoaders created successfully!")

# -------------------------
# Persistence Image Utilities
# -------------------------
if TDA_AVAILABLE and PERSIM_AVAILABLE:
    pimgr = PersistenceImager(birth_range=(0.0, 1.0),
                              pers_range=(0.0, 1.0),
                              pixel_size=None)
else:
    pimgr = None

def compute_persistence_diagram_from_image(img_arr):
    cc = gd.CubicalComplex(
        dimensions=img_arr.shape,
        top_dimensional_cells=img_arr.flatten()
    )
    persistence = cc.persistence()
    diag = []
    for dim, (b, d) in persistence:
        if dim in (0, 1) and d != np.inf and d > b:
            diag.append([b, d])
    if len(diag) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(diag, dtype=np.float32)

def extract_persistence_image_batch(images_tensor, tda_img_size=32):
    B = images_tensor.size(0)
    if not (TDA_AVAILABLE and PERSIM_AVAILABLE):
        return torch.zeros((B, tda_img_size * tda_img_size), device=images_tensor.device)

    small_imgs = torch.nn.functional.interpolate(
        images_tensor,
        size=(tda_img_size, tda_img_size),
        mode='bilinear',
        align_corners=False
    )
    gray_imgs = (
        0.299 * small_imgs[:, 0, :, :] +
        0.587 * small_imgs[:, 1, :, :] +
        0.114 * small_imgs[:, 2, :, :]
    )
    gray_imgs_np = gray_imgs.cpu().detach().numpy()

    diagrams = []
    for i in range(B):
        img_arr = gray_imgs_np[i]
        diag = compute_persistence_diagram_from_image(img_arr)
        diagrams.append(diag)

    global pimgr
    if pimgr is not None and not getattr(pimgr, "fitted_", False):
        non_empty_diags = [d for d in diagrams if d.shape[0] > 0]
        if len(non_empty_diags) > 0:
            pimgr.fit(non_empty_diags, skew=True)

    batch_imgs = []
    for dgm in diagrams:
        if dgm.shape[0] == 0:
            pim = np.zeros((tda_img_size, tda_img_size), dtype=np.float32)
        else:
            pim = pimgr.transform(dgm, skew=True)
            pim = cv2.resize(pim.astype(np.float32), (tda_img_size, tda_img_size),
                             interpolation=cv2.INTER_LINEAR)
        batch_imgs.append(pim.flatten())

    batch_tensor = torch.tensor(batch_imgs, dtype=torch.float32, device=images_tensor.device)
    min_vals = batch_tensor.min(dim=1, keepdim=True)[0]
    max_vals = batch_tensor.max(dim=1, keepdim=True)[0]
    batch_tensor = (batch_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return batch_tensor

# -------------------------
# CBAM & Student Model
# -------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        c_out = torch.sigmoid(
            self.channel_gate(avg_pool) + self.channel_gate(max_pool)
        ).view(b, c, 1, 1)
        x = x * c_out

        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        s_out = torch.sigmoid(
            self.spatial_gate(torch.cat([avg_s, max_s], dim=1))
        )
        return x * s_out

class EfficientNetWithCBAM_PI(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2',
                 num_classes=5, use_cbam=True,
                 pretrained=True, use_tda=False,
                 tda_img_size=32, tda_mlp_dims=[512, 128]):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True
            )

        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.use_cbam = use_cbam
        self.cbam = CBAMBlock(out_channels) if use_cbam else None

        self.use_tda = use_tda
        self.tda_img_size = tda_img_size
        self.tda_vec_dim = tda_img_size * tda_img_size if use_tda else 0

        if use_tda:
            mlp_layers = []
            in_dim = self.tda_vec_dim
            for dim in tda_mlp_dims:
                mlp_layers.append(nn.Linear(in_dim, dim))
                mlp_layers.append(nn.ReLU(inplace=True))
                in_dim = dim
            self.tda_mlp = nn.Sequential(*mlp_layers)
            fusion_in_dim = out_channels + tda_mlp_dims[-1]
        else:
            self.tda_mlp = None
            fusion_in_dim = out_channels

        self.classifier = nn.Linear(fusion_in_dim, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        if self.cbam:
            feats = self.cbam(feats)

        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        if self.use_tda:
            tda_feats = extract_persistence_image_batch(
                x, tda_img_size=self.tda_img_size
            )
            tda_feats = self.tda_mlp(tda_feats)
            combined = torch.cat((pooled, tda_feats), dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(pooled)

        return logits

# -------------------------
# Loss & Trainer
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_loss, teacher_model=None,
                 alpha_start=0.5, alpha_end=0.5, temperature=4.0):
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

            # Consistency regularization (rotation)
            if self.config.get("consistency_weight", 0.0) > 0.0:
                angle = random.choice(self.config["rotation_angles"])
                if angle % 360 != 0:
                    rotated_imgs = torch.stack(
                        [transforms.functional.rotate(img.cpu(), angle) for img in imgs]
                    ).to(self.device)

                    with torch.no_grad():
                        out_orig = nn.functional.log_softmax(outputs, dim=1)

                    out_rot = self.model(rotated_imgs)
                    kl_loss = nn.functional.kl_div(
                        out_orig,
                        nn.functional.softmax(out_rot.detach(), dim=1),
                        reduction='batchmean'
                    )
                    loss += self.config["consistency_weight"] * kl_loss

            loss.backward()

            if self.config.get("max_grad_norm", None):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["max_grad_norm"]
                )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

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

# 1. Setup Class Weights & Base Loss (using train set only)
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

base_loss = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=config["label_smoothing"]
)

# 2. Teacher Setup & Training
teacher_model = None
if config["use_distill"]:
    teacher_model = timm.create_model(
        config["teacher_backbone"],
        pretrained=True,
        num_classes=config["num_classes"]
    ).to(device)
    print(f"✅ Teacher initialized: {config['teacher_backbone']}")

    if config["train_teacher"]:
        print("\n🎓 Starting Teacher Training...")

        for p in teacher_model.parameters():
            p.requires_grad = True

        t_opt = optim.AdamW(teacher_model.parameters(), lr=config["lr"] * 0.5)
        t_sched = torch.optim.lr_scheduler.OneCycleLR(
            t_opt,
            max_lr=config["lr"],
            steps_per_epoch=len(train_loader),
            epochs=config["teacher_epochs"]
        )
        t_trainer = Trainer(teacher_model, base_loss, t_opt, t_sched, device, config)

        best_t_loss = float("inf")
        for ep in range(config["teacher_epochs"]):
            tl = t_trainer.train_epoch(train_loader, ep, config["teacher_epochs"])

            teacher_model.eval()
            vl, tot = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    vl += base_loss(teacher_model(imgs), labels).item() * imgs.size(0)
                    tot += imgs.size(0)
            vl /= max(1, tot)
            print(f"Teacher Epoch {ep+1}/{config['teacher_epochs']} - "
                  f"Train: {tl:.4f} - Val: {vl:.4f}")

            if vl < best_t_loss:
                best_t_loss = vl
                torch.save(teacher_model.state_dict(), "best_teacher.pth")

        if os.path.exists("best_teacher.pth"):
            teacher_model.load_state_dict(
                torch.load("best_teacher.pth", map_location=device)
            )
        print("🎓 Teacher Training Complete.\n")

# 3. Initialize Distillation Loss
if config["use_distill"]:
    criterion = DistillationLoss(
        base_loss,
        teacher_model=teacher_model,
        alpha_start=config["distill_alpha_start"],
        alpha_end=config["distill_alpha_end"]
    )
else:
    criterion = base_loss

# 4. Student Setup (with persistence images)
model = EfficientNetWithCBAM_PI(
    backbone_name=config["backbone"],
    num_classes=config["num_classes"],
    use_cbam=config["use_cbam"],
    pretrained=True,
    use_tda=config["use_tda"],
    tda_img_size=config["tda_img_size"],
    tda_mlp_dims=config["tda_mlp_dims"]
).to(device)
print(f"✅ Student initialized: {config['backbone']} (TDA PI Fusion: {config['use_tda']})")

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["lr"] * 5,
    steps_per_epoch=len(train_loader),
    epochs=config["epochs"]
)
trainer = Trainer(model, criterion, optimizer, scheduler, device, config)

# 5. Student Training Loop (using val set for early stopping)
train_losses = []
best_val_loss = float("inf")
patience = config["early_stop_patience"]
counter = 0

for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(train_loader, epoch, config["epochs"])
    train_losses.append(train_loss)

    # Validation (for early stopping & model selection)
    model.eval()
    val_loss, total = 0.0, 0

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

    val_loss /= max(1, total)
    print(f"Epoch {epoch+1}/{config['epochs']} - "
          f"Train: {train_loss:.4f} - Val: {val_loss:.4f}")

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

# 6. Evaluation on TEST SET (unbiased final evaluation)
plt.figure(figsize=(10, 4))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Load best model for final evaluation
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

print("\n" + "="*60)
print("🎯 FINAL TEST SET EVALUATION (UNBIASED)")
print("="*60)
test_preds, test_trues = trainer.evaluate(test_loader)
print(classification_report(test_trues, test_preds, target_names=class_names))

cm = confusion_matrix(test_trues, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.title("Test Set Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class ROC
y_true_bin = label_binarize(test_trues, classes=list(range(config["num_classes"])))

# Get model prediction probabilities (not just argmax labels!)
model.eval()
y_prob = []

with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        prob = nn.functional.softmax(out, dim=1)
        y_prob.extend(prob.cpu().numpy())

y_prob = np.array(y_prob)

# Compute ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(config["num_classes"]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro & Macro AUC
roc_auc["micro"] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='micro')
roc_auc["macro"] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')

print("\n🔹 ROC-AUC Scores")
for i, cls in enumerate(class_names):
    print(f"{cls}: AUC = {roc_auc[i]:.4f}")
print(f"Micro AUC = {roc_auc['micro']:.4f}")
print(f"Macro AUC = {roc_auc['macro']:.4f}")

plt.figure(figsize=(8, 6))

for i in range(config["num_classes"]):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# -------------------------
# 7. Grad-CAM Visualization (using test set samples)
# -------------------------
def denormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def get_gradcam_target_layer(model):
    if getattr(model, "cbam", None) is not None:
        return model.cbam
    return model.feature_extractor

def generate_gradcam(model, img_tensor, class_idx, target_layer):
    model.eval()
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        if isinstance(out, (list, tuple)):
            activations["value"] = out[-1].detach()
        else:
            activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        g = grad_out[0]
        if isinstance(g, (list, tuple)):
            gradients["value"] = g[-1].detach()
        else:
            gradients["value"] = g.detach()

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    outputs = model(img_tensor)

    model.zero_grad()
    score = outputs[0, class_idx]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    A = activations["value"]
    G = gradients["value"]
    weights = torch.mean(G, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * A, dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam = cam.cpu().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam

# Generate Grad-CAMs from test set
os.makedirs(config["gradcam_out_dir"], exist_ok=True)
model.eval()
print("\n🔥 Generating Grad-CAMs from test set...")

target_layer = get_gradcam_target_layer(model)
cls_seen = {i: False for i in range(config["num_classes"])}

dataset_ref = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
indices = test_ds.indices if isinstance(test_ds, Subset) else range(len(test_ds))

for ds_idx in indices:
    if all(cls_seen.values()):
        break
        
    img, label = dataset_ref[ds_idx]

    if cls_seen[label]:
        continue

    cls_seen[label] = True

    cam = generate_gradcam(model, img, label, target_layer)
    cam = cv2.resize(cam, (config["image_size"], config["image_size"]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = denormalize_image(img)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(
        config["gradcam_out_dir"],
        f"gradcam_test_class_{class_names[label]}.jpg"
    )
    cv2.imwrite(save_path, overlay)
    print(f"✔ Saved Grad-CAM for class {class_names[label]} → {save_path}")

print("🎉 Grad-CAM generation complete.")
print("✅ Pipeline complete with proper train/val/test split!")

# extension of model_9_v3
# final code
# test set evaluation is there in this code.
# no stratified sampling