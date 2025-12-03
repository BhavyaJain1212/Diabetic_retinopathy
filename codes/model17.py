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
warnings.filterwarnings("ignore", category=FutureWarning) # Silence other minor warnings
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Config
# -------------------------
config = {
    "csv_path": "train.csv",
    "img_root": "./diabetes/colored_images",
    
    # Reduced Batch Size for A5000 + ViT-Large
    "batch_size": 16, 
    
    "backbone": "efficientnet_b2",
    "num_classes": 5,
    "epochs": 25,            
    "lr": 1e-4, 
    
    "use_cbam": True,
    "use_distill": True,
    
    "teacher_backbone": "vit_large_patch16_224", 
    "train_teacher": True,
    "teacher_epochs": 25,

    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_RETFound_Preprocess_TDA", 
    "image_size": 224,

    "median_ksize": 3,
    "gamma_value": 1.2,

    "distill_alpha_start": 0.3,
    "distill_alpha_end": 0.7,
    "distill_temperature": 4.0,

    "consistency_weight": 0.1,
    "use_class_weights": True,
    "label_smoothing": 0.1,
    "max_grad_norm": 5.0,
    "rotation_angles": [0, 90, 180, 270],

    "use_tda": True,
    "tda_img_size": 32,          
    "vessel_downsample": 64,     
    "tda_mlp_structure": [64, 32, 16],

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
# Preprocessing Class
# -------------------------
class PreprocessImage(object):
    def __init__(self, median_ksize=3, gamma=1.2):
        self.median_ksize = median_ksize
        self.gamma = gamma

    def __call__(self, img):
        img_np = np.array(img)
        img_np = cv2.medianBlur(img_np, self.median_ksize)
        
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        
        img_np = cv2.LUT(img_np, table)
        return Image.fromarray(img_np)

# -------------------------
# Data Transforms
# -------------------------
train_transform = transforms.Compose([
    PreprocessImage(median_ksize=config["median_ksize"], gamma=config["gamma_value"]),
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
    PreprocessImage(median_ksize=config["median_ksize"], gamma=config["gamma_value"]),
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
# Load Dataset
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

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

if use_imagefolder:
    val_dataset_full = datasets.ImageFolder(root=config["img_root"], transform=val_transform)
    val_ds = Subset(val_dataset_full, val_ds.indices)
else:
    val_ds = Subset(
        RetinopathyDataset(config["csv_path"], config["img_root"], transform=val_transform),
        val_ds.indices
    )

train_loader = DataLoader(
    train_ds, batch_size=config["batch_size"],
    shuffle=True, num_workers=0,pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=config["batch_size"],
    shuffle=False, num_workers=0,pin_memory=True
)

print(f"Dataset: {len(dataset)} images → train: {len(train_ds)}, val: {len(val_ds)}")

# -------------------------
# Vessel Extraction & TDA Logic
# -------------------------
if TDA_AVAILABLE and PERSIM_AVAILABLE:
    pimgr = PersistenceImager(birth_range=(0.0, 1.0),
                              pers_range=(0.0, 1.0),
                              pixel_size=None) 
else:
    pimgr = None

def extract_vessel_mask(img_tensor, size=128):
    img_np = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_np = cv2.resize(img_np, (size, size))

    green = img_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    inverted = 255 - enhanced
    
    mask = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    if dist_transform.max() > 0:
        dist_transform /= dist_transform.max()
        
    return dist_transform

def compute_vessel_topology(mask_arr):
    cc = gd.CubicalComplex(
        dimensions=mask_arr.shape,
        top_dimensional_cells=mask_arr.flatten()
    )
    persistence = cc.persistence()
    diag = []
    for dim, (b, d) in persistence:
        if dim in (0, 1) and d != np.inf:
             if (d - b) > 0.05: 
                 diag.append([b, d])
                 
    if len(diag) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(diag, dtype=np.float32)

def extract_persistence_image_batch(images_tensor, tda_img_size=32, vessel_size=64):
    B = images_tensor.size(0)
    if not (TDA_AVAILABLE and PERSIM_AVAILABLE):
        return torch.zeros((B, tda_img_size * tda_img_size), device=images_tensor.device)

    batch_imgs = []
    diagrams = []

    for i in range(B):
        mask = extract_vessel_mask(images_tensor[i], size=vessel_size)
        diag = compute_vessel_topology(mask)
        diagrams.append(diag)

    for dgm in diagrams:
        if dgm.shape[0] == 0:
            pim = np.zeros((tda_img_size, tda_img_size), dtype=np.float32)
        else:
            try:
                pim = pimgr.transform(dgm, skew=True)
                if pim is None or pim.size == 0:
                    pim = np.zeros((tda_img_size, tda_img_size), dtype=np.float32)
                else:
                    pim = pim.astype(np.float32)
                    if pim.shape[0] != tda_img_size:
                        pim = cv2.resize(pim, (tda_img_size, tda_img_size),
                                         interpolation=cv2.INTER_LINEAR)
            except Exception:
                pim = np.zeros((tda_img_size, tda_img_size), dtype=np.float32)

        batch_imgs.append(pim.flatten())

    batch_tensor = torch.tensor(np.array(batch_imgs), dtype=torch.float32, device=images_tensor.device)
    
    min_vals = batch_tensor.min(dim=1, keepdim=True)[0]
    max_vals = batch_tensor.max(dim=1, keepdim=True)[0]
    batch_tensor = (batch_tensor - min_vals) / (max_vals - min_vals + 1e-8)

    return batch_tensor

# -------------------------
# CBAM Block
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
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        c_out = torch.sigmoid(
            self.channel_gate(avg_pool) + self.channel_gate(max_pool)
        ).view(b, c, 1, 1)
        x = x * c_out
        
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        s_out = torch.sigmoid(
            self.spatial_gate(torch.cat([avg_s, max_s], dim=1))
        )
        return x * s_out

# -------------------------
# TEACHER: RETFound (ViT-Large)
# -------------------------
class RETFoundTeacher(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.base = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        self.embed_dim = self.base.embed_dim 
        self.attn = CBAMBlock(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        x = self.base.forward_features(x)
        patches = x[:, 1:, :] 
        
        B, N, C = patches.shape
        H_grid = int(N**0.5) 
        W_grid = H_grid
        
        spatial_feats = patches.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        refined_feats = self.attn(spatial_feats)
        global_pool = nn.functional.adaptive_avg_pool2d(refined_feats, (1, 1)).flatten(1)
        return self.classifier(global_pool)

# -------------------------
# STUDENT: EfficientNet + TDA
# -------------------------
class EfficientNetWithTDAGating(nn.Module):
    def __init__(self, backbone_name, num_classes, use_cbam, use_tda, tda_img_size, mlp_structure):
        super().__init__()
        self.feature_extractor = timm.create_model(backbone_name, pretrained=True, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        
        self.cbam = CBAMBlock(out_channels) if use_cbam else None
        
        self.use_tda = use_tda
        self.tda_img_size = tda_img_size
        
        if use_tda:
            layers = []
            in_dim = tda_img_size * tda_img_size
            for out_dim in mlp_structure:
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)])
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, out_channels))
            layers.append(nn.Sigmoid())
            self.tda_gate = nn.Sequential(*layers)
        
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)[-1]
        
        if self.cbam:
            feats = self.cbam(feats)
        
        if self.use_tda:
            tda_raw = extract_persistence_image_batch(
                x, tda_img_size=self.tda_img_size, vessel_size=config["vessel_downsample"]
            )
            tda_weights = self.tda_gate(tda_raw)
            tda_weights = tda_weights.view(tda_weights.size(0), tda_weights.size(1), 1, 1)
            feats = feats * tda_weights
            
        cnn_emb = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.classifier(cnn_emb)
        return logits

# -------------------------
# Loss & Trainer with AMP (Updated for PyTorch Future Warnings)
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
            # AMP is handled in the outer context
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
        # --- UPDATED: Correct syntax for newer PyTorch ---
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, loader, epoch_idx=0, total_epochs=1):
        self.model.train()
        running_loss = 0.0
        n = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # --- UPDATED: Correct syntax for newer PyTorch ---
            with torch.amp.autocast('cuda'):
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

            # Scaler backward pass
            self.scaler.scale(loss).backward()

            if self.config.get("max_grad_norm", None):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["max_grad_norm"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
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
                # --- UPDATED: Correct syntax for newer PyTorch ---
                with torch.amp.autocast('cuda'):
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
    base_loss = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"]
    )
else:
    base_loss = nn.CrossEntropyLoss(weight=class_weights)

# -------------------------
# 2. Teacher Setup (RETFound)
# -------------------------
teacher_model = None
if config["use_distill"]:
    print(f"🏗 Initializing Teacher: RETFound (ViT-Large) Architecture...")
    
    teacher_model = RETFoundTeacher(
        backbone_name=config["teacher_backbone"],
        num_classes=config["num_classes"]
    ).to(device)
    
    print(f"✅ Teacher initialized: ViT-Large + CBAM")

    if config["train_teacher"]:
        print("\n🎓 Starting Teacher Training (RETFound)...")
        
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

        best_t_acc = 0.0
        for ep in range(config["teacher_epochs"]):
            tl = t_trainer.train_epoch(train_loader, ep, config["teacher_epochs"])

            teacher_model.eval()
            correct, total = 0, 0
            vl = 0.0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    # --- UPDATED: Correct syntax for newer PyTorch ---
                    with torch.amp.autocast('cuda'):
                        out = teacher_model(imgs)
                        vl += base_loss(out, labels).item() * imgs.size(0)
                        preds = out.argmax(dim=1)
                    
                    correct += (preds == labels).sum().item()
                    total += imgs.size(0)
            
            vl /= max(1, total)
            v_acc = correct / max(1, total)
            
            print(f"Teacher Epoch {ep+1}/{config['teacher_epochs']} - "
                  f"Train Loss: {tl:.4f} - Val Loss: {vl:.4f} - Val Acc: {v_acc:.4f}")

            if v_acc > best_t_acc:
                best_t_acc = v_acc
                torch.save(teacher_model.state_dict(), "best_teacher.pth")

        if os.path.exists("best_teacher.pth"):
            teacher_model.load_state_dict(
                torch.load("best_teacher.pth", map_location=device)
            )
            print(f"✅ Loaded Best Teacher (Acc: {best_t_acc:.4f})")
        print("🎓 Teacher Training Complete.\n")

# -------------------------
# 3. STUDENT TRAINING
# -------------------------
print("\n🏆 Starting Student Training (Distillation + TDA)...")

final_model = EfficientNetWithTDAGating(
    backbone_name=config["backbone"],
    num_classes=config["num_classes"],
    use_cbam=config["use_cbam"],
    use_tda=config["use_tda"],
    tda_img_size=config["tda_img_size"],
    mlp_structure=config["tda_mlp_structure"]
).to(device)

final_optimizer = optim.AdamW(final_model.parameters(), lr=config["lr"])
final_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    final_optimizer,
    max_lr=config["lr"] * 5,
    steps_per_epoch=len(train_loader),
    epochs=config["epochs"]
)

final_criterion = DistillationLoss(
    base_loss,
    teacher_model=teacher_model,
    alpha_start=config["distill_alpha_start"],
    alpha_end=config["distill_alpha_end"],
    temperature=config["distill_temperature"]
)

final_trainer = Trainer(final_model, final_criterion, final_optimizer, final_scheduler, device, config)

train_losses = []
best_val_loss = float("inf")
patience = config["early_stop_patience"]
counter = 0

for epoch in range(config["epochs"]):
    train_loss = final_trainer.train_epoch(train_loader, epoch, config["epochs"])
    train_losses.append(train_loss)

    # Validation
    final_model.eval()
    val_loss, total = 0.0, 0
    preds, trues = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # --- UPDATED: Correct syntax for newer PyTorch ---
            with torch.amp.autocast('cuda'):
                outputs = final_model(imgs)

                if isinstance(final_criterion, DistillationLoss):
                    final_criterion.set_epoch(epoch, config["epochs"])
                    loss = final_criterion(outputs, imgs, labels)
                else:
                    loss = final_criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    val_loss /= max(1, total)
    acc = np.mean(np.array(preds) == np.array(trues))
    
    print(f"Student Epoch {epoch+1}/{config['epochs']} - "
          f"Train: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(final_model.state_dict(), "best_model.pth")
        print("✅ Model saved.")
    else:
        counter += 1
        print(f"⚠ No improvement ({counter}/{patience})")
        if counter >= patience:
            print("🛑 Early stopping.")
            break

# -------------------------
# 6. Evaluation & Grad-CAM
# -------------------------
if os.path.exists("best_model.pth"):
    final_model.load_state_dict(torch.load("best_model.pth", map_location=device))

preds, trues = final_trainer.evaluate(val_loader)
print(classification_report(trues, preds, target_names=class_names))

cm = confusion_matrix(trues, preds)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.show()

# -------------------------
# 7. FIXED GRAD-CAM
# -------------------------
def denormalize_image(img_tensor):
    """Undo ImageNet normalization and convert to uint8 HWC."""
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
    # --- UPDATED: No autocast for GradCAM backward pass usually safer ---
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

# Generate Grad-CAMs
os.makedirs(config["gradcam_out_dir"], exist_ok=True)
final_model.eval()

print("\n🔥 Generating Grad-CAMs...")
target_layer = get_gradcam_target_layer(final_model)
cls_seen = {i: False for i in range(config["num_classes"])}

dataset_ref = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
indices = val_ds.indices if isinstance(val_ds, Subset) else range(len(val_ds))

for ds_idx in indices:
    img, label = dataset_ref[ds_idx]
    if cls_seen[label]:
        continue
    cls_seen[label] = True
    
    cam = generate_gradcam(final_model, img, label, target_layer)
    cam = cv2.resize(cam, (config["image_size"], config["image_size"]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = denormalize_image(img)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    
    save_path = os.path.join(
        config["gradcam_out_dir"],
        f"gradcam_class_{class_names[label]}.jpg"
    )
    cv2.imwrite(save_path, overlay)
    print(f"✔ Saved Grad-CAM for class {class_names[label]} → {save_path}")

print("🎉 Process Complete!")
