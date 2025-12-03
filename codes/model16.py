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

# --- Optuna Import ---
try:
    import optuna
    print(f"✅ Optuna available: v{optuna.__version__}")
except ImportError:
    raise ImportError("⚠ Optuna not found. Please run: pip install optuna")

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
    "img_root": "./diabetes/colored_images",
    
    # Batch Size 32 for A5000 Stability
    "batch_size": 32, 
    
    "backbone": "efficientnet_b2",
    "num_classes": 5,
    "epochs": 25,            
    "optuna_trials": 10,     
    "optuna_epochs": 8,      
    
    "lr": 1e-4, 
    
    # Both models will now use Attention
    "use_cbam": True,
    "use_distill": True,
    
    "teacher_backbone": "densenet121", 
    "train_teacher": True,
    "teacher_epochs": 15,

    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "B2_DenseNet_Attn_Distill_TDA",
    "image_size": 224,

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

    # --- TDA CONFIG ---
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
# CBAM Block (Channel + Spatial Attention)
# -------------------------
class CBAMBlock(nn.Module):
    """
    Channel Attention Module + Spatial Attention Module.
    We use this for BOTH Teacher and Student.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False)
        )
        # Spatial Attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 1. Channel Attention (Global Info)
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        
        c_out = torch.sigmoid(
            self.channel_gate(avg_pool) + self.channel_gate(max_pool)
        ).view(b, c, 1, 1)
        x = x * c_out
        
        # 2. Spatial Attention (Where to look)
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        s_out = torch.sigmoid(
            self.spatial_gate(torch.cat([avg_s, max_s], dim=1))
        )
        return x * s_out

# -------------------------
# TEACHER: DenseNet with Attention
# -------------------------
class DenseNetTeacher(nn.Module):
    """
    DenseNet121 + CBAM Attention Block at the bottleneck.
    """
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        # Load Pretrained DenseNet (Features Only)
        self.base = timm.create_model(backbone_name, pretrained=True, features_only=False)
        
        # Extract In-Features (DenseNet121 usually 1024)
        self.in_features = self.base.classifier.in_features
        
        # Remove original classifier
        self.base.reset_classifier(0) 
        
        # Add Attention Block
        self.attn = CBAMBlock(self.in_features)
        
        # New Classifier
        self.classifier = nn.Linear(self.in_features, num_classes)
        
    def forward(self, x):
        # Extract features (B, 1024, 7, 7)
        f = self.base.forward_features(x)
        
        # Apply Channel+Spatial Attention
        f = self.attn(f)
        
        # Global Pooling & Classify
        f = nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return self.classifier(f)

# -------------------------
# STUDENT: EfficientNet with TDA + Attention
# -------------------------
class EfficientNetWithTDAGating(nn.Module):
    def __init__(self, backbone_name, num_classes, use_cbam, use_tda, tda_img_size, mlp_structure):
        super().__init__()
        self.feature_extractor = timm.create_model(backbone_name, pretrained=True, features_only=True)
        out_channels = self.feature_extractor.feature_info[-1]['num_chs']
        
        # Student also gets the Attention Block
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
# 2. Teacher Setup (DenseNet121 + Attention)
# -------------------------
teacher_model = None
if config["use_distill"]:
    print(f"🏗 Initializing Teacher: {config['teacher_backbone']} (Pretrained + Attention)...")
    
    # --- CHANGED: Use Custom DenseNetTeacher Class ---
    teacher_model = DenseNetTeacher(
        backbone_name=config["teacher_backbone"],
        num_classes=config["num_classes"]
    ).to(device)
    
    print(f"✅ Teacher initialized: DenseNet121 + CBAM")

    if config["train_teacher"]:
        print("\n🎓 Starting Teacher Training (DenseNet + Attn)...")
        
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
# 3. OPTUNA OBJECTIVE
# -------------------------
def objective(trial):
    # --- A. Suggest Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    alpha_start = trial.suggest_float("distill_alpha_start", 0.1, 0.5)
    alpha_end = trial.suggest_float("distill_alpha_end", 0.5, 0.9)
    temperature = trial.suggest_float("distill_temperature", 2.0, 8.0)
    
    # Dynamic TDA MLP Structure
    n_layers = trial.suggest_int("tda_n_layers", 1, 3)
    mlp_structure = []
    for i in range(n_layers):
        out_dim = trial.suggest_categorical(f"tda_layer_{i}", [16, 32, 64, 128])
        mlp_structure.append(out_dim)
        
    # --- B. Build Model ---
    model = EfficientNetWithTDAGating(
        backbone_name=config["backbone"],
        num_classes=config["num_classes"],
        use_cbam=config["use_cbam"],
        use_tda=config["use_tda"],
        tda_img_size=config["tda_img_size"],
        mlp_structure=mlp_structure
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    criterion = DistillationLoss(
        base_loss,
        teacher_model=teacher_model,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        temperature=temperature
    )
    
    # --- C. Short Training Loop (with Pruning) ---
    search_epochs = config["optuna_epochs"]
    trainer = Trainer(model, criterion, optimizer, None, device, config)
    
    best_acc = 0.0
    
    for epoch in range(search_epochs):
        _ = trainer.train_epoch(train_loader, epoch, search_epochs)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        
        val_acc = correct / max(1, total)
        
        # Report for Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_acc = max(best_acc, val_acc)
        
    return best_acc

# -------------------------
# 4. RUN OPTUNA
# -------------------------
print("\n🚀 Starting Optuna Hyperparameter Search...")
print(f"   - Trials: {config['optuna_trials']}")
print(f"   - Search Epochs per Trial: {config['optuna_epochs']}")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=config["optuna_trials"])

print("\n✅ Optuna Search Complete!")
print("   - Best Trial:", study.best_trial.number)
print("   - Best Accuracy:", study.best_value)
print("   - Best Params:", study.best_params)

# Extract Best Params for Final Training
best_lr = study.best_params["lr"]
best_alpha_start = study.best_params["distill_alpha_start"]
best_alpha_end = study.best_params["distill_alpha_end"]
best_temp = study.best_params["distill_temperature"]

# Reconstruct MLP structure from flat params
best_mlp_structure = []
n_layers = study.best_params["tda_n_layers"]
for i in range(n_layers):
    best_mlp_structure.append(study.best_params[f"tda_layer_{i}"])

print(f"   - Best TDA Structure: {best_mlp_structure}")

# -------------------------
# 5. FINAL TRAINING (With Best Params)
# -------------------------
print("\n🏆 Starting FINAL TRAINING with Best Hyperparameters...")

# Re-Initialize Student with Best Architecture
final_model = EfficientNetWithTDAGating(
    backbone_name=config["backbone"],
    num_classes=config["num_classes"],
    use_cbam=config["use_cbam"],
    use_tda=config["use_tda"],
    tda_img_size=config["tda_img_size"],
    mlp_structure=best_mlp_structure
).to(device)

final_optimizer = optim.AdamW(final_model.parameters(), lr=best_lr)
final_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    final_optimizer,
    max_lr=best_lr * 5,
    steps_per_epoch=len(train_loader),
    epochs=config["epochs"]
)

final_criterion = DistillationLoss(
    base_loss,
    teacher_model=teacher_model,
    alpha_start=best_alpha_start,
    alpha_end=best_alpha_end,
    temperature=best_temp
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
            outputs = final_model(imgs)

            # Recalculate loss for validation monitoring
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
    
    print(f"Final Epoch {epoch+1}/{config['epochs']} - "
          f"Train: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(final_model.state_dict(), "best_model_optuna.pth")
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
if os.path.exists("best_model_optuna.pth"):
    final_model.load_state_dict(torch.load("best_model_optuna.pth", map_location=device))

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
