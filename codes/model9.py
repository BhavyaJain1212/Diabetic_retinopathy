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
import torch.nn.functional as F
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
    "csv_path": "train.csv",           # if missing, code will fallback to ImageFolder on img_root
    "img_root": "./diabetes/colored_images",        # folder root OR ImageFolder root (class subfolders)
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 25,
    "lr": 1e-4,
    "loss_fn": "CrossEntropyLoss",
    "early_stop_patience": 5,
    "gradcam_out_dir": "HETNet_GradCAM",
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

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
# (Corrected magic methods __init__, __len__, __getitem__)
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

class_names = dataset.classes if use_imagefolder else ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

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
# HET-Net Model
# -------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, local_dim, global_dim, fused_dim):
        super().__init__()
        self.to_q = nn.Linear(global_dim, fused_dim, bias=False)
        self.to_k = nn.Linear(local_dim, fused_dim, bias=False)
        self.to_v = nn.Linear(local_dim, fused_dim, bias=False)
        self.scale = fused_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, local_feat, global_feat):
        # local_feat: (B, N, local_dim)
        # global_feat: (B, M, global_dim)
        
        Q = self.to_q(global_feat)
        K = self.to_k(local_feat)
        V = self.to_v(local_feat)
        
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = self.softmax(attn)
        out = torch.bmm(attn, V)  
        return out

class HETNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout=0.3):
        super().__init__()
        # Local feature extractor
        self.local_backbone = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=True, features_only=True)
        local_out_ch = self.local_backbone.feature_info[-1]['num_chs']
        
        # Global feature extractor
        self.global_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True)
        
        print(f"✓ Local backbone output channels (reported): {local_out_ch}")
        
        # Test actual output shape with a dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            local_test = self.local_backbone(dummy_input)[-1]
            global_test = self.global_backbone(dummy_input)[-1]
            
            print(f"✓ Local backbone actual output shape: {local_test.shape}")
            print(f"✓ Global backbone actual output shape: {global_test.shape}")
            
            # Detect the channel dimension for global backbone
            # Swin outputs [B, H, W, C] format, not [B, C, H, W]
            if len(global_test.shape) == 4:
                if global_test.shape[1] < global_test.shape[3]:
                    # Likely [B, H, W, C] format
                    actual_global_ch = global_test.shape[3]
                    self.global_format = "BHWC"
                    print(f"✓ Global backbone uses BHWC format, channels: {actual_global_ch}")
                else:
                    # Standard [B, C, H, W] format
                    actual_global_ch = global_test.shape[1]
                    self.global_format = "BCHW"
                    print(f"✓ Global backbone uses BCHW format, channels: {actual_global_ch}")
            
            actual_local_ch = local_test.shape[1]
        
        # Use actual channel dimensions
        self.local_proj = nn.Linear(actual_local_ch, 256)
        self.global_proj = nn.Linear(actual_global_ch, 256)
        
        self.fusion = CrossAttentionFusion(local_dim=256, global_dim=256, fused_dim=256)
        
        # Use LayerNorm instead of BatchNorm to avoid batch size issues
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Extract local features
        local_feats = self.local_backbone(x)[-1]  # (B, C_l, H_l, W_l)
        B, C_l, H_l, W_l = local_feats.shape
        local_feats_flat = local_feats.flatten(2).permute(0, 2, 1)  # (B, H_l*W_l, C_l)
        local_feats_proj = self.local_proj(local_feats_flat)  # (B, H_l*W_l, 256)
        
        # Extract global features
        global_feats = self.global_backbone(x)[-1]
        
        # Handle different output formats
        if self.global_format == "BHWC":
            # Swin format: [B, H, W, C] -> permute to [B, C, H, W]
            B, H_g, W_g, C_g = global_feats.shape
            global_feats = global_feats.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            # Standard format: [B, C, H, W]
            B, C_g, H_g, W_g = global_feats.shape
        
        # Now global_feats is in standard [B, C, H, W] format
        global_feats_flat = global_feats.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        global_feats_proj = self.global_proj(global_feats_flat)  # (B, H*W, 256)

        # Cross-attention fusion
        fused = self.fusion(local_feats_proj, global_feats_proj)  # (B, M, 256)
        
        # Global pooling and classification
        fused_pooled = fused.mean(dim=1)  # (B, 256)
        fused_pooled = self.norm(fused_pooled)  # LayerNorm works with any batch size
        fused_pooled = self.dropout(fused_pooled)
        
        logits = self.classifier(fused_pooled)  # (B, num_classes)
        return logits

# -------------------------
# Loss function
# -------------------------
loss_name = config["loss_fn"].lower()
if loss_name in ("crossentropyloss", "crossentropy"):
    criterion = nn.CrossEntropyLoss()
elif loss_name == "mse":
    criterion = nn.MSELoss()
elif loss_name in ("l1loss", "l1"):
    criterion = nn.L1Loss()
elif loss_name == "huberloss":
    criterion = nn.SmoothL1Loss()
else:
    raise ValueError(f"Unsupported loss function: {config['loss_fn']}")

# -------------------------
# Accuracy calculation
# -------------------------
def accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)

# -------------------------
# Trainer Class
# -------------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
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
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)
        return running_loss / total, running_correct / total

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0
        preds_list, trues_list = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                total += imgs.size(0)
                preds_list.extend(preds.cpu().numpy().tolist())
                trues_list.extend(labels.cpu().numpy().tolist())
        return running_loss / total, running_correct / total, preds_list, trues_list

# -------------------------
# Initialize model, optimizer, scheduler
# -------------------------
model = HETNet(num_classes=config["num_classes"]).to(device)
print("✅ Student model with HET-Net architecture created")

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = None
if len(train_loader) > 0:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"] * 5,
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"]
    )

trainer = Trainer(model, criterion, optimizer, scheduler, device, config)

# -------------------------
# Training loop with epoch stats + early stopping
# -------------------------
train_losses, train_accs = [], []
val_losses, val_accs = [], []

best_val_loss = float("inf")
patience_counter = 0
patience = config["early_stop_patience"]

for epoch in range(config["epochs"]):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc, _, _ = trainer.evaluate(val_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{config['epochs']} - "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

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
# Final classification report and confusion matrix
# -------------------------
print("\n🎯 Final evaluation on validation set...")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

preds, trues = [], []
_, _, preds, trues = trainer.evaluate(val_loader)

print("\nFinal Classification Report:")
print(classification_report(trues, preds, digits=4, target_names=class_names))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

os.makedirs(config["gradcam_out_dir"], exist_ok=True)
cm_save_path = os.path.join(config["gradcam_out_dir"], "confusion_matrix.png")
plt.savefig(cm_save_path)
plt.close()
print(f"✅ Confusion matrix saved to {cm_save_path}")

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

print("\n🎉 Training complete. Use Grad-CAM visualization by executing your Grad-CAM code with this model and saved weights.")

