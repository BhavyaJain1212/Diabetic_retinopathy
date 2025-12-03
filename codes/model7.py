import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision.utils import save_image
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# ===================================================
# CBAM with Refinement Block
# ===================================================
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
        b, c, _, _ = x.size()
        avg = torch.mean(x, dim=(2,3))
        mx, _ = torch.max(x.view(b, c, -1), dim=2)
        y = self.mlp(avg) + self.mlp(mx)
        return torch.sigmoid(y).view(b, c, 1, 1)

class SpatialGateMultiK(nn.Module):
    def __init__(self, kernel_sizes=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(2,1,k,padding=(k-1)//2,bias=False) for k in kernel_sizes])
        self.bns = nn.ModuleList([nn.BatchNorm2d(1) for _ in kernel_sizes])
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        outs = [torch.sigmoid(bn(conv(cat))) for conv,bn in zip(self.convs,self.bns)]
        return sum(outs)/len(outs)

class CBAM_MultiK(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cg = ChannelGate(channels)
        self.sg = SpatialGateMultiK()
    def forward(self, x):
        x = x * self.cg(x)
        x = x * self.sg(x)
        return x

# ===================================================
# Refinement Block
# ===================================================
class RefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.refine(x))  # residual refinement

# ===================================================
# CONFIG
# ===================================================
DATA_DIR = "./diabetes/colored_images"
OUTPUT_DIR = "B2_B5_Distill"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gradcam_cbam_refine"), exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================
# DATA
# ===================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===================================================
# MODEL: EfficientNet + CBAM + Refinement
# ===================================================
class EfficientNet_CBAM_Refine(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = models.efficientnet_b2(weights="IMAGENET1K_V1")
        self.features = base.features
        self.cbam = CBAM_MultiK(1408)
        self.refine = RefinementBlock(1408)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(1408, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.refine(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

model = EfficientNet_CBAM_Refine(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===================================================
# TRAINING
# ===================================================
best_val_acc = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    val_acc = correct / total
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "cbam_refine_best.pth"))
        print("✅ Best model (CBAM + Refine) saved!")

# ===================================================
# LOSS PLOTS
# ===================================================
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Training and Validation Loss (CBAM + Refinement)")
plt.savefig(os.path.join(OUTPUT_DIR, "cbam_refine_loss_curve.png"))

# ===================================================
# EVALUATION
# ===================================================
print("Classification Report:")
print(classification_report(labels_all, preds_all, target_names=class_names))

cm = confusion_matrix(labels_all, preds_all)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - CBAM + Refinement")
plt.savefig(os.path.join(OUTPUT_DIR, "cbam_refine_confusion_matrix.png"))

# ===================================================
# GRAD-CAM
# ===================================================
cam_extractor = GradCAM(model)
model.eval()
for i in range(min(5, len(val_ds))):
    img, label = val_ds[i]
    input_tensor = img.unsqueeze(0).to(DEVICE)
    out = model(input_tensor)
    pred_class = out.argmax(dim=1).item()

    activation_map = cam_extractor(pred_class, out)
    heatmap = activation_map[0].squeeze().cpu().numpy()
    heatmap_img = transforms.ToPILImage()(img)
    result = overlay_mask(heatmap_img.convert("RGB"),
                          Image.fromarray((heatmap * 255).astype(np.uint8)),
                          alpha=0.5)
    result.save(os.path.join(OUTPUT_DIR, "gradcam_cbam_refine",
                             f"sample_{i}_true{label}_pred{pred_class}.png"))

print("🎉 CBAM + Refinement training complete! Outputs saved to:", OUTPUT_DIR)
