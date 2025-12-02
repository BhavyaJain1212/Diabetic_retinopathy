import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import random
from collections import defaultdict, Counter
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Import GradCAM and utilities
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BalancedSplitDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', 
                 train_frac=0.7, val_frac=0.15, test_frac=0.15,
                 no_dr_undersample=1000, oversample_target_train=750, random_seed=42):
        super().__init__()
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.random_seed = random_seed

        class_to_idx = self.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        targets = np.array([label for _, label in self.dataset.samples])
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)

        # Undersample 'No_DR'
        cls_name = 'No_DR'
        cls_idx = class_to_idx[cls_name]
        idxs = class_indices[cls_idx]
        if len(idxs) > no_dr_undersample:
            np.random.seed(self.random_seed)
            class_indices[cls_idx] = list(np.random.choice(idxs, no_dr_undersample, replace=False))

        combined_indices = []
        for idxs in class_indices.values():
            combined_indices.extend(idxs)

        np.random.seed(self.random_seed)
        np.random.shuffle(combined_indices)

        targets_new = [targets[i] for i in combined_indices]
        n_total = len(combined_indices)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        train_indices = combined_indices[:n_train]
        val_indices = combined_indices[n_train:n_train + n_val]
        test_indices = combined_indices[n_train + n_val:]

        train_labels = [targets[i] for i in train_indices]

        # Compute class weights
        counts = Counter(train_labels)
        total_train = len(train_labels)
        class_weights = []
        for cls_i in range(len(class_to_idx)):
            count = counts[cls_i]
            if count > 0:
                class_weight = total_train / (len(class_to_idx) * count)
            else:
                class_weight = 0.0
            class_weights.append(class_weight)
        self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        # Oversample underrepresented classes
        train_class_indices = defaultdict(list)
        for idx, label in zip(train_indices, train_labels):
            train_class_indices[label].append(idx)

        oversample_classes = ['Severe', 'Proliferate_DR', 'Mild']
        oversample_class_indices = {class_to_idx[c]: c for c in oversample_classes}

        np.random.seed(self.random_seed)
        new_train_indices = []
        for cls_idx, inds in train_class_indices.items():
            if cls_idx in oversample_class_indices:
                if len(inds) < oversample_target_train:
                    extra = list(np.random.choice(inds, oversample_target_train - len(inds), replace=True))
                    inds += extra
            new_train_indices.extend(inds)
        np.random.shuffle(new_train_indices)

        self.split_indices = {'train': new_train_indices, 'val': val_indices, 'test': test_indices}
        self.indices = self.split_indices[split]
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        def print_class_counts(name, indices):
            labels = [targets[i] for i in indices]
            counts = Counter(labels)
            print(f"\nClass-wise {name} split samples:")
            for cls_i, cls_name_print in sorted(idx_to_class.items()):
                print(f"  {cls_name_print}: {counts.get(cls_i, 0)}")
        print_class_counts('Train', new_train_indices)
        print_class_counts('Validation', val_indices)
        print_class_counts('Test', test_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# Settings
data_dir = '../dataset/india_dataset/colored_images'
num_classes = 5
input_size = 224
batch_size = 16
learning_rate = 1e-4
num_epochs = 30
patience = 7

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

train_dataset = BalancedSplitDataset(data_dir, transform=train_transforms, split='train')
val_dataset = BalancedSplitDataset(data_dir, transform=val_test_transforms, split='val')
test_dataset = BalancedSplitDataset(data_dir, transform=val_test_transforms, split='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model setup
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights_tensor.to(device))

# Progressive cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

best_val_loss = float('inf')
early_stop_counter = 0
best_model_wts = None

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = 100. * val_correct / val_total
    print(f"Validation loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# Test evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# Confusion matrix visualization
cm = confusion_matrix(all_labels, all_preds)
class_names = [k for k, v in sorted(datasets.ImageFolder(root=data_dir).class_to_idx.items(), key=lambda x: x[1])]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Grad-CAM visualization
random_idx = random.randint(0, len(test_dataset) - 1)
img_tensor, true_label = test_dataset[random_idx]

# Denormalize for visualization
img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = std * img_np + mean
img_np = np.clip(img_np, 0, 1)

input_tensor = img_tensor.unsqueeze(0).to(device)
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

outputs = model(input_tensor)
predicted_class = outputs.argmax(dim=1).item()

targets = [ClassifierOutputTarget(predicted_class)]
grayscale_cam = cam(input_tensor, targets=targets)[0, :]

visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f'Original Image - True: {class_names[true_label]}')
plt.axis('off')
plt.imshow(img_np)

plt.subplot(1, 2, 2)
plt.title(f'Grad-CAM - Predicted: {class_names[predicted_class]}')
plt.axis('off')
plt.imshow(visualization)
plt.show()


# unfreezing all layers
# progressive learnign rate
# rest all is same as last time. 