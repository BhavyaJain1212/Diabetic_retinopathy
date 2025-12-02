import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from collections import defaultdict, Counter

# Settings
data_dir = '../dataset/india_dataset/colored_images'
num_classes = 5
batch_size = 32
learning_rate = 1e-3
num_epochs = 15
input_size = 224
patience = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
targets = np.array([label for _, label in full_dataset.samples])

# Collect indices per class
class_indices = defaultdict(list)
for idx, label in enumerate(targets):
    class_indices[label].append(idx)

# Undersample >715
for label, idxs in class_indices.items():
    if len(idxs) > 715:
        class_indices[label] = list(np.random.choice(idxs, 715, replace=False))
# Oversample <500
for label, idxs in class_indices.items():
    if len(idxs) < 500:
        resample_needed = 500 - len(idxs)
        extra = list(np.random.choice(idxs, resample_needed, replace=True))
        class_indices[label].extend(extra)

# Combine balanced indices and shuffle
balanced_indices = []
for idxs in class_indices.values():
    balanced_indices.extend(idxs)
np.random.shuffle(balanced_indices)

# Split balanced indices into train, val, test
n_total = len(balanced_indices)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_idxs = balanced_indices[:n_train]
val_idxs = balanced_indices[n_train:n_train + n_val]
test_idxs = balanced_indices[n_train + n_val:]

train_dataset = Subset(full_dataset, train_idxs)
val_dataset = Subset(full_dataset, val_idxs)
test_dataset = Subset(full_dataset, test_idxs)

# Function to count class distribution of a Subset dataset
def count_classes(subset):
    labels = []
    for i in range(len(subset)):
        _, label = subset[i]
        labels.append(label)
    return Counter(labels)

# Print class distributions after balancing per dataset
train_class_counts = count_classes(train_dataset)
val_class_counts = count_classes(val_dataset)
test_class_counts = count_classes(test_dataset)

class_names = [k for k, v in sorted(full_dataset.class_to_idx.items(), key=lambda x: x[1])]

print("Class distributions after balancing and splitting:")
print("Train set:")
for class_idx in range(num_classes):
    print(f"  {class_names[class_idx]}: {train_class_counts[class_idx]}")
print("Validation set:")
for class_idx in range(num_classes):
    print(f"  {class_names[class_idx]}: {val_class_counts[class_idx]}")
print("Test set:")
for class_idx in range(num_classes):
    print(f"  {class_names[class_idx]}: {test_class_counts[class_idx]}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
model = models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    epoch_loss = running_loss / n_train
    epoch_acc = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] | Train loss: {epoch_loss:.4f} | Train acc: {epoch_acc:.2f}%')

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
    val_loss /= n_val
    val_acc = 100. * val_correct / val_total
    print(f'Validation loss: {val_loss:.4f} | Validation acc: {val_acc:.2f}%')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

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
print(f"\nTest accuracy on test set: {test_acc*100:.2f}%")

f1_scores = f1_score(all_labels, all_preds, average=None)
print("\nPer-class F1 scores on test set:")
for idx, name in enumerate(class_names):
    print(f"{name}: {f1_scores[idx]:.4f}")

torch.save(model.state_dict(), 'resnext_dr_classification_2.pth')

print("Class-to-index mapping (label names):", full_dataset.class_to_idx)
