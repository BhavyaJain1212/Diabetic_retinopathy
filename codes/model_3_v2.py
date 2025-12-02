import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict, Counter
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

class BalancedSplitDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_frac=0.7, val_frac=0.15, test_frac=0.15, undersample_threshold=715, oversample_target_train=500, random_seed=42):
        super().__init__()
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.random_seed = random_seed

        # Map class names to indices (needed for class-specific operations)
        class_to_idx = self.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Get original indices per class
        targets = np.array([label for _, label in self.dataset.samples])
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)

        # Step 1: Undersample 'moderate' and 'no_dr' classes to 670 each (if more)
        for cls_name in ['Moderate', 'No_DR']:
            cls_idx = class_to_idx[cls_name]
            idxs = class_indices[cls_idx]
            if len(idxs) > undersample_threshold:
                np.random.seed(self.random_seed)
                class_indices[cls_idx] = list(np.random.choice(idxs, undersample_threshold, replace=False))

        # Combine all indices after undersampling
        combined_indices = []
        for idxs in class_indices.values():
            combined_indices.extend(idxs)

        # Shuffle combined indices for split reproducibility
        np.random.seed(self.random_seed)
        np.random.shuffle(combined_indices)

        # Extract labels for combined indices to split accordingly
        combined_labels = [targets[i] for i in combined_indices]
        combined_labels = np.array(combined_labels)

        # Step 2: Perform train/val/test split (70/15/15)
        n_total = len(combined_indices)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val

        train_indices = combined_indices[:n_train]
        val_indices = combined_indices[n_train:n_train + n_val]
        test_indices = combined_indices[n_train + n_val:]

        # For train set subset, get labels again
        train_labels = [targets[i] for i in train_indices]

        # Step 3: Oversample smaller classes (severe, profilate_dr, mild) in TRAIN split to 500 images each
        train_class_indices = defaultdict(list)
        for idx, label in zip(train_indices, train_labels):
            train_class_indices[label].append(idx)

        # Classes to oversample:
        oversample_classes = ['Severe', 'Proliferate_DR', 'Mild']
        oversample_class_indices = {class_to_idx[c]: c for c in oversample_classes}

        np.random.seed(self.random_seed)
        new_train_indices = []
        for cls_idx, inds in train_class_indices.items():
            if cls_idx in oversample_class_indices:
                # Oversample to 500 if less than 500
                n_samples = len(inds)
                if n_samples < oversample_target_train:
                    extra = list(np.random.choice(inds, oversample_target_train - n_samples, replace=True))
                    inds = inds + extra
            new_train_indices.extend(inds)

        np.random.seed(self.random_seed)
        np.random.shuffle(new_train_indices)

        # Save split indices for __getitem__
        self.split_indices = {
            'train': new_train_indices,
            'val': val_indices,
            'test': test_indices
        }

        if split not in self.split_indices:
            raise ValueError("split must be 'train', 'val' or 'test'")

        self.indices = self.split_indices[split]

        # Store for external use
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        # Print class-wise sample count per split
        def print_class_counts(name, indices):
            labels_ = [targets[i] for i in indices]
            c = Counter(labels_)
            print(f"\nClass-wise {name} split samples:")
            for cls_i, cls_name_print in sorted(idx_to_class.items()):
                print(f"  {cls_name_print}: {c.get(cls_i, 0)}")

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


# Settings and transforms
data_dir = '../dataset/india_dataset/colored_images'
num_classes = 5
input_size = 224
batch_size = 32
learning_rate = 1e-3
num_epochs = 15
patience = 5

train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_test_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Create datasets per split (prints class-wise counts internally)
train_dataset = BalancedSplitDataset(data_dir, transform=train_transforms, split='train')
val_dataset = BalancedSplitDataset(data_dir, transform=val_test_transforms, split='val')
test_dataset = BalancedSplitDataset(data_dir, transform=val_test_transforms, split='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model setup
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

train_labels = [label for _, label in train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
early_stop_counter = 0
best_model_wts = None

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

from sklearn.metrics import classification_report
class_names = [k for k, v in sorted(datasets.ImageFolder(root=data_dir).class_to_idx.items(), key=lambda x: x[1])]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

torch.save(model.state_dict(), 'resnext_balanced_split_augmented.pth')
