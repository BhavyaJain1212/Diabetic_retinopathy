# 1. Necessary Imports
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter

# 2. Setup and Configuration
def setup():
    """Initializes device, paths, and class names."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_dir = './diabetes/colored_images/'
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"The directory '{base_dir}' was not found. Please ensure it exists.")
        
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    return device, base_dir, class_names, num_classes


# 3. Data Loading and Preprocessing
def get_dataloaders(base_dir, class_names):
    """Creates stratified data loaders with weighted sampling for imbalance."""
    
    # Define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load full dataset
    full_dataset = datasets.ImageFolder(base_dir)
    targets = full_dataset.targets

    # Stratified split (70/15/15)
    sss_train_valtest = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_indices, val_test_indices = next(sss_train_valtest.split(np.zeros(len(targets)), targets))
    val_test_targets = np.array(targets)[val_test_indices]
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_sub_indices, test_sub_indices = next(sss_val_test.split(np.zeros(len(val_test_targets)), val_test_targets))
    val_indices = val_test_indices[val_sub_indices]
    test_indices = val_test_indices[test_sub_indices]

    # Create subsets
    train_dataset = Subset(datasets.ImageFolder(base_dir, transform=data_transforms['train']), train_indices)
    val_dataset = Subset(datasets.ImageFolder(base_dir, transform=data_transforms['val']), val_indices)
    test_dataset = Subset(datasets.ImageFolder(base_dir, transform=data_transforms['val']), test_indices)

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test images.")

    # ---- HANDLE CLASS IMBALANCE ----
    # Extract labels from train subset
    train_labels = [full_dataset.targets[i] for i in train_indices]
    class_counts = Counter(train_labels)
    print(f"Training class distribution: {dict(class_counts)}")

    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    }

    return dataloaders


# 4. Model Training and Evaluation
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, class_names):
    """Trains the model with early stopping."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloaders['train'], desc="Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        val_loss, val_acc, _ = evaluate_model(model, dataloaders['val'], criterion, device, class_names)

        print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_convnext_model.pth')
            print("Validation loss improved. Saving model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    model.load_state_dict(torch.load('best_convnext_model.pth'))
    return model


def evaluate_model(model, dataloader, criterion, device, class_names, is_test=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_labels) == np.array(all_preds))
    report_dict = {'true_labels': all_labels, 'pred_labels': all_preds}

    if is_test:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    return epoch_loss, epoch_acc, report_dict


# 5. Grad-CAM (same as before)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()


def visualize_cam(model, device, base_dir, class_names):
    output_dir = 'ConvNeXt_CAM'
    os.makedirs(output_dir, exist_ok=True)

    target_layer = model.features[-1]  # for ConvNeXt
    grad_cam = GradCAM(model, target_layer)

    vis_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        img_name = random.choice(os.listdir(class_dir))
        img_path = os.path.join(class_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        input_tensor = vis_transform(img).unsqueeze(0).to(device)

        heatmap = grad_cam(input_tensor)

        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Original: {class_name}')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Grad-CAM: {class_name}')
        ax2.axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'cam_{class_name}.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Saved Grad-CAM for {class_name} to {save_path}")


# 6. Main Execution Block
if __name__ == '__main__':
    device, base_dir, class_names, num_classes = setup()
    dataloaders = get_dataloaders(base_dir, class_names)

    # Load ConvNeXt-Tiny
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, class_names=class_names)

    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_acc, test_report_dict = evaluate_model(trained_model, dataloaders['test'], criterion, device, class_names, is_test=True)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nFinal Test Classification Report:")
    print(classification_report(test_report_dict['true_labels'], test_report_dict['pred_labels'], target_names=class_names, digits=4))

    print("\n--- Generating Grad-CAM Visualizations ---")
    visualize_cam(trained_model, device, base_dir, class_names)
