import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
from collections import defaultdict
from PIL import Image

# --- CONFIG ---
DATA_DIR = "processed_data"
MODEL_PATH = "models/resnet18_scratch_100.pth"
BATCH_SIZE = 32
EPOCHS = 35  # Needs more epochs since training from scratch
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# --- UTILS ---
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MechanicalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

def get_grouped_split(root_dir):
    """
    Groups quadrants by original image to PREVENT LEAKAGE.
    """
    dataset = datasets.ImageFolder(root_dir)
    classes = dataset.classes
    grouped = defaultdict(list)
    
    for path, label in dataset.samples:
        # Group by parent image name
        parent = os.path.basename(path).rsplit('_', 1)[0]
        grouped[parent].append((path, label))
        
    all_groups = list(grouped.values())
    random.shuffle(all_groups)
    
    # 85% Train / 15% Val
    split = int(0.85 * len(all_groups))
    train_samples = [x for g in all_groups[:split] for x in g]
    val_samples = [x for g in all_groups[split:] for x in g]
    
    print(f"Total Quadrants: {len(dataset)} | Train: {len(train_samples)} | Val: {len(val_samples)}")
    return train_samples, val_samples, classes

# --- TRAIN ---
def train():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)
    print(f"🚀 Training ResNet18 FROM SCRATCH on {DEVICE}...")

    # Standardize transforms for Synthetic Data
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # Use 0.5 mean/std for training from scratch on synthetic data
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_files, val_files, classes = get_grouped_split(DATA_DIR)
    
    train_loader = DataLoader(MechanicalDataset(train_files, train_tf), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(MechanicalDataset(val_files, val_tf), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- RULE COMPLIANCE: FROM SCRATCH ---
    # We load ResNet18 but explicitly set weights=None
    print("Initialize Standard ResNet18 (Weights=None)...")
    model = models.resnet18(weights=None) 
    
    # Modify the final layer for our 5 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  --> ⭐ Best Model Saved!")

    print(f"✅ Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()