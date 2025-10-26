import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
LOCAL_DATA_DIR = "/mnt/d/trainmodelcnn/dataset"
OUTPUT_DIR = "/mnt/d/huggingfacecnn"
HF_DATASETS = [
    "sin3142/memes-1500",
    "sin3142/people-1500",
    "sin3142/advertisements-1500",
    "sin3142/handwritten-notes",
    "sin3142/digital-documents"
]
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 6  # adjust if you have more/less

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRANSFORMS
# ============================================================
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# LOCAL DATASETS
# ============================================================
train_ds_local = datasets.ImageFolder(os.path.join(LOCAL_DATA_DIR, "train"), transform=train_tfms)
val_ds_local   = datasets.ImageFolder(os.path.join(LOCAL_DATA_DIR, "val"), transform=val_tfms)

# ============================================================
# HUGGING FACE STREAMING DATASETS WRAPPER
# ============================================================
class HFDataset(torch.utils.data.IterableDataset):
    def __init__(self, name, split="train", transform=None):
        self.ds = load_dataset(name, split=split, streaming=True)
        self.transform = transform

    def __iter__(self):
        for example in self.ds:
            try:
                img = example["image"].convert("RGB")
                label = example.get("label", 0)
                if self.transform:
                    img = self.transform(img)
                yield img, label
            except Exception as e:
                print("⚠️ Skipping a sample:", e)

# ============================================================
# COMBINE LOCAL + HF DATASETS
# ============================================================
train_loaders = []
train_loaders.append(DataLoader(train_ds_local, batch_size=BATCH_SIZE, shuffle=True))
for hf_name in HF_DATASETS:
    hf_train = HFDataset(hf_name, split="train", transform=train_tfms)
    train_loaders.append(DataLoader(hf_train, batch_size=BATCH_SIZE))

val_loader = DataLoader(val_ds_local, batch_size=BATCH_SIZE)

# ============================================================
# MODEL SETUP
# ============================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = CNNClassifier(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for loader in train_loaders:
        for imgs, labels in tqdm(loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loaders)
    print(f"✅ Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"🎯 Validation Accuracy: {acc*100:.2f}%")

# ============================================================
# SAVE MODEL
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "huggingface_cnn_trained.pth")
torch.save(model.state_dict(), save_path)
print(f"\n✅ Model saved successfully at: {save_path}")
