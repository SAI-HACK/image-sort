import os, time, json, io, requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
SAVE_DIR    = "your_save_directory_goes_here"
MODEL_NAME  = "imagesort.pth"
LABELS_JSON = "imagesort_labels.json"

EPOCHS      = 6
BATCH_SIZE  = 32
LR          = 1e-3
NUM_WORKERS = 2
HF_SAMPLES  = 800  # max per dataset

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- ACTIVE DATASETS MAP ----------------
DATASET_MAP = {
    "people": [
        "FacePerceiver/laion-face",
        "schirrmacher/humans"
    ],
    "memes": [
        "not-lain/meme-dataset",
        "sin3142/memes-500"
    ],
    "documents": [
        "nielsr/funsd"
    ],
    "handwritten": [
        "agomberto/FrenchCensus-handwritten-texts"
    ],
    "advertisements": [
        "PeterBrendan/AdImageNet",
        "multimodalart/vintage-ads"
    ],
    "digitalnotes": [
        "HuggingFaceM4/DocumentVQA"
    ]
}

# ---------------- TRANSFORMS ----------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- HF WRAPPER ----------------
class HFDataset(Dataset):
    def __init__(self, name, category_idx, transform=None, max_samples=HF_SAMPLES):
        self.transform = transform
        self.samples = []
        self.category_idx = category_idx

        print(f"🔹 Loading {name} (max {max_samples})...")
        try:
            ds = load_dataset(name, split="train", streaming=True)
            for i, ex in enumerate(tqdm(ds, total=max_samples, desc=name[:25])):
                if i >= max_samples:
                    break
                img = None
                try:
                    if "image" in ex and isinstance(ex["image"], Image.Image):
                        img = ex["image"]
                    elif "img" in ex and isinstance(ex["img"], Image.Image):
                        img = ex["img"]
                    elif "image_url" in ex:
                        r = requests.get(ex["image_url"], timeout=5)
                        img = Image.open(io.BytesIO(r.content))
                except Exception:
                    continue
                if img:
                    self.samples.append(img)
        except Exception as e:
            print(f"⚠️ Could not load {name}: {e}")

        print(f"✅ {len(self.samples)} images from {name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx].convert("RGB")
        label = self.category_idx
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------- LOAD DATASETS ----------------
hf_datasets = []
class_names = list(DATASET_MAP.keys())
print(f"\n📂 Loading Hugging Face datasets for categories: {class_names}")

for idx, cat in enumerate(class_names):
    found = False
    for name in DATASET_MAP[cat]:
        ds = HFDataset(name, category_idx=idx, transform=train_tfms)
        if len(ds) > 50:
            hf_datasets.append(ds)
            print(f"✅ Using '{name}' for '{cat}' ({len(ds)} samples)\n")
            found = True
            break
        else:
            print(f"❌ Not enough data in '{name}'")
    if not found:
        print(f"⚠️ No usable dataset found for '{cat}' — skipping it.\n")

# ---------------- COMBINE & SPLIT ----------------
combined = ConcatDataset(hf_datasets)
train_size = int(0.8 * len(combined))
val_size   = len(combined) - train_size
train_set, val_set = random_split(combined, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"\n📊 Total training samples: {len(train_set)}, validation: {len(val_set)}")

# ---------------- MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🧠 Using device: {device}")

model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ---------------- TRAIN ----------------
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, correct, total = 0, 0, 0

    print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # validation
    val_correct, val_total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    dt = time.time() - t0
    print(f"📈 TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | Time={dt:.1f}s")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
        with open(os.path.join(SAVE_DIR, LABELS_JSON), "w") as f:
            json.dump(class_names, f)
        print(f"💾 Saved best model (val_acc={best_acc:.3f})")

print(f"\n✅ Training complete. Best validation accuracy = {best_acc:.3f}")
