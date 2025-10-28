import torch
from torchvision import models
import json

# ----------- CONFIG -----------
PTH_PATH = "/pth_path"          # your trained .pth model
LABELS_PATH = "imagesort_labels.json"      # labels file
PT_PATH  = "/pt_path"     # output file name
NUM_CLASSES = None                           # will be auto-loaded from labels

# ----------- LOAD LABELS -----------
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
    NUM_CLASSES = len(class_names)
print(f"✅ Loaded {NUM_CLASSES} classes from {LABELS_PATH}")

# ----------- LOAD MODEL -----------
model = models.shufflenet_v2_x1_0(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
model.eval()

# ----------- CONVERT TO TORCHSCRIPT -----------
example_input = torch.randn(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)

# ----------- SAVE AS .PT -----------
traced_script_module.save(PT_PATH)
print(f"💾 Exported TorchScript model saved as: {PT_PATH}")
