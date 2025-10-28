import torch
from torchvision import models
import json

PTH_PATH = "/pth_path"         
LABELS_PATH = "imagesort_labels.json"     
PT_PATH  = "/pt_path"    
NUM_CLASSES = None                        

with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
    NUM_CLASSES = len(class_names)
print(f"✅ Loaded {NUM_CLASSES} classes from {LABELS_PATH}")

model = models.shufflenet_v2_x1_0(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
model.eval()

example_input = torch.randn(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)

traced_script_module.save(PT_PATH)
print(f"💾 Exported TorchScript model saved as: {PT_PATH}")
