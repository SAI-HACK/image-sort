import os
import torch
from torchvision import transforms
from PIL import Image
import shutil
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup

# ---- Globals ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("imagesort_labels.json", "r") as f:
    class_names = json.load(f)

# ---- Load TorchScript model ----
MODEL_PATH = "imagesort_shufflenetv2.pt"
print(f"Loading TorchScript model from {MODEL_PATH} ...")

model = torch.jit.load(MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# ---- Image transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Sorting Function ----
def sort_whatsapp_images():
    # Possible source folders
    possible_src = [
        os.path.expanduser("/mnt/c/whatsappp"),  # fallback for PC testing
        "/storage/emulated/0/WhatsApp/Media/WhatsApp Images",  # Android path
    ]

    src_dir = None
    for path in possible_src:
        if os.path.exists(path):
            src_dir = path
            break

    if src_dir is None:
        raise FileNotFoundError("❌ Could not find WhatsApp folder or fallback data folder.")

    # Sort images directly within the source folder
    print(f"Sorting images inside {src_dir} ...")

    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Skip already sorted images (inside category subfolders)
        if any(fname.lower().startswith(cat.lower()) for cat in class_names):
            continue

        try:
            img = Image.open(fpath).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(x)
                pred_class = class_names[preds.argmax(1).item()]

            # Create category subfolder inside same directory
            out_dir = os.path.join(src_dir, pred_class)
            os.makedirs(out_dir, exist_ok=True)

            # Move (not copy) image into predicted folder
            shutil.move(fpath, os.path.join(out_dir, fname))

            print(f"Moved {fname} → {pred_class}")

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    return True

# ---- Kivy UI ----
class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 20
        self.spacing = 20

        self.label = Label(text="IMAGE SORT",
                           font_size=24,
                           size_hint=(1, 0.2),
                           color=(0.2, 0.6, 1, 1))
        self.add_widget(self.label)

        self.sort_btn = Button(text="Sort WhatsApp Images (In-Place)",
                               font_size=20,
                               size_hint=(1, 0.3),
                               background_color=(0.2, 0.6, 1, 1))
        self.sort_btn.bind(on_press=self.sort_images)
        self.add_widget(self.sort_btn)

    def sort_images(self, instance):
        ok = sort_whatsapp_images()
        if ok:
            popup = Popup(title="Done",
                          content=Label(text="✔ Sorting Completed! Images moved in source folder."),
                          size_hint=(0.7, 0.4))
            popup.open()

class ImageSortApp(App):
    def build(self):
        return MainScreen()

if __name__ == "__main__":
    ImageSortApp().run()
