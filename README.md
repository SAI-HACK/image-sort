Here is a complete, copy-and-paste-ready README.md with the project renamed everywhere to Image Sort and structured according to common Python/ML README conventions for easy onboarding and collaboration.[3][4]

### README.md
```markdown
# Image Sort

AI-powered desktop/Android app that automatically sorts WhatsApp (or general) images into categories using a lightweight CNN (ShuffleNetV2) trained on local and public datasets, with an offline Kivy UI for batch sorting and review.

## Overview

Image Sort classifies images into six categories—Advertisements, Digital Notes, Documents, Handwritten Notes, Memes, People—and routes them into folders, enabling fast cleanup of WhatsApp media and photo libraries on Ubuntu desktop or Android (via Buildozer). The model is trained by combining a local dataset with streaming public datasets, then saved as a .pth file for offline inference inside the Kivy app.

## Features

- Lightweight CNN backbone (ShuffleNetV2) optimized for speed on CPU/mobile.  
- Mixed training from local folders and public datasets, with augmentation and label mapping saved as JSON.  
- Offline inference in a simple Kivy UI with a single “Sort Images” action and color-themed layout.  
- Deterministic sorting from a source directory to per-class destination folders on disk.  

## Project Structure

```
ImageSort/
│
├── main.py                   # Kivy app: loads CNN, sorts images, displays UI
├── train_huggingface_cnn.py  # Train by combining local + public datasets
├── imagesort_labels.json     # Saved label mapping
├── imagesort_hf_cnn.pth      # Trained model weights
│
├── dataset/
│   ├── train/
│   │   ├── advertisements/
│   │   ├── documents/
│   │   ├── handwritten/
│   │   ├── memes/
│   │   ├── people/
│   │   └── digitalnotes/
│   └── val/
│       ├── advertisements/
│       ├── documents/
│       ├── handwritten/
│       ├── memes/
│       ├── people/
│       └── digitalnotes/
│
├── huggingface_datasets.txt   # One dataset source per line
└── /mnt/d/huggingfacecnn/     # Output model + labels
```

## Categories

- advertisements  
- digitalnotes  
- documents  
- handwritten  
- memes  
- people  

## Installation

System packages (Ubuntu/WSL):

```
sudo apt update && sudo apt install -y \
python3 python3-pip python3-venv git wget curl build-essential \
libgl1 libglib2.0-0 ffmpeg libsdl2-dev libsdl2-image-dev \
libsdl2-mixer-dev libsdl2-ttf-dev libmtdev-dev
```

Python environment:

```
python3 -m venv ~/imagesort_env
source ~/imagesort_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio datasets transformers huggingface_hub \
pillow tqdm requests numpy matplotlib scikit-learn opencv-python kivy
```

Optional Android build prerequisites:

```
sudo apt install -y openjdk-17-jdk unzip zlib1g-dev libffi-dev libssl-dev \
autoconf automake libtool pkg-config
pip install buildozer cython
```

## Dataset

Local dataset should follow the structure in `dataset/train` and `dataset/val`, with subfolders per class as shown above. Public dataset sources should be listed one per line in `huggingface_datasets.txt`, and the training script will stream them during training when authenticated.

Login once to access public datasets:

```
huggingface-cli login
# Paste token when prompted; cached under ~/.cache/huggingface/token
```

## Training

The training script fine-tunes ShuffleNetV2 on mixed local+public data with augmentation, Adam optimizer, CrossEntropyLoss, and ReduceLROnPlateau scheduling.

Run training:

```
source ~/imagesort_env/bin/activate
python train_huggingface_cnn.py
```

Outputs are written to:

```
/mnt/d/huggingfacecnn/
 ├── imagesort_hf_cnn.pth
 └── imagesort_labels.json
```

## Inference (Desktop)

The Kivy app loads the `.pth` and `.json` and sorts images from a source folder to per-class folders under a destination base.

Default paths:

```
Source:      ~/adp_data
Destination: ~/adp_sorted
```

Run the app:

```
source ~/imagesort_env/bin/activate
python main.py
```

Usage:

- Launch app and click “Sort Images” to begin scanning `~/adp_data`.  
- Images are routed into `~/adp_sorted/<category>` based on model predictions.  
- Consider adding an “unsorted_review” threshold in code to triage low-confidence predictions.

## Android (APK)

Build notes:

- Keep the model lightweight for CPU-only devices; consider a smaller ShuffleNetV2 variant or quantized weights.  
- Ensure required Python dependencies are included in `buildozer.spec` under `requirements` (e.g., kivy, pillow, numpy, opencv-python, torch, torchvision as feasible for target).  
- Typical steps: `buildozer init`, edit `buildozer.spec`, then `buildozer -v android debug` to build a debug APK.

## Configuration

- Model/labels: set paths to `imagesort_hf_cnn.pth` and `imagesort_labels.json` in `main.py`.  
- Folders: adjust `src_dir` and `dst_base` in `main.py` to match your environment.  
- Classes: update the class list consistently across training and inference if categories change.

## Troubleshooting

- Missing GPU is fine; the project supports CPU, but training will be slower.  
- If public dataset streaming fails, re-run `huggingface-cli login` or remove invalid entries in `huggingface_datasets.txt`.  
- For UI freezes during large batches, enable background-thread inference and a progress bar in the Kivy app.

## Roadmap

- OCR for handwritten/digital note detection.  
- Face recognition to enhance “people” classification.  
- Progress UI with KivyMD and non-blocking inference.  
- Quantization or ONNX/TFLite export for smaller APKs and faster CPU inference.  
- Held-out test set with confusion matrix for per-class quality tracking.

## Acknowledgments

- ShuffleNetV2 architecture for efficient mobile inference.  
- Public datasets and tooling from the Python ML ecosystem and Kivy for cross-platform UI.

## License

Specify a license (e.g., MIT) in a `LICENSE` file at the repository root.
```

[1](https://github.com/onesamblack/machine-learning-template/blob/main/README.md)
[2](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md)
[3](https://www.makeareadme.com)
[4](https://realpython.com/readme-python-project/)
[5](https://www.drupal.org/docs/develop/managing-a-drupalorg-theme-module-or-distribution-project/documenting-your-project/readmemd-template)
[6](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
[7](https://bea.stollnitz.com/blog/vscode-ml-project/)
[8](https://deepsense.ai/blog/standard-template-for-machine-learning-projects-deepsense-ais-approach/)
[9](https://www.reddit.com/r/opensource/comments/txl9zq/next_level_readme/)
[10](https://git.wur.nl/bioinformatics/fte40306-advanced-machine-learning-project-data/-/blob/main/README.md)
