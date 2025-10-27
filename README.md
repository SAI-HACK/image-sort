<!-- PROJECT LOGO -->
<p align="center">
  <img src="./assets/banner.png" alt="IMAGE SORT Banner" width="80%">
</p>

<h1 align="center">🖼️ IMAGE SORT</h1>
<p align="center">
  <b>Smart Image Categorization using Deep Learning and Computer Vision</b>  
</p>

<p align="center">
  <a href="https://github.com/SAI-HACK/IMAGE-SORT/stargazers"><img src="https://img.shields.io/github/stars/<your-username>/IMAGE-SORT?color=yellow&style=for-the-badge"></a>
  <a href="https://github.com/SAI-HACK/IMAGE-SORT/issues"><img src="https://img.shields.io/github/issues/<your-username>/IMAGE-SORT?style=for-the-badge"></a>
  <a href="https://github.com/<your-username>/IMAGE-SORT"><img src="https://img.shields.io/github/license/<your-username>/IMAGE-SORT?color=blue&style=for-the-badge"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python"></a>
</p>

---

## 🧠 About the Project

**IMAGE SORT** is an intelligent **AI-powered image organization system** that automatically sorts images into categorized folders using a **pre-trained deep learning model (ShuffleNetV2)** built with **PyTorch**.

Designed to handle large image collections, this project helps you:
- Save time 🕒  
- Eliminate manual sorting 🖐️  
- Keep your image libraries neat and organized 🗂️  

---

## 🎯 Features

✅ **Automatic Classification** — Sorts images based on visual similarity.  
⚡ **High Performance** — Optimized with TorchScript for fast inference.  
🧩 **Lightweight** — Uses the efficient **ShuffleNetV2** backbone.  
🖥️ **Cross-Platform** — Works on Linux, WSL2, and Ubuntu systems.  
📁 **Organized Output** — Creates structured category folders automatically.  
🧾 **Transparent Logs** — Displays categorized results in real-time.  

---

## 🧰 Tech Stack

| Technology | Description |
|-------------|-------------|
| 🐍 **Python 3.8+** | Core programming language |
| 🔥 **PyTorch** | Deep learning framework |
| 🧠 **ShuffleNetV2 (TorchScript)** | Model architecture |
| 👁️ **OpenCV** | Image processing and manipulation |
| 📦 **NumPy** | Array and matrix operations |
| 🧮 **OS / sys** | File system interaction and automation |

---

## 🧩 Folder Structure

IMAGE SORT/

│

├── intellisort_shufflenetv2.pt # Pre-trained TorchScript model

├── docrm.py # Main Python script

├── input_images/ # Folder for unsorted images

├── sorted_images/ # Output folder (auto-generated)

├── requirements.txt # Project dependencies

└── README.md # This documentation



---

## ⚙️ Installation

> 🧠 *It’s recommended to use a virtual environment for isolation.*

```bash
# Clone the repo
git clone https://github.com/SAI-HACK/image-sort.git
cd image-sort 

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python3 main.py



Loading TorchScript model from imagesort_shufflenetv2.pt ...
✅ Processed: dog.jpg → Animals/
✅ Processed: car.png → Vehicles/
✅ Processed: flower.jpeg → Nature/
🎉 Sorting Completed Successfully!

