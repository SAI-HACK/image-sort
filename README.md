<!-- PROJECT LOGO -->
<p align="center">
  <img src="./assets/banner.png" alt="IMAGE SORT Banner" width="80%">
</p>

<h1 align="center">🖼️ IMAGE SORT</h1>
<p align="center">
  <b>Smart Image Categorization using Deep Learning and Computer Vision</b>  
</p>

<p align="center">
  
</p>

---

## 🧠 About the Project

**IMAGE SORT** is an intelligent **AI-powered image organization system** that automatically sorts images into categorized folders using a **pre-trained deep learning model (ShuffleNetV2)** built with **PyTorch**.

Designed to handle large image collections, this project helps you:
- Save time 🕒  
- Eliminate manual sorting 🖐️  
- Keep your image libraries neat and organized 🗂️  
- Open-source and free
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
## ✅  Datasets used from [https://huggingface.co/] 
(Public, free and working as of 28-10-2025)

| Category          | Hugging Face Dataset                       |
| ----------------- | ------------------------------------------ |
| 🧍 People         | `schirrmacher/humans`                      |
| 😂 Memes          | `poloclub/diffusiondb`                     |
| 📄 Documents      | `nielsr/funsd`                             |
| ✍️ Handwritten    | `agomberto/FrenchCensus-handwritten-texts` |
| 📢 Advertisements | `multimodalart/ads-dataset`                |
| 💻 Digital Notes  | `HuggingFaceM4/DocumentVQA`                |



## 🧩 Folder Structure

IMAGE SORT/

│

├── imagesort_shufflenetv2.pt 

├── main.py 

├── requirements.txt 

└── README.md 

(Make sure all the files are present in the same folder)


---

## ⚙️ Installation

> 🧠 *It’s recommended to use a virtual environment for isolation.*

```python
# Clone the repo
git clone https://github.com/SAI-HACK/image-sort.git
cd image-sort 
```
# Create and activate virtual environment
```python
python3 -m venv venv
source venv/bin/activate
```
# Install dependencies
```python
pip install -r requirements.txt
```

# Run the application
```python
python3 main.py
```
