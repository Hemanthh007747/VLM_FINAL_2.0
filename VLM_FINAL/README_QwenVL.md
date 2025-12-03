# 🧠 Qwen_VL Streamlit Colab App — Project Documentation

## 📘 Overview
This project demonstrates a **visual-language AI system** built using **Qwen2-VL-2B-Instruct**, an open-source multimodal model by Alibaba Cloud.  
It can understand **both images and text**, making it capable of **Visual Question Answering (VQA)**, **caption generation**, and **image-based reasoning**.

The system is deployed via **Streamlit** and hosted publicly through **Cloudflare Tunnel** inside **Google Colab**, enabling anyone to access the model through a web interface.

## 🏗️ System Architecture
```
┌────────────────────────┐
│ Google Colab Notebook  │
│ (Python Runtime)       │
└──────────┬─────────────┘
           │
     Streamlit Web App
           │
           ▼
┌────────────────────────┐
│ Qwen2-VL-2B-Instruct   │
│ (HuggingFace Transformers) │
└──────────┬─────────────┘
           │
   Cloudflare Tunnel (Public URL)
           │
           ▼
    🌐 Accessible Web UI
```

## 🧩 Key Components

| Component | Description |
|------------|--------------|
| **Model** | `Qwen2-VL-2B-Instruct` — a multimodal model supporting vision-language tasks. |
| **Framework** | Streamlit — builds the interactive frontend interface. |
| **Deployment** | Cloudflared tunnel — creates a secure, sharable public link from Colab. |
| **Backend Libraries** | `transformers`, `torch`, `accelerate`, `pillow`. |
| **Environment** | Google Colab (Python 3.10+, GPU runtime recommended). |

## 🚀 Features
✅ **Visual Question Answering (VQA)**  
✅ **OCR + QA Mode**  
✅ **Multimodal Reasoning**  
✅ **Web Interface via Cloudflare Tunnel**  
✅ **Colab Deployable**  

## ⚙️ Setup Instructions
```bash
!pip install -q streamlit cloudflared transformers accelerate pillow torch
!nohup streamlit run app.py --server.port 8501 > /dev/null 2>&1 &
import time; time.sleep(5)
!cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

## 💻 Streamlit App Structure
```python
import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

st.title("🧠 Qwen-VL Image Understanding App")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
question = st.text_input("Ask a question about the image:")

if uploaded_image and question:
    image = Image.open(uploaded_image)
    inputs = processor(text=question, images=image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    answer = processor.decode(output[0], skip_special_tokens=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"**Answer:** {answer}")
```

## 🧰 Troubleshooting

| Problem | Cause | Fix |
|----------|--------|-----|
| **DNS_PROBE_FINISHED_NXDOMAIN** | Cloudflare tunnel failed | Use `--no-autoupdate` and ensure Streamlit is running first. |
| **Black screen / no output** | Streamlit crashed | Run `!streamlit run app.py` to check logs. |
| **GPU not used** | Colab CPU runtime | Switch to GPU under *Runtime → Change runtime type → GPU*. |

## 🧾 Credits
- **Model:** [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)  
- **Libraries:** Hugging Face Transformers, PyTorch, Streamlit, Pillow  
- **Author:** Hemanth V  
- **Platform:** Google Colab (with Cloudflare Tunnel)

## 🧭 Future Improvements
- Add **speech input/output** using `gTTS` or `pyttsx3`.  
- Implement **multi-image comparison**.  
- Enhance **handwritten text recognition** using hybrid OCR models.  
- Deploy permanently on **Streamlit Cloud / Hugging Face Spaces**.
