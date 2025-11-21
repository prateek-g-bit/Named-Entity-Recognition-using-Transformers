# 📘 Named Entity Recognition using Transformers (BERT Fine-Tuning)

This repository contains a complete implementation of **Named Entity Recognition (NER)** using a fine-tuned **BERT-base-cased** model.  
The model was trained on a custom dataset and can identify entities such as **Person**, **Location**, and **Organization**.

---

## 📦 Download Trained Model

The final trained model checkpoint is stored on Google Drive:

🔗 https://drive.google.com/drive/folders/1zZBjzqEHuCB3vDhY3G5jxLtGU4RjRK5l?usp=sharing

Inside the Drive folder:

ner_model/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
    training_args.bin
    trainer_state.json

Download the folder and load it directly with HuggingFace Transformers.

---

## 🚀 Project Overview

This project fine-tunes the **BERT-base-cased** model for token classification (NER).  
It includes:

- Dataset preprocessing  
- Tokenization and word-label alignment  
- Training using HuggingFace Trainer  
- Final model checkpoint provided through Google Drive  
- Inference function for predicting NER tags  

---

## 🧠 Model Architecture

- Base Model: **BERT-base-cased**  
- Task: **Token Classification (NER)**  
- Framework: **PyTorch**  
- Library: **HuggingFace Transformers**  
- Optimizer: **AdamW**  
- Loss Function: **CrossEntropyLoss**  

---

## 🧪 Inference Example

### Load the trained model:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "PATH_TO_DOWNLOADED_ner_model_FOLDER"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
## Define the prediction function:
import torch

def ner_predict(text):
    words = text.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = logits.argmax(-1)[0].tolist()
    id2label = model.config.id2label

    results = []
    for token, pred, word_id in zip(inputs.tokens(), pred_ids, inputs.word_ids()):
        if word_id is not None:
            results.append((words[word_id], id2label[pred]))
    return results
```
### Example usage:
ner_predict("Barack Obama visited India yesterday")
## 📘 Requirements
transformers
datasets
seqeval
torch
numpy
tqdm
## 🖥️ Hardware Used
Windows 10
NVIDIA GeForce RTX 3050 (4GB VRAM)
PyTorch
HuggingFace Transformers
## ⭐ Features
A complete NER training pipeline
Custom dataset preprocessing
Final model checkpoint provided via Drive
Easy-to-use inference function
Works in Jupyter, VSCode, and Anaconda
## **🤝 Contributing**
Contributions and suggestions are welcome.
## 📄 License

---

If you'd like, I can also generate:

✔ A project banner image  
✔ A HuggingFace model card  
✔ A Streamlit web app  
✔ A Google Colab demo notebook

Just tell me!
