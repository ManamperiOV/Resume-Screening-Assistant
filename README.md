# 🧠 Resume Screening Assistant – Fine-tuned DistilBERT + Gradio PDF Viewer

An AI-powered tool that automates resume classification using a fine-tuned **DistilBERT** model. This assistant predicts the most suitable **job category** for a resume (e.g., Data Scientist, Java Developer, HR) by analyzing its content. It supports **PDF upload**, **plain text input**, and provides **confidence scores** in a sleek **Gradio-based UI** with **live PDF viewing**.

> 💡 Designed as a real-world NLP project for recruiters, HR tech startups, or AI portfolios.

---

## 🔍 Project Overview

Resume screening is time-consuming, repetitive, and error-prone. This project uses **Natural Language Processing (NLP)** and **Transformers** to automate the task by predicting the job role a resume fits best.

It provides:

- ✅ Support for **PDF file upload** (live preview)
- ✅ Option to **paste raw resume text**
- ✅ Clean UI with **job category prediction + confidence**
- ✅ Inference powered by **DistilBERT**, fine-tuned on a labeled resume dataset

This project mimics a real-world ML pipeline from **raw data → preprocessing → model training → deployment**.

---

## 📁 Dataset

The dataset used is from [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset), containing labeled resumes across 25 categories.

| Column    | Description                                |
|-----------|--------------------------------------------|
| Resume    | Raw resume text (unstructured)             |
| Category  | Label (job role, e.g., 'Data Science')     |

**Example Categories:**
`Data Science`, `Python Developer`, `HR`, `Web Designing`, `Java Developer`, etc.

---

## 🔧 Preprocessing Pipeline

The resumes were unstructured and noisy. A thorough preprocessing pipeline was implemented in a Jupyter notebook using Python and NLP libraries.

### 🔎 Steps:

1. **Text Cleaning**:
   - Removed emails, URLs, numbers, and special characters
   - Lowercased all text
   - Removed extra whitespaces

2. **Stopword Removal**:
   - Used `nltk.corpus.stopwords` to remove common English stopwords

3. **Lemmatization** *(optional but planned for future)*:
   - To reduce words to their base form (e.g., "running" → "run")

4. **Label Encoding**:
   - Converted categorical labels into numeric IDs using `sklearn.preprocessing.LabelEncoder`

5. **Train-Test Split**:
   - 80% Training / 20% Testing split using `train_test_split`

6. **Text Tokenization**:
   - Tokenized using `AutoTokenizer` from Hugging Face (`distilbert-base-uncased`)
   - Truncated to max token length of 512 (BERT's limit)

7. **Hugging Face Dataset Integration**:
   - Wrapped into `datasets.Dataset` format for training with `Trainer`

📁 Tools used:
- `pandas`
- `nltk`
- `sklearn`
- `transformers`
- `datasets`

---

## 🧠 Model Architecture: Transformers

This project uses **DistilBERT**, a lighter and faster version of BERT.

### 🤖 What are Transformers?

Transformers are deep learning models designed to understand sequential data (like language) using **self-attention**. Instead of reading one word at a time (like RNNs), Transformers look at the entire sentence at once.

**Key Features:**
- Attention mechanism to understand word importance
- Parallelizable — faster training
- Bidirectional context understanding (left + right)

### 🔍 Why DistilBERT?

- 40% smaller than BERT, but retains 95% of its performance
- Faster inference for real-time applications
- Ideal for deploying in lightweight environments

---

## ⚙️ Training Process

Training was done using Hugging Face’s high-level `Trainer` API.

### 🏋️ Model Details:

- **Base model:** `distilbert-base-uncased`
- **Head:** Classification layer with 25 output labels
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** AdamW
- **Evaluation Metrics:** Accuracy, Weighted F1

### 🧪 Training Steps:

1. Tokenized input resumes and converted to PyTorch tensors
2. Used `Trainer` for fine-tuning with:
   - Batch size = 8
   - Epochs = 3
   - Weight decay = 0.01
3. Evaluated model on test set
4. Saved model and tokenizer for inference

---

## 🖥️ Application Interface (Gradio)

The model is deployed with an interactive Gradio interface using `Blocks` and the new `gradio-pdf` component.

### ✨ Features:

| Feature                    | Description                                      |
|----------------------------|--------------------------------------------------|
| 📄 PDF Upload              | Upload and preview resume inside browser         |
| ✍️ Paste Text              | Alternative input via textbox                    |
| 🎯 Prediction Output       | Displays predicted job category + confidence     |
| 🧹 Clear Button            | Resets input and output fields                   |

### 🖼️ UI Screenshot *(optional)*

![Screenshot 2025-06-24 132121](https://github.com/user-attachments/assets/46745f00-4463-408d-801b-355ebf71ea0f)
![Screenshot 2025-06-24 125929](https://github.com/user-attachments/assets/29d31e28-78ac-4faa-ab95-227aeef625a5)
![Screenshot 2025-06-24 130321](https://github.com/user-attachments/assets/6d03f13b-d34d-48e0-9780-2c5fb3a6ce19)

---

## 📦 How to Run the App

python resume_app.py
