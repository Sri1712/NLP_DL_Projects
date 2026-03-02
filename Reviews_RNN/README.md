# 🎬 Sentiment Analysis Using Simple RNN (IMDB Reviews)

This project implements a **Simple Recurrent Neural Network (RNN)** for **sentiment analysis** using the IMDB movie reviews dataset available in `tensorflow.keras.datasets`.

The model classifies movie reviews as **positive** or **negative** based on textual content.

---

## 📖 Project Overview

- **Problem Type:** Binary Sentiment Classification  
- **Domain:** Natural Language Processing (NLP)  
- **Model:** Simple RNN  
- **Framework:** TensorFlow / Keras  

This project is intended as a beginner-friendly introduction to using recurrent neural networks for text-based machine learning tasks.

---

## 📂 Dataset

The IMDB Movie Reviews dataset contains:

- 50,000 movie reviews  
- 25,000 training samples  
- 25,000 testing samples  
- Labels:
  - `1` → Positive sentiment  
  - `0` → Negative sentiment  

The dataset is automatically loaded using:

```python
from tensorflow.keras.datasets import imdb
```

## 🧠 Model Architecture

The neural network consists of:

- **Embedding Layer** – Converts words into dense vector representations  
- **Simple RNN Layer** – Learns sequential dependencies in text  
- **Dense Output Layer (Sigmoid)** – Produces binary classification output  

### Architecture Flow
```
Embedding → SimpleRNN → Dense (Sigmoid)
```

---

## 🛠 Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib 

---

## 🔧 Training Configuration

Typical configuration used:

- **Vocabulary Size:** 10,000  
- **Maximum Review Length:** 200  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metric:** Accuracy  

---

## 📊 Results

The Simple RNN model achieves reasonable accuracy on the test dataset, demonstrating how recurrent neural networks can process and learn from sequential text data.

> **Note:** Simple RNNs are primarily used for educational purposes. Models such as **LSTM** and **GRU** generally provide better performance on NLP tasks.