# 🧠 CNN From Scratch in NumPy

A minimal Convolutional Neural Network built using only NumPy — no deep learning libraries used.  
Tested on the MNIST dataset.

## 🚀 Final Accuracy
**81.2%** test accuracy using 1000 training and 1000 test samples.

## 🛠️ Features
- Manual convolution, ReLU, max pooling, flattening
- Dense layer with softmax output
- Cross-entropy loss and full backpropagation
- Gradient descent optimizer

## 📊 Limitations
- No batching or GPU acceleration
- Slow due to Python loops in conv/pool ops
- Trained on a small subset for performance

## 🖼️ Sample Output
![sample](/sample_predictions.png)

## 📁 Structure
- `cnn_from_scratch.ipynb` — Full implementation and training
- `maths_notes.md` — Maths behind CNNs
- `Images` — Images

## 🧠 Learnings
This project offers deep intuition on how CNNs work under the hood, a great foundation before using PyTorch/TensorFlow.

---
## 📘 Maths behind CNNs
See [maths_notes.md](maths_overview.md) for the maths behind forward and backward passes.

---

## 📌 Run this notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iam-vsr/cnn_from_scratch/blob/main/cnn_from_scratch.ipynb)
