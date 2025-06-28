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
![sample](/sample_prediction.png)

## 📁 Structure
- `cnn_numpy.ipynb` — Full implementation and training
- `cnn_numpy.py` — (Optional) Script version
- `assets/` — Sample predictions

## 🧠 Learnings
This project offers deep intuition on how CNNs work under the hood, a great foundation before using PyTorch/TensorFlow.

---

## 📌 Run this notebook in [Google Colab](https://colab.research.google.com)
