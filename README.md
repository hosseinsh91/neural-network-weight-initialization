# CIFAR-10 Deep Learning Model: Weight Initialization Analysis

## **📌 Project Overview**
This project trains a **fully connected neural network (MLP)** on the **CIFAR-10 dataset** to classify images into 10 categories. The notebook investigates the impact of **weight initialization** on learning behavior and explores why initializing weights to **zero** results in a lack of weight updates.

### **🚀 Key Features**
✅ Uses the **CIFAR-10 dataset** for image classification  
✅ Implements a **Multi-Layer Perceptron (MLP) model** in TensorFlow/Keras  
✅ Demonstrates the effect of **zero weight initialization**  
✅ Uses the **SGD optimizer** for training  
✅ Analyzes **weight updates and training stagnation**  

---

## **📌 Dataset: CIFAR-10**
- The **CIFAR-10 dataset** consists of **60,000 color images** in **10 categories**, including:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Each image is **32x32 pixels with 3 color channels**.

### **📌 Data Preprocessing**
- The dataset is **normalized** by scaling pixel values to the `[0,1]` range:
  ```python
  (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
  X_train, X_test = X_train / 255.0, X_test / 255.0
