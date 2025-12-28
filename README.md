# 🧠 CIFAR-10 Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN) from scratch** using PyTorch to perform image classification on the CIFAR-10 dataset.  
The goal is to **understand deep learning fundamentals, CNN training behavior, and model evaluation**, rather than chasing benchmark scores.  

This is my first deep learning project, built with a strong focus on **learning, correctness, and explainability**.

---

## 📌 Dataset: CIFAR-10

- **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- **Images:** 60,000 color images (32×32)  
  - 50,000 training images  
  - 10,000 test images  

**Challenges:**
- Low image resolution  
- High inter-class similarity  

---

## 🏗️ Model Architecture

The CNN is designed to progressively learn spatial features from images:

- **Convolutional layers** for feature extraction  
- **ReLU activations** for non-linearity  
- **MaxPooling layers** for downsampling  
- **Fully connected layers** for classification  

**Note:** The model was trained **from scratch** without:

- Pretrained weights  
- Transfer learning  
- Advanced architectures (e.g., ResNet)  

This helps in understanding **how CNNs actually learn**.

---

## ⚙️ Training Details

- **Framework:** PyTorch  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Stochastic Gradient Descent (SGD)  
- **Epochs:** 5  
- **Batch Training:** Mini-batch gradient descent  
- **Hardware:** CPU / local environment  

**Observations:**  
- Training loss showed a consistent downward trend, indicating **proper learning** and **stable optimization**.  

---

## 📊 Results

- **Test Accuracy:** 61%  
- Evaluated on **10,000 unseen test images**  

**Performance Highlights:**  
- Solid for a CNN trained from scratch  
- No data augmentation or transfer learning  

### 🖼️ Sample Predictions

| Ground Truth | cat  | ship | ship | plane |
|--------------|------|------|------|-------|
| Predicted    | ship | ship | car  | plane |

This demonstrates that:

- The model learns meaningful visual patterns  
- Misclassifications are reasonable and expected for CIFAR-10  

---

## 📈 Why 61% Accuracy?

CIFAR-10 is a **challenging dataset** due to low resolution and diverse object categories.  
The main objectives of this project were to:

- Understand CNN training dynamics  
- Analyze loss behavior  
- Evaluate generalization on test data  
- Build a complete deep learning pipeline  

**Ways to improve accuracy:**  

- Data augmentation  
- Learning rate scheduling  
- Deeper architectures (ResNet, VGG)  

---

## 🚀 How to Run the Project

**Clone the repository:**
```bash
git clone <repo-link>
cd CIFAR10-CNN
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Trained the model
```bash
python cifar10_training.py
```

## Test the Model

The training script also evaluates the model on the test set. You can optionally split it into a separate **test.py** if desired.
