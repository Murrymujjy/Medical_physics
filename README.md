# Low-Parameter Grayscale CNN for Localized Breast Lesion Classification

An AI-powered diagnostic framework engineered for the binary classification of benign and malignant lesions in digital mammography. By focusing on localized mass boundaries rather than downsampling full-field images, this architecture decouples background tissue noise from training data memorization.

## 🚀 Live Deployment
The model is deployed via Streamlit Community Cloud. You can interact with the system live here: 
👉 To be uploaded soon

---

## 🛠️ Tech Stack & Implementation Specs
* **Core Framework:** TensorFlow / Keras (Sequential API)
* **Deployment Interface:** Streamlit Cloud
* **Input Constraints:** Single-channel grayscale matrices (256 × 256 × 1)
* **Optimization Pipeline:** Adam Optimizer ($LR = 0.0003$), Batch Size = 32, Binary Cross-Entropy Loss

---

## 📐 Architecture Blueprint
To accommodate strict workstation hardware limitations, the network uses a specialized medium-capacity design restricted to **249,281 trainable parameters**:

1. **Input Layer:** (256, 256, 1) with real-time spatial data augmentations (Random Flips & Rotations).
2. **Feature Extraction Blocks:** Four sequential Convolutional layers configured with progressive filter depths (32, 64, 128, and 128 filters), using ReLU activations and Batch Normalization.
3. **Dimensional Collapse:** Global Average Pooling (GAP) to prevent a dense parameter explosion.
4. **Classification Head:** A 64-unit Dense layer with Dropout (0.4) paired with a final single-unit Sigmoid activation layer for binary inference.

---

## 📊 Performance Summary
Due to the intense structural and textural overlap present in isolated parenchymal tissue matrices, the model enforces tight structural constraints to achieve uniform convergence without overfitting:

* **Training AUC:** 0.6201
* **Validation AUC:** 0.5716
* **Stable Validation Accuracy:** 53.55%

---

## 💻 Local Installation & Setup

If you wish to run this repository locally on your workstation:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
