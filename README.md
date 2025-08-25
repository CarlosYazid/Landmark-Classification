# Landmark-Classification

This project demonstrates how to train and evaluate Convolutional Neural Networks (CNNs) for image classification using **PyTorch**.  
It includes three main workflows: training a CNN from scratch, applying transfer learning, and deploying a simple prediction app.

---

## 📂 Project Structure

```
.
├── cnn_from_scratch.ipynb   # Training a custom CNN from scratch
├── transfer_learning.ipynb  # Transfer learning with pretrained ResNet
├── app.ipynb                # Simple inference and demo app
├── src/
│   ├── helpers.py           # Project utilities
│   ├── model.py             # Custom CNN architecture
│   ├── transfer.py          # Transfer learning model builder
│   ├── train.py             # Training, validation, testing loops
│   ├── optimization.py      # Loss functions and optimizers
│   ├── predictor.py         # Inference helper
│   └── data.py              # Data loading and preprocessing
```

---

## 🚀 Workflows

### 1. CNN From Scratch
- Implements a custom CNN defined in `model.py`.
- Architecture: **5 convolutional blocks** (Conv2D → BatchNorm → ReLU → MaxPool), followed by fully connected layers with dropout.
- Trained with **CrossEntropyLoss** and optimizers (SGD/Adam).
- Pros: full control of the architecture.  
- Cons: requires more training data and computation.

### 2. Transfer Learning
- Uses a **ResNet18 pretrained on ImageNet** (`torchvision.models`).
- Freezes all backbone layers (`requires_grad=False`).
- Replaces the final fully connected layer with a custom linear classifier for our dataset (50 classes by default).
- Much faster training and better performance on small datasets.

### 3. Inference App
- Loads a trained model checkpoint.
- Uses `predictor.py` to preprocess input images and predict class labels.
- Demonstrates how to move from training to deployment.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CarlosYazid/Landmark-Classification.git
   cd cnn-image-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies:
   - PyTorch
   - Torchvision
   - Matplotlib
   - tqdm
   - livelossplot

---

## 🛠 Usage

### Training from scratch
```bash
jupyter notebook cnn_from_scratch.ipynb
```

### Transfer learning
```bash
jupyter notebook transfer_learning.ipynb
```

### Run inference app
```bash
jupyter notebook app.ipynb
```

---

## 📊 Results
- **From Scratch**: flexible architecture, but requires more epochs to converge.
- **Transfer Learning**: achieves higher accuracy with fewer epochs by leveraging pretrained features.
- **App**: demonstrates how to perform predictions on unseen images.

---

## 🔬 Key Files

- `src/model.py` → Custom CNN (`MyModel`)  
- `src/transfer.py` → Transfer learning setup with pretrained ResNet  
- `src/train.py` → Training/validation loops, early stopping, learning rate scheduling  
- `src/optimization.py` → Loss and optimizer selection  
- `src/predictor.py` → Preprocessing and prediction logic  

---

## 📌 Notes
- Images are resized to **224×224** pixels for consistency.  
- Data augmentation (flips, crops, rotations) improves generalization when training from scratch.  
- Validation loss is tracked, and the model is checkpointed when improvement is detected.  
- Learning rate scheduling (`ReduceLROnPlateau`) is used to stabilize training.

---

## 📖 License
This project is for **educational purposes**. You are free to use and adapt it under the MIT license.
