# AI 100 Midterm Project Report: MNIST Digit Classification

**Name:** <YOUR NAME HERE>  
**Date:** <DATE>

## 1) Problem definition and dataset curation
**Goal.** Classify a 28x28 grayscale image of a handwritten digit into one of 10 classes (0–9). This is a standard multi-class image classification problem.

**Dataset.** MNIST consists of 60,000 training images and 10,000 test images of handwritten digits. We use the official MNIST dataset provided by `torchvision.datasets.MNIST` (downloaded automatically on first run).  
We split the original training set into **55,000 training** and **5,000 validation** samples using a fixed random seed for reproducibility.

**Preprocessing.**
- Convert image to tensor in [0,1].
- Normalize using MNIST mean/std (commonly used values).

## 2) Deep learning models
We train a small CNN because convolutional layers are a good fit for images: they exploit local spatial patterns and share weights across the image.

**Model: SimpleCNN**
- Conv(1→32, 3x3) + ReLU
- Conv(32→64, 3x3) + ReLU + MaxPool(2x2)
- MaxPool(2x2)
- Flatten
- FC(64*7*7→128) + ReLU + Dropout
- FC(128→10 logits)

**Loss:** Cross-Entropy Loss  
**Optimizer:** Adam  
**Hyperparameters:** epochs=5, batch_size=128, lr=0.001, dropout=0.25

## 3) Results and presentation
We report:
- **Validation accuracy per epoch**
- **Final test accuracy**
- Confusion matrix
- Training curves (train/val loss)
- Sample predictions

After running `python train.py`, all artifacts are saved under `runs/<timestamp>/`.

**Expected performance (typical).** A simple CNN on MNIST commonly achieves around ~99% accuracy when trained for a few epochs with standard settings (see references).

## 4) Lessons learned
- **CNNs beat MLPs on images** because they use local receptive fields and parameter sharing.
- **Normalization and train/val split** matters: it stabilizes training and lets you pick the best model.
- **Overfitting is easy on small datasets**; dropout helps, but MNIST is still “easy” compared to real-world images.
- **Reproducibility**: setting seeds and saving metrics/plots makes your work easier to grade and explain.

## References
1. Yann LeCun et al., “Gradient-Based Learning Applied to Document Recognition,” 1998.  
2. Google Developers Codelab: “TensorFlow, Keras and deep learning, without a PhD” (MNIST, 99% accuracy).  
3. PyTorch Documentation (tutorials and torchvision datasets).

