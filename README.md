# MNIST Digit Classification (AI 100 Midterm Project)

This repo trains a simple Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset (10 classes: 0-9).

## Quick start

### 1) Create environment + install deps
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Train
```bash
python train.py --epochs 5 --batch-size 128 --lr 0.001
```

### 3) Evaluate + artifacts
The script saves:
- `runs/<timestamp>/metrics.json`
- `runs/<timestamp>/confusion_matrix.png`
- `runs/<timestamp>/training_curves.png`
- `runs/<timestamp>/sample_predictions.png`
- `runs/<timestamp>/best_model.pt`

## Notes
- First run downloads MNIST automatically via torchvision.
- Works on CPU. If you have a GPU + CUDA, PyTorch will use it automatically.

