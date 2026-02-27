import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def confusion_matrix(preds: np.ndarray, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, y):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm: np.ndarray, outpath: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (MNIST)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_curves(history: Dict[str, list], outpath: Path) -> None:
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_sample_predictions(images: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, outpath: Path, n: int = 16) -> None:
    images = images[:n].cpu().numpy()
    y_true = y_true[:n].cpu().numpy()
    y_pred = y_pred[:n].cpu().numpy()

    rows = int(np.ceil(n / 4))
    fig = plt.figure(figsize=(8, 2 * rows))
    for i in range(n):
        ax = fig.add_subplot(rows, 4, i + 1)
        ax.imshow(images[i][0], cmap="gray")
        ax.set_title(f"t={y_true[i]} p={y_pred[i]}")
        ax.axis("off")
    fig.suptitle("Sample Predictions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_json(obj: Dict, outpath: Path) -> None:
    outpath.write_text(json.dumps(obj, indent=2), encoding="utf-8")
