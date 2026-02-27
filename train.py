import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import SimpleCNN
from utils import (
    accuracy_from_logits,
    confusion_matrix,
    plot_confusion_matrix,
    plot_curves,
    plot_sample_predictions,
    save_json,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    # 55k train / 5k val
    train_size = 55_000
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_acc += accuracy_from_logits(logits, y) * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # Test using best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_preds, all_true = [], []
    sample_batch = None
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.numpy())
            if sample_batch is None:
                sample_batch = (x.cpu(), y, torch.from_numpy(preds))

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    test_acc = float((all_preds == all_true).mean())

    cm = confusion_matrix(all_preds, all_true, num_classes=10)
    plot_confusion_matrix(cm, run_dir / "confusion_matrix.png")
    plot_curves(history, run_dir / "training_curves.png")
    if sample_batch is not None:
        x_s, y_s, p_s = sample_batch
        plot_sample_predictions(x_s, y_s, p_s, run_dir / "sample_predictions.png", n=16)

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_val_acc": float(best_val_acc),
        "test_acc": test_acc,
    }
    save_json(metrics, run_dir / "metrics.json")
    print("Saved artifacts to:", run_dir)
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    main()
