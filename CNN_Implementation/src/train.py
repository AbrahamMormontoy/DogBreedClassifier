import os
import csv
import argparse
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model   import build_model


# Default hyperparameters 

DEFAULT_EPOCHS = 15
DEFAULT_LR     = 0.001
CHECKPOINT     = "best_model.pth"
LOG_FILE       = os.path.join("logs", "training_log.csv")
TRAINING_CURVE = os.path.join("plots", "training_curves.png")


# Training & validation helpers 

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()         

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1) 
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return total_loss / total, correct / total


def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    model.eval()

    total_loss, top1_correct, top3_correct, total = 0.0, 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, top1_pred = outputs.max(1)
            top1_correct += top1_pred.eq(labels).sum().item()

            _, top3_pred = outputs.topk(3, dim=1)
            top3_correct += sum(
                labels[i].item() in top3_pred[i].tolist()
                for i in range(labels.size(0))
            )

            total += labels.size(0)

    return total_loss / total, top1_correct / total, top3_correct / total


# Main training loop 

def train(epochs: int, lr: float) -> None:
    os.makedirs("logs", exist_ok=True)


    # Load data 
    print("=" * 60)
    print("STEP 1 — Loading dataset")
    print("=" * 60)
    train_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    # Build model
    print()
    print("=" * 60)
    print("STEP 2 — Building model")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = build_model(
        num_classes=num_classes,
        learning_rate=lr,
        device=device,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )

    # Training loop 
    print()
    print("=" * 60)
    print(f"STEP 3 — Training for {epochs} epoch(s)")
    print("=" * 60)

    best_top1 = 0.0
    log_rows   = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, top1_acc, top3_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        print(
            f"  Epoch {epoch:>2}/{epochs}  |  "
            f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.2%}  |  "
            f"Test loss: {test_loss:.4f}  Top-1: {top1_acc:.2%}  Top-3: {top3_acc:.2%}"
        )

        # Checkpoint
        if top1_acc > best_top1:
            best_top1 = top1_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"<Saved:> {CHECKPOINT}")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "test_loss":  round(test_loss,  4),
            "top1_acc":   round(top1_acc,   4),
            "top3_acc":   round(top3_acc,   4),
        })

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Final summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Top-1 accuracy : {best_top1:.2%}")
    print(f"  Baseline (M1)       : 57.80%  (logistic regression)")
    print(f"  Improvement         : {(best_top1 - 0.578) * 100:+.1f} percentage points")
    print(f"  Checkpoint saved to : {CHECKPOINT}")
    print(f"  Training log saved  : {LOG_FILE}")

def plot_training_curves(log_csv: str, out_path: str) -> None:
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["top1_acc"], label="Test Top-1 Accuracy")
    plt.plot(df["epoch"], df["top3_acc"], label="Test Top-3 Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Curves")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"<Saved:> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the ResNet-18 dog breed classifier."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate for Adam optimizer (default: {DEFAULT_LR})"
    )
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr)
    os.makedirs("plots", exist_ok=True)
    plot_training_curves(LOG_FILE, TRAINING_CURVE)
