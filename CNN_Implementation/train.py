"""
train.py
--------
Main training script for the 50-breed dog classifier.

Ties together dataset.py (data pipeline) and model.py (ResNet-18 setup),
runs the training loop, and saves the best model checkpoint.

Usage:
    python train.py                      # Train with default settings
    python train.py --epochs 20          # Train for 20 epochs
    python train.py --epochs 5 --lr 0.0001   # Custom epochs and learning rate

Output:
    best_model.pth   — saved model weights (best validation accuracy)
    training_log.csv — loss and accuracy for every epoch (use for report graphs)
"""

import os
import csv
import argparse
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model   import build_model


# ── Default hyperparameters ───────────────────────────────────────────────────

DEFAULT_EPOCHS = 15
DEFAULT_LR     = 0.001
CHECKPOINT     = "best_model.pth"
LOG_FILE       = "training_log.csv"


# ── Training & validation helpers ────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        avg_loss: Average CrossEntropyLoss over all batches.
        accuracy: Fraction of training images classified correctly.
    """
    model.train()   # Enable dropout / batch norm in training mode

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # Clear gradients from previous batch
        outputs = model(images)         # Forward pass → raw logits [B, 50]
        loss    = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backpropagate gradients
        optimizer.step()                # Update final layer weights

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)   # Index of highest logit = predicted class
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return total_loss / total, correct / total


def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the model on the test set with no gradient computation.

    Returns:
        avg_loss:    Average loss over the test set.
        top1_acc:    Fraction of images where the top prediction is correct.
        top3_acc:    Fraction of images where the correct breed is in the
                     top 3 predictions. This is the headline metric for our
                     project since we output the 3 most likely breeds.
    """
    model.eval()    # Disable dropout / use running stats for batch norm

    total_loss, top1_correct, top3_correct, total = 0.0, 0, 0, 0

    with torch.no_grad():   # No gradients needed during evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            # Top-1: is the single best prediction correct?
            _, top1_pred = outputs.max(1)
            top1_correct += top1_pred.eq(labels).sum().item()

            # Top-3: is the correct label among the 3 highest scores?
            _, top3_pred = outputs.topk(3, dim=1)
            top3_correct += sum(
                labels[i].item() in top3_pred[i].tolist()
                for i in range(labels.size(0))
            )

            total += labels.size(0)

    return total_loss / total, top1_correct / total, top3_correct / total


# ── Main training loop ────────────────────────────────────────────────────────

def train(epochs: int, lr: float) -> None:

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading dataset")
    print("=" * 60)
    train_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    # ── 2. Build model ────────────────────────────────────────────────────────
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

    # Learning rate scheduler: reduce LR by 10x every 5 epochs.
    # Helps the model converge to a better minimum in later epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )

    # ── 3. Training loop ──────────────────────────────────────────────────────
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

        # Print a one-line summary for this epoch
        print(
            f"  Epoch {epoch:>2}/{epochs}  |  "
            f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.2%}  |  "
            f"Test loss: {test_loss:.4f}  Top-1: {top1_acc:.2%}  Top-3: {top3_acc:.2%}"
        )

        # Save best checkpoint
        if top1_acc > best_top1:
            best_top1 = top1_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"            ✓ New best Top-1 accuracy — checkpoint saved")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "test_loss":  round(test_loss,  4),
            "top1_acc":   round(top1_acc,   4),
            "top3_acc":   round(top3_acc,   4),
        })

    # ── 4. Write log CSV ──────────────────────────────────────────────────────
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # ── 5. Final summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Top-1 accuracy : {best_top1:.2%}")
    print(f"  Baseline (M1)       : 57.80%  (logistic regression)")
    print(f"  Improvement         : {(best_top1 - 0.578) * 100:+.1f} percentage points")
    print(f"  Checkpoint saved to : {CHECKPOINT}")
    print(f"  Training log saved  : {LOG_FILE}")


# ── Entry point ───────────────────────────────────────────────────────────────

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
