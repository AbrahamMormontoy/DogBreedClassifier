import os
import csv
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.model import build_model
from src.dataset import BREEDS_DIR, train_transforms, test_transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_balanced_indices(base_dataset, train_per_class: int, test_per_class: int, seed: int):
    class_to_indices = defaultdict(list)
    for idx, (_, y) in enumerate(base_dataset.samples):
        class_to_indices[y].append(idx)

    rng = random.Random(seed)
    train_idx, test_idx = [], []

    for c, idxs in class_to_indices.items():
        rng.shuffle(idxs)
        if len(idxs) < train_per_class + test_per_class:
            raise ValueError(
                f"Clase {base_dataset.classes[c]}: necesita >= {train_per_class + test_per_class}, "
                f"tiene {len(idxs)}."
            )
        train_idx.extend(idxs[:train_per_class])
        test_idx.extend(idxs[train_per_class:train_per_class + test_per_class])

    return train_idx, test_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_per_class(model, loader, num_classes, device):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)

        for yi, pi in zip(y.cpu().numpy(), pred.cpu().numpy()):
            total[yi] += 1
            if yi == pi:
                correct[yi] += 1

    per_class_acc = np.divide(correct, total, out=np.zeros_like(correct, dtype=float), where=total > 0)
    mean_acc = float(per_class_acc.mean())
    return per_class_acc, mean_acc


def run_experiment(breeds_dir: str, small_n: int, large_n: int, test_per_class: int, epochs: int):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    base = datasets.ImageFolder(root=breeds_dir)
    class_names = base.classes
    num_classes = len(class_names)

    train_ds_full = datasets.ImageFolder(root=breeds_dir, transform=train_transforms)
    test_ds_full = datasets.ImageFolder(root=breeds_dir, transform=test_transforms)

    sizes = [small_n, large_n]
    per_class_results = {}
    mean_results = []

    for n in sizes:
        print(f"\n{'─'*60}")
        print(f"Training with {n} images per class")
        print(f"{'─'*60}")
        
        train_idx, test_idx = make_balanced_indices(base, n, test_per_class, seed=42 + n)

        train_ds = Subset(train_ds_full, train_idx)
        test_ds = Subset(test_ds_full, test_idx)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        print(f"  Train set: {len(train_ds)} images ({n} per class)")
        print(f"  Test set:  {len(test_ds)} images ({test_per_class} per class)\n")

        model, criterion, optimizer = build_model(num_classes=num_classes, learning_rate=0.001, device=device)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            print(f"  Epoch {ep:2d}/{epochs}  Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2%}")

        per_class_acc, mean_acc = evaluate_per_class(model, test_loader, num_classes, device)
        per_class_results[n] = per_class_acc
        mean_results.append((n, mean_acc))
        
        print(f"\n  ✓ Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)\n")

    per_class_csv = os.path.join("logs", "per_class_accuracy.csv")
    with open(per_class_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["breed", f"{small_n}_images", f"{large_n}_images"])
        for i, name in enumerate(class_names):
            w.writerow([name, per_class_results[small_n][i], per_class_results[large_n][i]])
    print(f"<Saved:> {per_class_csv}")

    mean_csv = os.path.join("logs", "mean_accuracy.csv")
    with open(mean_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_images_per_class", "mean_accuracy"])
        for n, m in mean_results:
            w.writerow([n, m])
    print(f"<Saved:> {mean_csv}")

    return class_names, per_class_results, mean_results, small_n, large_n


def plot_per_class_comparison(class_names, per_class_results, small_n, large_n, out_path):
    small_acc = per_class_results[small_n]
    large_acc = per_class_results[large_n]

    order = np.argsort(large_acc)
    class_sorted = [class_names[i] for i in order]
    small_sorted = small_acc[order]
    large_sorted = large_acc[order]

    y = np.arange(len(class_sorted))

    fig, ax = plt.subplots(figsize=(10, 18))
    
    bar_height = 0.35
    ax.barh(y - bar_height/2, small_sorted, bar_height, color="navy", label=f"{small_n} Train")
    ax.barh(y + bar_height/2, large_sorted, bar_height, color="maroon", label=f"{large_n} Train")

    ax.set_yticks(y)
    ax.set_yticklabels(class_sorted, fontsize=7)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-class accuracy comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1.05])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"<Saved:> {out_path}")


def plot_mean_accuracy_curve(mean_results, out_path):
    xs = np.array([x for x, _ in mean_results], dtype=float)
    ys = np.array([y for _, y in mean_results], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(xs, ys, color="red", marker="o", markersize=8, linewidth=2, markerfacecolor="red")
    ax.set_xlabel("Number of Training Images per Class", fontsize=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_title("Mean Accuracy vs Training Images per Class", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--small",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--large",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    class_names, per_class_results, mean_results, small_n, large_n = run_experiment(
        breeds_dir=BREEDS_DIR,
        small_n=args.small,
        large_n=args.large,
        test_per_class=30,
        epochs=args.epochs,
    )

    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}\n")
    
    plot_per_class_comparison(
        class_names,
        per_class_results,
        small_n=small_n,
        large_n=large_n,
        out_path=os.path.join("plots", "per_class_compare.png")
    )
    
    plot_mean_accuracy_curve(
        mean_results,
        out_path=os.path.join("plots", "mean_accuracy_curve.png")
    )

    print(f"\n{'='*60}")
    print("Experiment completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()