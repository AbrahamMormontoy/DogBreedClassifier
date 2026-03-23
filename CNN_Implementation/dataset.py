"""
dataset.py
----------
Defines the image transform pipelines and builds the train/test DataLoaders
for the 50-breed dog classifier.

ImageFolder automatically assigns a class label to each image based on
the name of the subfolder it lives in, so no manual labelling is needed
as long as selected_breeds/ has one folder per breed.

Usage (from train.py or a notebook):
    from dataset import get_dataloaders
    train_loader, test_loader, class_names = get_dataloaders()
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────

BREEDS_DIR  = "selected_breeds"   # Root folder containing one subfolder per breed
IMAGE_SIZE  = 224                  # ResNet-18 expects 224x224 inputs
BATCH_SIZE  = 32
NUM_WORKERS = 2                    # Parallel workers for data loading
TRAIN_RATIO = 0.75                 # 75% train, 25% test

# ImageNet normalisation values — mandatory when using a pretrained ResNet
# These are the mean and std of the dataset ResNet was originally trained on.
# Applying the same normalisation ensures our pixel values are in the same
# range the model has already learned to interpret.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transform Pipelines ──────────────────────────────────────────────────────

# Training pipeline — includes data augmentation to artificially increase
# the variety of examples the model sees, which reduces overfitting.
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize every image to 224x224
    transforms.RandomHorizontalFlip(),            # Randomly mirror image left-right
    transforms.RandomRotation(15),                # Randomly rotate up to ±15 degrees
    transforms.ToTensor(),                        # Convert PIL image to float tensor [0, 1]
    transforms.Normalize(                         # Shift pixel values to ImageNet distribution
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])

# Test/validation pipeline — NO augmentation.
# We want deterministic outputs at evaluation time so accuracy numbers
# are reproducible and comparable across runs.
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    breeds_dir: str = BREEDS_DIR,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Loads the full dataset from breeds_dir, splits it into train and test
    sets, applies the appropriate transforms to each, and returns two
    DataLoaders plus the list of class names.

    Args:
        breeds_dir:  Path to the folder containing one subfolder per breed.
        batch_size:  Number of images per batch.
        train_ratio: Fraction of data used for training (default 0.75).
        seed:        Random seed for reproducible splits.

    Returns:
        train_loader: DataLoader for the training set (with augmentation).
        test_loader:  DataLoader for the test set (no augmentation).
        class_names:  List of breed names inferred from folder names,
                      sorted alphabetically. Index i = class label i.
    """

    # Load the full dataset with training transforms first so we can split it.
    # We'll re-apply the correct transform to each subset below.
    full_dataset = datasets.ImageFolder(root=breeds_dir, transform=train_transforms)
    class_names  = full_dataset.classes

    # Compute split sizes
    total      = len(full_dataset)
    train_size = int(total * train_ratio)
    test_size  = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    # Override the transform on the test subset so it gets no augmentation
    test_subset.dataset = datasets.ImageFolder(
        root=breeds_dir, transform=test_transforms
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle every epoch so batches vary
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,          # No need to shuffle test data
        num_workers=NUM_WORKERS,
    )

    print(f"Dataset loaded from: {breeds_dir}")
    print(f"  Total images : {total}")
    print(f"  Training     : {train_size} ({train_ratio:.0%})")
    print(f"  Test         : {test_size} ({1 - train_ratio:.0%})")
    print(f"  Classes      : {len(class_names)} breeds")
    print(f"  Batch size   : {batch_size}")

    return train_loader, test_loader, class_names
