import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

BREEDS_DIR = "dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
TRAIN_RATIO = 0.75 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])



def get_dataloaders(
    breeds_dir: str = BREEDS_DIR,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
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

    test_subset.dataset = datasets.ImageFolder(
        root=breeds_dir, transform=test_transforms
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Dataset loaded from: {breeds_dir}")
    print(f"  Total images : {total}")
    print(f"  Training     : {train_size} ({train_ratio:.0%})")
    print(f"  Test         : {test_size} ({1 - train_ratio:.0%})")
    print(f"  Classes      : {len(class_names)} breeds")
    print(f"  Batch size   : {batch_size}")

    return train_loader, test_loader, class_names
