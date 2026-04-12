import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES  = 50
LEARNING_RATE = 0.001


def build_model(
    num_classes: int   = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    device: torch.device | None = None,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready.")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable:,}  "
          f"({trainable / total_params:.1%} of total)")
    print(f"  Output classes       : {num_classes}")

    return model, criterion, optimizer


if __name__ == "__main__":
    model, criterion, optimizer = build_model()

    dummy_input = torch.randn(4, 3, 224, 224)   # Batch of 4 RGB 224x224 images
    output = model(dummy_input)
    print(f"\nForward pass OK — output shape: {output.shape}")  # Expected: [4, 50]
