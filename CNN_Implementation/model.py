"""
model.py
--------
Builds the dog breed classifier by adapting a pretrained ResNet-18 model
using transfer learning.

Architecture overview:
    ResNet-18 (pretrained, frozen) --> New FC layer (512 -> NUM_CLASSES)

The pretrained ResNet-18 layers act as a fixed feature extractor.
Only the final fully-connected layer is trained — it learns to map
those features to our 50 dog breeds.

Usage:
    from model import build_model
    model, criterion, optimizer = build_model(num_classes=50)
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES  = 50       # One output node per breed
LEARNING_RATE = 0.001   # Adam learning rate for the final layer


def build_model(
    num_classes: int   = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    device: torch.device | None = None,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    Constructs the transfer learning model, loss function, and optimizer.

    Steps performed:
        1. Load ResNet-18 with ImageNet pretrained weights.
        2. Freeze all base layer parameters so they are not updated
           during training — we only want to train the final layer.
        3. Replace the final fully-connected layer (originally 512 -> 1000
           for ImageNet's 1000 classes) with a new layer (512 -> num_classes).
        4. Define CrossEntropyLoss as the loss function.
        5. Attach an Adam optimizer that targets only the new final layer.

    Args:
        num_classes:    Number of output classes (50 breeds).
        learning_rate:  Learning rate for the Adam optimizer.
        device:         torch.device to move the model to.
                        Auto-detected (GPU if available, else CPU) if None.

    Returns:
        model:     The adapted ResNet-18 model, moved to device.
        criterion: The loss function (CrossEntropyLoss).
        optimizer: Adam optimizer targeting only the final layer.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Step 1: Load pretrained ResNet-18 ────────────────────────────────────
    # weights=DEFAULT loads the best available ImageNet pretrained weights.
    # ResNet-18 has 18 layers and was trained on 1.2 million images across
    # 1000 classes, giving it rich general visual knowledge.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # ── Step 2: Freeze all base layers ───────────────────────────────────────
    # Setting requires_grad=False tells PyTorch not to compute gradients
    # for these parameters, so they will not be updated during backpropagation.
    # This preserves the pretrained feature extraction capability.
    for param in model.parameters():
        param.requires_grad = False

    # ── Step 3: Replace the final fully-connected layer ───────────────────────
    # ResNet-18's original classifier: Linear(512 -> 1000) for ImageNet.
    # We replace it with Linear(512 -> 50) for our 50 breeds.
    # Because we create a new layer, its requires_grad defaults to True —
    # this is the ONLY layer that will be trained.
    in_features = model.fc.in_features      # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    # ── Step 4: Loss function ─────────────────────────────────────────────────
    # CrossEntropyLoss is the standard loss for multi-class classification.
    # Internally it applies log-softmax to the model's raw output (logits)
    # and then computes negative log-likelihood against the true label.
    criterion = nn.CrossEntropyLoss()

    # ── Step 5: Optimizer ─────────────────────────────────────────────────────
    # We pass only model.fc.parameters() so that Adam only updates the new
    # final layer. Passing all parameters would unfreeze everything.
    # Adam (Adaptive Moment Estimation) adjusts the learning rate per
    # parameter and is generally more stable than vanilla SGD.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    # ── Sanity check ──────────────────────────────────────────────────────────
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready.")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable:,}  "
          f"({trainable / total_params:.1%} of total)")
    print(f"  Output classes       : {num_classes}")

    return model, criterion, optimizer


if __name__ == "__main__":
    # Quick standalone test — run this file directly to confirm the model builds
    model, criterion, optimizer = build_model()

    # Feed a dummy batch through to verify shapes
    dummy_input = torch.randn(4, 3, 224, 224)   # Batch of 4 RGB 224x224 images
    output = model(dummy_input)
    print(f"\nForward pass OK — output shape: {output.shape}")  # Expected: [4, 50]
