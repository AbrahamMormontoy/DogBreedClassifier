import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
import matplotlib.pyplot as plt

from src.model import build_model
from src.dataset import BREEDS_DIR, test_transforms


TYPE_IMAGE = (".jpg", ".jpeg", ".png")


def load_class_names(breeds_dir: str) -> List[str]:
    ds = datasets.ImageFolder(root=breeds_dir)
    return ds.classes


def load_checkpoint_weights(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]

        looks_like_state_dict = all(isinstance(k, str) for k in ckpt.keys())
        if looks_like_state_dict:
            return ckpt


def build_inference_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
):
    model, _, _ = build_model(
        num_classes=num_classes,
        learning_rate=0.001,
        device=device,
    )
    state_dict = load_checkpoint_weights(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_topk_for_image(model: torch.nn.Module, image_path: str, class_names: List[str], device: torch.device, topk: int = 3,) -> List[Tuple[str, float]]:
    image = Image.open(image_path).convert("RGB")
    x = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(k=topk, dim=1)

    results = []
    for p, idx in zip(top_probs[0].tolist(), top_idxs[0].tolist()):
        results.append((class_names[idx], p))
    return results


def collect_images(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = []
        for name in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, name)
            if os.path.isfile(full) and name.lower().endswith(TYPE_IMAGE):
                files.append(full)
        return files

    raise FileNotFoundError(f"No existe la ruta: {input_path}")


def plot_predictions(image_paths: List[str], predictions: dict, out_path: str):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 2, figsize=(14, 4 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(os.path.basename(image_path), fontsize=12, fontweight="bold")
        axes[idx, 0].axis("off")
        
        preds = predictions[image_path]
        breeds = [p[0] for p in preds]
        confidences = [p[1] * 100 for p in preds]
        
        bars = axes[idx, 1].barh(breeds, confidences, color="steelblue")
        axes[idx, 1].set_xlabel("Confidence (%)", fontsize=10)
        axes[idx, 1].set_xlim(0, 100)
        
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            axes[idx, 1].text(conf + 1, i, f"{conf:.2f}%", va="center", fontsize=9)
        
        axes[idx, 1].set_title("Top-K Predictions", fontsize=12, fontweight="bold")
        axes[idx, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n<Saved Result Plot:> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="best_model.pth",
    )
    parser.add_argument(
        "--breeds-dir",
        default=BREEDS_DIR,
    )
    parser.add_argument(
        "--input",
        default="predictionimages",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Classes
    class_names = load_class_names(args.breeds_dir)

    # Build Model
    model = build_inference_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(class_names),
        device=device,
    )

    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"No images found in {args.input}")
        return

    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {len(class_names)}")
    print("=" * 70)

    predictions = {}
    
    for image_path in image_paths:
        preds = predict_topk_for_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                device=device,
                topk=args.topk,
                )
        predictions[image_path] = preds

        print(f"\nImagen: {image_path}")
        for rank, (breed, prob) in enumerate(preds, start=1):
            print(f"  {rank}. {breed:<30} {prob:.2%}")
    
    if len(image_paths) > 0:
        out_file = os.path.join("predictions", "predictions.png")
        plot_predictions(image_paths, predictions, out_path=out_file)


if __name__ == "__main__":
    main()