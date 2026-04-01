import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from model import build_model
from dataset import test_transforms


TYPE_IMAGE = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_class_names(breeds_dir: str) -> List[str]:
    ds = datasets.ImageFolder(root=breeds_dir)
    return ds.classes


def load_checkpoint_weights(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Soporta dos formatos:
    # 1) state_dict puro
    # 2) diccionario con llave model_state_dict/state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]

        # Si parece ser directamente un state_dict (keys tipo layer.weight)
        looks_like_state_dict = all(isinstance(k, str) for k in ckpt.keys())
        if looks_like_state_dict:
            return ckpt

    raise ValueError(
        "Formato de checkpoint no reconocido. "
        "Esperaba state_dict o dict con 'model_state_dict'/'state_dict'."
    )


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


def predict_topk_for_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    device: torch.device,
    topk: int = 3,
) -> List[Tuple[str, float]]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Predice Top-K razas para imagen(es) con el modelo entrenado."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta a una imagen o carpeta con imágenes.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_model.pth",
        help="Checkpoint del modelo (default: best_model.pth).",
    )
    parser.add_argument(
        "--breeds-dir",
        default="selected_breeds",
        help="Carpeta de dataset para reconstruir class_names (default: selected_breeds).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Cantidad de predicciones a mostrar (default: 3).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.breeds_dir)

    model = build_inference_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(class_names),
        device=device,
    )

    image_paths = collect_images(args.input)
    if not image_paths:
        print("No se encontraron imágenes válidas en la ruta.")
        return

    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Clases: {len(class_names)}")
    print("=" * 70)

    for image_path in image_paths:
        preds = predict_topk_for_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                device=device,
                topk=args.topk,
                )

        print(f"\nImagen: {image_path}")
        for rank, (breed, prob) in enumerate(preds, start=1):
            print(f"  {rank}. {breed:<30} {prob:.2%}")


if __name__ == "__main__":
    main()