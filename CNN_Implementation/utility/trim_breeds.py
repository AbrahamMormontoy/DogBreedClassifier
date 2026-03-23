"""
trim_breeds.py
--------------
Randomly removes images from each breed folder until every folder
contains exactly TARGET_COUNT images.

Folders that already have <= TARGET_COUNT images are left untouched.

Usage:
    python trim_breeds.py

    # Dry run (preview what would be deleted, nothing actually removed):
    python trim_breeds.py --dry-run
"""

import os
import random
import argparse

BREEDS_DIR    = "selected_breeds"
TARGET_COUNT  = 150
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder: str) -> list[str]:
    """Return a list of image filenames inside folder."""
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def trim_breeds(base_dir: str, target: int, dry_run: bool) -> None:
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' not found. "
              "Run this script from your project root.")
        return

    breed_folders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ])

    if not breed_folders:
        print("No breed folders found.")
        return

    if dry_run:
        print("=== DRY RUN — no files will be deleted ===\n")

    total_removed = 0

    for breed in breed_folders:
        folder_path = os.path.join(base_dir, breed)
        images      = get_images(folder_path)
        count       = len(images)

        if count <= target:
            print(f"  OK       {breed:<40} {count} images (no change needed)")
            continue

        to_remove = random.sample(images, count - target)

        print(f"  Trimming {breed:<40} {count} → {target}  "
              f"(removing {len(to_remove)})")

        for filename in to_remove:
            filepath = os.path.join(folder_path, filename)
            if not dry_run:
                os.remove(filepath)

        total_removed += len(to_remove)

    action = "would be" if dry_run else "were"
    print(f"\nDone. {total_removed} image(s) {action} removed across "
          f"{len(breed_folders)} breed folder(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim each breed folder to exactly TARGET_COUNT images."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without actually removing any files."
    )
    args = parser.parse_args()

    random.seed(42)   # Fixed seed so the same images are removed every run
    trim_breeds(BREEDS_DIR, TARGET_COUNT, dry_run=args.dry_run)
