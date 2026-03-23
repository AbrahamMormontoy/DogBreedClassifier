"""
rename_breeds.py
----------------
Strips the Stanford Dogs dataset prefix from breed folder names.

Before: selected_breeds/n02085620-Chihuahua/
After:  selected_breeds/Chihuahua/

Usage:
    python rename_breeds.py
"""

import os
import re

BREEDS_DIR = "selected_breeds"

def clean_name(folder_name: str) -> str:
    """
    Remove the leading 'nXXXXXXXX-' prefix from a folder name.
    e.g. 'n02085620-Chihuahua' -> 'Chihuahua'
    """
    return re.sub(r'^n\d+-', '', folder_name)

def rename_breed_folders(base_dir: str) -> None:
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' folder not found. "
              f"Make sure you run this script from your project root.")
        return

    folders = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f))]

    if not folders:
        print("No folders found inside selected_breeds/.")
        return

    renamed, skipped = 0, 0

    for folder in sorted(folders):
        new_name = clean_name(folder)

        if new_name == folder:
            print(f"  Skipped (already clean): {folder}")
            skipped += 1
            continue

        src = os.path.join(base_dir, folder)
        dst = os.path.join(base_dir, new_name)

        if os.path.exists(dst):
            print(f"  Skipped (destination already exists): {dst}")
            skipped += 1
            continue

        os.rename(src, dst)
        print(f"  Renamed: {folder} -> {new_name}")
        renamed += 1

    print(f"\nDone. {renamed} folder(s) renamed, {skipped} skipped.")

if __name__ == "__main__":
    rename_breed_folders(BREEDS_DIR)
