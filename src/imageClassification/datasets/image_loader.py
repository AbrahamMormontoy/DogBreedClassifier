from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import cv2

# Find all the images in the directory
def load_imgs(image_dir):
    image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    print(f"Found {len(image_files)} images in {image_dir}.")
    images = [] 
    for file in image_files:
        image_path = join(image_dir, file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        images.append((file, img))
    
    return images
