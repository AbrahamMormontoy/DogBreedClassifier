import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import argparse

# Following https://www.geeksforgeeks.org/python/resize-multiple-images-using-opencv-python/

ap = argparse.ArgumentParser(description='Resize images in a directory.')

ap.add_argument("-i",
                "--image_dir",
                required=True,
                help="Path to the directory containing images to resize.")

args = vars(ap.parse_args())

# Find all the images in the directory
image_dir = args["image_dir"]
image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
print(f"Found {len(image_files)} images in {image_dir}.")
images = np.empty(len(image_files), dtype=object)

# Resize each image and store it in the array
for i, image_file in enumerate(image_files):
    image_path = join(image_dir, image_file)
    images[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Crop the image to a square
    height, width = images[i].shape[:2]
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    images[i] = images[i][start_y:start_y + size, start_x:start_x + size]

    # Resize the image to 224x224
    images[i] = cv2.resize(images[i], (224, 224))

    # Save the resized image back to the directory
    cv2.imwrite(image_path, images[i])

