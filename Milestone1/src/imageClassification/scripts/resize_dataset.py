from os import listdir
from os.path import isfile, join
import cv2
import argparse
from imageClassification.datasets.image_loader import load_imgs
from imageClassification.preprocessing.image_transform import crop_resize

# Following https://www.geeksforgeeks.org/python/resize-multiple-img-using-opencv-python/
def main():
    ap = argparse.ArgumentParser(description='Resize img in a directory.')

    ap.add_argument("-i",
                    "--image_dir",
                    required=True,
                    help="Path to the directory containing img to resize.")

    args = vars(ap.parse_args())

    images = load_imgs(args["image_dir"])

    for filename, image in images:
        img = crop_resize(image, (224, 224))
        cv2.imwrite(join(args["image_dir"], filename), img)

if __name__ == "__main__":
    main()