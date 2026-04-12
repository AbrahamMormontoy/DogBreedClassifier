import numpy as np
import cv2
import matplotlib.pyplot as plt


def crop_resize(img, imsize):
    # Crop the image to a square
    height, width = img.shape[:2]
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    img = img[start_y:start_y + size, start_x:start_x + size]

    # Resize the image to size
    img = cv2.resize(img, imsize)

    # Save the resized image back to the directory
    return img[start_y:start_y + size, start_x:start_x + size]


def normalize(gray_img):
    norm_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img))
    return norm_img

def filter_img():
    pass
