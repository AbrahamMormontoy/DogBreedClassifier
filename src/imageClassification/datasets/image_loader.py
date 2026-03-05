from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import cv2
from imageClassification.preprocessing.image_transform import crop_resize, normalize

# Find all the images in the directory
def load_imgs(image_dir):
    image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    print(f"Found {len(image_files)} images in {image_dir}.")
    images = [] 
    for file in image_files:
        image_path = join(image_dir, file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        images.append((file, img))
    
    return images


def flatten_images(images, imsize=None, to_gray=True, do_normalize=True, label_delimiter='_'):

    X_list = []
    y_list = []

    for filename, img in images:
        if imsize is not None:
            try:
                img = crop_resize(img, imsize)
            except Exception:
                # If crop/resize fails, skip this image
                continue

        if to_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if do_normalize:
            try:
                img = normalize(img)
            except Exception:
                # fallback: simple scaling
                img = img.astype(np.float32) / 255.0

        flat = img.ravel().astype(np.float32)
        X_list.append(flat)
        # y_list.append(_extract_label_from_filename(filename, delimiter=label_delimiter))

    if len(X_list) == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    # Stack into array; handle varying image sizes by padding/truncating is out-of-scope
    # We require all images to have the same number of features here (same imsize / channels)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=object)
    return X, y


def prepare_knn_data(image_dir, imsize=(224, 224), to_gray=True, do_normalize=True, label_delimiter='_'):

    images = load_imgs(image_dir)
    X, y = flatten_images(images, imsize=imsize, to_gray=to_gray, do_normalize=do_normalize, label_delimiter=label_delimiter)
    return X, y
