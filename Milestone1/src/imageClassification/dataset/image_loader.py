import os
import cv2
import numpy as np
<<<<<<< HEAD
from imageClassification.preprocessing.image_transform import normalize
=======

>>>>>>> fd3b08a1b4069d22bcab92803d5c65dd6c5aac03
def load_dataset(data_dir, imsize=(64, 64), to_gray=True, breeds=None):
    """
    Loads images from breed folders, resizes them, and flattens them into vectors.
    """
    X = []
    y = []
    
    if breeds is None:
        breeds = sorted(os.listdir(data_dir))

    print(f"Targeting breeds: {breeds}")

    for idx, breed in enumerate(breeds):
        path = os.path.join(data_dir, breed)
        if not os.path.isdir(path):
            continue
            
        count = 0
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                
                if to_gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                img_resized = cv2.resize(img, imsize)
<<<<<<< HEAD
                normalized_img = normalize(img_resized)
                # Flattening 64x64 into 4096 features
                X.append(normalized_img.flatten())
=======
                # Flattening 64x64 into 4096 features
                X.append(img_resized.flatten())
>>>>>>> fd3b08a1b4069d22bcab92803d5c65dd6c5aac03
                y.append(idx)
                count += 1
            except Exception:
                continue
        print(f"  - Loaded {count} images for {breed}")
                
    return np.array(X), np.array(y)