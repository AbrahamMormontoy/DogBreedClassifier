import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from src.imageClassification.dataset.image_loader import load_dataset
from src.imageClassification.logisticRegression.logistic_regression import lr_train, predict_single_image_proba

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "src", "imageClassification", "data")
BREEDS = ["Italian_greyhound", "Mexican_hairless", "Pomeranian"]
IMSIZE = (64, 64)


def predict_image_from_path(image_path, clf, imsize=(64, 64), to_gray=True):

    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_resized = cv2.resize(img, imsize)
    img_flattened = img_resized.flatten()
        
    # Get prediction
    result = predict_single_image_proba(clf, img_flattened)
    result['image_path'] = image_path
        
    return result


def main():
    print("Logistic Regression - Predict Single Image\n")
    print("Loading dataset:")
    X, y = load_dataset(DATA_DIR, imsize=IMSIZE, to_gray=True, breeds=BREEDS)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.167, random_state=42, stratify=y
    )
    
    print("\nTraining model:")
    clf = lr_train(X_train, y_train, n_components=150)
    print("Model trained successfully!")
    
    display_names = [b.split('_')[-1] if '_' in b else b for b in BREEDS]
    

    print("Predict your custom images")
    
    custom_images_dir = os.path.join(BASE_DIR, "data", "images")
    if os.path.exists(custom_images_dir):
        custom_files = [f for f in os.listdir(custom_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if custom_files:
            print(f"\nFound {len(custom_files)} image(s) in data/images/")
            
            for img_file in custom_files:
                img_path = os.path.join(custom_images_dir, img_file)
                print(f"\nPredicting: {img_file}")
                
                result = predict_image_from_path(img_path, clf, imsize=IMSIZE, to_gray=True)
                
                if result:
                    print(f"Predicted breed: {display_names[result['predicted_class']]}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print(f"Probabilities:")
                    for i, prob in enumerate(result['probabilities']):
                        print(f"     {display_names[i]:<18} {prob:.2%}")


if __name__ == "__main__":
    main()
