import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Imports
from src.imageClassification.dataset.image_loader import load_dataset
from src.imageClassification.logisticRegression.logistic_regression import lr_train, predict_single_image_proba

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "src", "imageClassification", "data")
BREEDS = ["Italian_greyhound", "Mexican_hairless", "Pomeranian"]
IMSIZE = (64, 64)


def predict_image_from_path(image_path, clf, imsize=(64, 64), to_gray=True):
    """
    Load an image from a file path and predict its breed with probabilities.
    
    Args:
        image_path: Path to the image file
        clf: Trained classifier
        imsize: Size to resize the image to
        to_gray: Whether to convert to grayscale
    
    Returns:
        dict: Prediction results with probabilities
    """
    try:
        # Load and preprocess the image
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
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def main():
    """Demo: Train model and predict single image probability"""
    print("=" * 60)
    print("SINGLE IMAGE PREDICTION DEMO")
    print("=" * 60)
    
    # Load data
    print("\nLoading dataset...")
    X, y = load_dataset(DATA_DIR, imsize=IMSIZE, to_gray=True, breeds=BREEDS)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.167, random_state=42, stratify=y
    )
    
    # Train the model
    print("\nTraining model...")
    clf = lr_train(X_train, y_train, n_components=150)
    print("Model trained successfully!")
    
    # Demo 1: Predict a random test image
    print("\n" + "-" * 60)
    print("DEMO 1: Random test image prediction")
    print("-" * 60)
    
    random_idx = np.random.randint(0, len(X_test))
    single_image = X_test[random_idx]
    true_label = y_test[random_idx]
    
    # Get probability prediction
    result = predict_single_image_proba(clf, single_image)
    
    display_names = [b.split('_')[-1] if '_' in b else b for b in BREEDS]
    
    print(f"\nTrue breed: {display_names[true_label]}")
    print(f"Predicted breed: {display_names[result['predicted_class']]}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    
    print("Class probabilities:")
    for i, prob in enumerate(result['probabilities']):
        print(f"  {display_names[i]:<20} {prob:.2%}")
    
    # Demo 2: Predict from image file path
    print("\n" + "-" * 60)
    print("DEMO 2: Predict from image file")
    print("-" * 60)
    
    # Find a sample image from the dataset
    sample_breed = BREEDS[0]
    sample_dir = os.path.join(DATA_DIR, sample_breed)
    if os.path.exists(sample_dir):
        image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            sample_image_path = os.path.join(sample_dir, image_files[0])
            print(f"\nPredicting image: {sample_image_path}")
            
            result = predict_image_from_path(sample_image_path, clf, imsize=IMSIZE, to_gray=True)
            
            if result:
                print(f"\nPredicted breed: {display_names[result['predicted_class']]}")
                print(f"Confidence: {result['confidence']:.2%}\n")
                
                print("Class probabilities:")
                for i, prob in enumerate(result['probabilities']):
                    print(f"  {display_names[i]:<20} {prob:.2%}")
    
    # Demo 3: Predict user's custom images from data/images
    print("\n" + "-" * 60)
    print("DEMO 3: Predict your custom images")
    print("-" * 60)
    
    custom_images_dir = os.path.join(BASE_DIR, "data", "images")
    if os.path.exists(custom_images_dir):
        custom_files = [f for f in os.listdir(custom_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if custom_files:
            print(f"\nFound {len(custom_files)} image(s) in data/images/")
            
            for img_file in custom_files:
                img_path = os.path.join(custom_images_dir, img_file)
                print(f"\n📸 Predicting: {img_file}")
                
                result = predict_image_from_path(img_path, clf, imsize=IMSIZE, to_gray=True)
                
                if result:
                    print(f"   Predicted breed: {display_names[result['predicted_class']]}")
                    print(f"   Confidence: {result['confidence']:.2%}")
                    print(f"   Probabilities:")
                    for i, prob in enumerate(result['probabilities']):
                        print(f"     {display_names[i]:<18} {prob:.2%}")
        else:
            print("\n⚠️  No images found in data/images/")
            print("   Add your dog images (.jpg, .jpeg, .png) to data/images/ and run again!")
    else:
        print("\n⚠️  Directory data/images/ doesn't exist")
        print("   Create it and add your dog images there!")
    
    print("\n" + "=" * 60)
    print("To predict your own image:")
    print("  from predict_single import predict_image_from_path, lr_train")
    print("  # Train your model first")
    print("  clf = lr_train(X_train, y_train)")
    print("  # Then predict")
    print("  result = predict_image_from_path('path/to/your/image.jpg', clf)")
    print("=" * 60)


if __name__ == "__main__":
    main()
