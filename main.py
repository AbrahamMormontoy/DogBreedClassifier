import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from src.imageClassification.dataset.image_loader import load_dataset
from src.imageClassification.logisticRegression.logistic_regression import lr_predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "src", "imageClassification", "data")
BREEDS = ["Italian_greyhound", "Mexican_hairless", "Pomeranian"]
IMSIZE = (64, 64)

def main():
    print("--- Logistic Regression ---")
    
    # Load data
    X, y = load_dataset(DATA_DIR, imsize=IMSIZE, to_gray=True, breeds=BREEDS)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.167, random_state=42, stratify=y
    )
    
    # Train & Predict
    y_pred = lr_predict(X_train, y_train, X_test, n_components=150)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOVERALL ACCURACY: {acc:.3f} out of {sum(y_test == y_pred)}/{len(y_test)} samples")

    # Use classification report from sklearn to get precision, recall, f1-score
    display_names = [b.split('-')[-1] for b in BREEDS]
    
    print("\nTECHNICAL REPORT: LOGISTIC REGRESSION\n")
    print(classification_report(y_test, y_pred, target_names=display_names))

    
if __name__ == "__main__":
    main()