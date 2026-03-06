from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def build_lr(C=0.5, max_iter=1000, n_components=150):
    """
    Creates a pipeline that scales data, applies PCA for dimensionality 
    reduction, and then fits a Logistic Regression model.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
        ("lr",     LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced", # Good for uneven datasets
            random_state=42,
        )),
    ])

def lr_predict(X_train, y_train, X_test, C=0.5, max_iter=1000, n_components=150):
    """Fits the model and returns predictions for the test set."""
    clf = build_lr(C=C, max_iter=max_iter, n_components=n_components)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def lr_train(X_train, y_train, C=0.5, max_iter=1000, n_components=150):
    """Trains and returns the fitted model."""
    clf = build_lr(C=C, max_iter=max_iter, n_components=n_components)
    clf.fit(X_train, y_train)
    return clf

def predict_single_image_proba(clf, X_single):
    """
    Predicts the probability for a single image.
    
    Args:
        clf: Trained classifier pipeline
        X_single: Single flattened image array (1D)
    
    Returns:
        dict: Dictionary with class probabilities
    """
    # Reshape to 2D if needed (sklearn expects 2D arrays)
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    # Get probability predictions
    probas = clf.predict_proba(X_single)[0]
    
    # Get the predicted class
    pred_class = clf.predict(X_single)[0]
    
    return {
        'predicted_class': int(pred_class),
        'probabilities': probas,
        'confidence': float(probas[pred_class])
    }