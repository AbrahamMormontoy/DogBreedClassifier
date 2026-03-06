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