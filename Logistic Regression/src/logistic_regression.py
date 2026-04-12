from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def build_lr(C=0.5, max_iter=1000, n_components=150):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
        ("lr",     LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])

def lr_predict(X_train, y_train, X_test, C=0.5, max_iter=1000, n_components=150):
    clf = build_lr(C=C, max_iter=max_iter, n_components=n_components)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def lr_train(X_train, y_train, C=0.5, max_iter=1000, n_components=150):
    clf = build_lr(C=C, max_iter=max_iter, n_components=n_components)
    clf.fit(X_train, y_train)
    return clf

def predict_single_image_proba(clf, X_single):
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    probas = clf.predict_proba(X_single)[0]
    
    pred_class = clf.predict(X_single)[0]
    
    return {
        'predicted_class': int(pred_class),
        'probabilities': probas,
        'confidence': float(probas[pred_class])
    }