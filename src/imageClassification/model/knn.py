import numpy as np
from imageClassification.utils.common import distance

def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        dist = distance(X_train, x)
        nn_idx = np.argpartition(dist, k)[:k]
        nn_labels = y_train[nn_idx]

        val, counts = np.unique(nn_labels, return_counts=True)
        preds.append(val[np.argmax(counts)])
    return np.array(preds)