import numpy as np

def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum() / float(len(y_true))

def f1(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    precision = tp / (tp + fp) if (tp +fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 2 * recall * precision / (precision + recall) if (precision + recall) > 0 else 0.0
