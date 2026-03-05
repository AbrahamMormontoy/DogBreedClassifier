import numpy as np

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2) ** 2, axis=1))