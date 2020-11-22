import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion(targets: np.array, predictions: np.array, ):
    cm = confusion_matrix(targets, predictions)