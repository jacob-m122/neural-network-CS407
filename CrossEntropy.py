import numpy as np

class CrossEntropy:
    """
    Elementwise binary cross-entropy suitable for sigmoid output neurons.
    This supports multi-class one-hot outputs by applying BCE to each output unit.
    """

    def __call__(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def derivative(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
    