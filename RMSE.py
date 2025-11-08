# RMSE.py
import numpy as np
from math import sqrt

class RMSE:
    """Simple RMSE accumulator with .reset(), .add(), and .error property."""

    def __init__(self):
        self._sum_sq = 0.0
        self._count = 0

    def reset(self):
        self._sum_sq = 0.0
        self._count = 0

    def add(self, y_pred, y_true):
        """
        y_pred, y_true: scalars or 1D/2D arrays/lists (batch or single).
        Accumulates sum of squared errors and count of elements.
        """
        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)
        diff = yp - yt
        self._sum_sq += float(np.sum(diff * diff))
        self._count += int(diff.size)

    @property
    def error(self) -> float:
        """Return current RMSE; 0.0 if no samples added."""
        if self._count == 0:
            return 0.0
        return sqrt(self._sum_sq / self._count)
