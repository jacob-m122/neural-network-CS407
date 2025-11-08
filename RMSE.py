# RMSE.py
import numpy as np

class RMSE:
    """Root Mean Squared Error metric with a simple mini-batch lifecycle."""

    def __init__(self):
        self._sum_mse = 0.0    # accumulate per-sample MSE (already averaged over outputs)
        self._count = 0        # number of samples seen

    def reset(self):
        self._sum_mse = 0.0
        self._count = 0

    def update(self, y_true, y_pred):
        """
        y_true, y_pred: array-like, shape (n_outputs,) or (1,) for scalar.
        Accumulates MSE for one sample (averaged over outputs).
        """
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch in RMSE.update: {yt.shape} vs {yp.shape}")
        diff = yt - yp
        mse = float(np.dot(diff, diff)) / diff.size
        self._sum_mse += mse
        self._count += 1

    def value(self) -> float:
        """Return RMSE over all updates since last reset()."""
        if self._count == 0:
            return 0.0
        mean_mse = self._sum_mse / self._count
        return mean_mse ** 0.5

    # Optional helper for callers that just want a one-off RMSE
    @staticmethod
    def distance(y_true, y_pred) -> float:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch in RMSE.distance: {yt.shape} vs {yp.shape}")
        diff = yt - yp
        mse = float(np.dot(diff, diff)) / diff.size
        return mse ** 0.5
