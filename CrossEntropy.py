# CrossEntropy.py
import numpy as np

class CrossEntropy:
    """
    Binary (or elementwise) cross-entropy metric:
    CE = -mean( y*log(p) + (1-y)*log(1-p) )
    Assumes y_true in {0,1} and y_pred are probabilities in (0,1).
    """

    def __init__(self, eps: float = 1e-12):
        self._sum_ce = 0.0
        self._count = 0
        self._eps = eps

    def reset(self):
        self._sum_ce = 0.0
        self._count = 0

    def update(self, y_true, y_pred):
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch in CrossEntropy.update: {yt.shape} vs {yp.shape}")
        # clamp to avoid log(0)
        yp = np.clip(yp, self._eps, 1.0 - self._eps)
        ce = -np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
        self._sum_ce += float(ce)
        self._count += 1

    def value(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum_ce / self._count

    @staticmethod
    def distance(y_true, y_pred, eps: float = 1e-12) -> float:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        yp = np.clip(yp, eps, 1.0 - eps)
        return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))
