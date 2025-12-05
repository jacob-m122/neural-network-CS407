# CrossEntropy.py
import numpy as np

class CrossEntropy:
    """
    Cross-entropy accumulator compatible with RMSE.
    Exposes reset(), add(y_pred, y_true), and .error (mean CE).
    This uses elementwise binary cross-entropy, which also works
    on one-hot multi-class outputs from sigmoid units.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self._sum = 0.0
        self._count = 0

    def reset(self):
        """Clear accumulated loss and count."""
        self._sum = 0.0
        self._count = 0

    def add(self, y_pred, y_true):
        """
        Accumulate cross-entropy for a single sample.

        y_pred: iterable of model outputs (e.g., sigmoid activations)
        y_true: iterable of targets (0/1 or one-hot)
        """
        yp = np.asarray(y_pred, dtype=float).ravel()
        yt = np.asarray(y_true, dtype=float).ravel()

        # clip predictions to avoid log(0)
        eps = self.eps
        yp = np.clip(yp, eps, 1.0 - eps)

        # elementwise binary cross-entropy:
        # - y*log(p) - (1-y)*log(1-p)
        ce_term = -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
        ce = float(np.sum(ce_term))
        n = int(yt.size)

        self._sum += ce
        self._count += n

    @property
    def error(self) -> float:
        """Mean cross-entropy over all added elements."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
