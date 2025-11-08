# CrossEntropy.py
import numpy as np

class CrossEntropy:
    """
    Cross-entropy accumulator (supports sigmoid or softmax outputs).
    Exposes .reset(), .add(), and .error (mean CE).
    """

    def __init__(self, eps: float = 1e-12, from_logits: bool = False, multi_class: bool = False):
        self.eps = eps
        self.from_logits = from_logits
        self.multi_class = multi_class
        self._sum = 0.0
        self._count = 0

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def _softmax(self, z):
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def add(self, y_pred, y_true):
        """
        y_pred: model outputs (probabilities by default; or logits if from_logits=True)
        y_true: one-hot (multi_class=True) or {0,1} for binary
        """
        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)

        # Handle logits if requested
        if self.from_logits:
            if self.multi_class:
                yp = self._softmax(yp)
            else:
                yp = self._sigmoid(yp)

        yp = np.clip(yp, self.eps, 1.0 - self.eps)

        if self.multi_class:
            # Cross-entropy for one-hot targets: -sum(y * log(p)) per sample
            # Supports batch or single sample
            if yp.ndim == 1:
                ce = -float(np.sum(yt * np.log(yp)))
                n = 1
            else:
                ce = -float(np.sum(yt * np.log(yp)))
                n = int(yp.shape[0])
        else:
            # Binary cross-entropy: - y*log(p) - (1-y)*log(1-p)
            ce_term = -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
            ce = float(np.sum(ce_term))
            n = int(np.size(yt))

        self._sum += ce
        self._count += n

    @property
    def error(self) -> float:
        """Mean cross-entropy over all added samples/elements."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
