"""Binary cross-entropy loss."""

import numpy as np

_EPS = 1e-12


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module -- see the note below.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities, shape (m,)
                or (m, 1). Clipped away from exactly 0 or 1 before use --
                see "Numerical stability" above.
            targets (np.ndarray): true labels, same shape as predictions,
                values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        a = np.clip(predictions, _EPS, 1.0 - _EPS)
        self.a = a
        self.y = targets
        loss = -np.mean(targets * np.log(a) + (1 - targets) * np.log(1 - a))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions passed to
            forward. Uses the same clipped predictions here as in forward
            -- see "Numerical stability" above.
        """
        m = self.a.shape[0]
        return -(self.y / self.a - (1 - self.y) / (1 - self.a)) / m
