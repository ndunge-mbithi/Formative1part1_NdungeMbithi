"""Sigmoid activation: squashes real values into (0, 1)."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Uses a branch-per-sign implementation rather than a single
        ``1 / (1 + exp(-x))`` expression: computing ``exp`` on the whole
        array first (as a plain ``np.where`` would) evaluates ``exp(-x)``
        even for large negative x, overflowing before the unused branch is
        discarded. Selecting each element's safe formula via boolean
        indexing means ``exp`` only ever sees non-positive exponents.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.
        """
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_neg = np.exp(x[neg])
        out[neg] = exp_neg / (1.0 + exp_neg)
        self.a = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect to
                this layer's output, same shape as the original input to
                forward.

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
            input, same shape as grad_output.
        """
        return grad_output * self.a * (1.0 - self.a)
