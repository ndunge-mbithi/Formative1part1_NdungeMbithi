"""Softmax activation: converts logits into a probability distribution."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in its own
    row, not just the matching input -- see "A shape subtlety" above.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C). Each row sums
            to 1.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        self.a = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Applies the per-example Jacobian ``diag(a) - a @ a.T`` to
        ``grad_output`` in closed form rather than building the Jacobian
        explicitly: for row ``i``, ``dz_i = a_i * (g_i - sum(g_i * a_i))``,
        which is the exact Jacobian-vector product, vectorized across the
        whole batch with no Python loop.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect to
                this layer's output, shape (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
            input (the logits), shape (batch_size, C).
        """
        dot = np.sum(grad_output * self.a, axis=1, keepdims=True)
        return self.a * (grad_output - dot)
