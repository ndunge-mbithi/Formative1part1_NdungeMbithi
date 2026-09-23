"""Linear (fully connected) layer: z = xW + b."""

import numpy as np

from nn.module import Module


class Linear(Module):
    """A fully connected layer computing z = xW + b.

    Attributes:
        W (np.ndarray): weight matrix, shape (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer's weights, bias, and gradient buffers.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output neurons.

        Sets:
            self.W (np.ndarray): weight matrix, shape
                (in_features, out_features). Xavier-initialized, not zeros
                (see "Weight initialization", Chapter 1).
            self.b (np.ndarray): bias vector, shape (out_features,).
                Initialized to zero.
            self.dW (np.ndarray): gradient buffer for self.W, same shape as
                self.W. Pre-allocated to zeros here so backward can write
                into it in place -- see "Why backward must write in place"
                above.
            self.db (np.ndarray): gradient buffer for self.b, same shape as
                self.b, pre-allocated the same way.
        """
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(
            -bound, bound, size=(in_features, out_features)
        )
        self.b = np.zeros(out_features, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this layer's output for a batch of inputs.

        Args:
            x (np.ndarray): input, shape (batch_size, in_features).

        Returns:
            np.ndarray: output, shape (batch_size, out_features).

        Sets:
            self.x (np.ndarray): the input, saved unchanged so backward can
                use it.
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect to
                this layer's output, shape (m, C).

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
            input, shape (m, n).

        Sets:
            self.dW (np.ndarray): overwritten in place
                (self.dW[...] = ...), not reassigned -- see "Why backward
                must write in place" above.
            self.db (np.ndarray): overwritten in place the same way.
        """
        self.dW[...] = self.x.T @ grad_output
        self.db[...] = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return this layer's learnable parameters.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: pairs of (parameter,
            gradient) -- [(self.W, self.dW), (self.b, self.db)].
        """
        return [(self.W, self.dW), (self.b, self.db)]
