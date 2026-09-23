"""Stochastic gradient descent optimizer."""

import numpy as np


class SGD:
    """Vanilla SGD: param -= lr * grad, for every tracked parameter."""

    def __init__(
        self,
        parameters: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Store the parameters to update and the learning rate.

        Args:
            parameters (list[tuple[np.ndarray, np.ndarray]]): (param, grad)
                pairs, as returned by a module's parameters() method.
            lr (float): learning rate, the step size eta.
        """
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        """Apply one update step to every tracked parameter.

        Updates every param in place using its current grad, following
        param -= lr * grad. Does not return anything.
        """
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self) -> None:
        """Reset every tracked parameter's gradient to zero.

        Does not return anything.
        """
        for _, grad in self.parameters:
            grad[...] = 0
