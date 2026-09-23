"""Base class defining the shared interface for every layer, activation, and loss."""


class Module:
    """Base class every layer and activation subclasses.

    Defines the forward/backward contract, plus default (empty) parameter
    and gradient-reset behavior for modules that have no learnable weights.
    """

    def forward(self, x):
        """Compute this module's output given input x.

        Args:
            x: input array.

        Returns:
            The module's output for this input.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """Compute gradients given the upstream gradient.

        Args:
            grad_output: gradient of the loss with respect to this module's
                output.

        Returns:
            Gradient of the loss with respect to this module's input.
        """
        raise NotImplementedError

    def parameters(self):
        """Return this module's learnable parameters.

        Returns:
            A list of (param, grad) pairs. Empty for modules with no
            learnable weights.
        """
        return []

    def zero_grad(self):
        """Reset any stored gradients to zero.

        No-op by default. In this project, resetting stored gradients is
        handled by the optimizer (Chapter 9) directly, not by individual
        modules -- you will not need to override this method anywhere,
        including in Linear.
        """
        pass
