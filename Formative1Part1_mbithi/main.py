"""Chapter 10 training pipeline: Linear + Sigmoid + CrossEntropyLoss + SGD.

Wires the pieces built in earlier chapters into a trained model on a toy
dataset, as the first end-to-end check of the whole pipeline before it is
pointed at real competition data.
"""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

# Populated by train(); accuracy() reads these back without needing to
# retrain, as long as train() has run at least once in this process.
_layer = None
_activation = None
_toy_X = None
_toy_y = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the AND-gate toy dataset.

    AND is linearly separable (unlike XOR), which matters here because this
    project's network is a single Linear layer plus one activation -- a
    linear classifier, whatever its initialization or training. See
    Chapter 10, "What a healthy run looks like".

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y). X has shape (4, 2), the four
        binary input pairs. y has shape (4, 1), the AND of each pair.
    """
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [0.0], [0.0], [1.0]])
    return x, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train Linear -> Sigmoid on the toy AND-gate dataset with SGD.

    Seeds NumPy's global random state before constructing the layer so
    that a fixed seed gives a fixed (reproducible) weight initialization
    and therefore a fixed loss history. The trained layer and activation
    are stored at module level so accuracy() can evaluate them afterward
    without repeating training.

    Args:
        epochs (int): number of full-batch gradient-descent steps.
        lr (float): learning rate passed to SGD.
        seed (int): seed for NumPy's global random state, applied before
            the layer's weights are initialized.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    global _layer, _activation, _toy_X, _toy_y

    np.random.seed(seed)

    x, y = toy_data()
    layer = Linear(in_features=2, out_features=1)
    activation = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(layer.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        z = layer.forward(x)
        a = activation.forward(z)
        loss = loss_fn.forward(a, y)
        history.append(loss)

        grad_a = loss_fn.backward()
        grad_z = activation.backward(grad_a)
        layer.backward(grad_z)

        optimizer.step()
        optimizer.zero_grad()

    _layer, _activation, _toy_X, _toy_y = layer, activation, x, y
    return history


def accuracy(loss_history: list[float] = None) -> float:
    """Report classification accuracy of the most recently trained model.

    ``loss_history`` is accepted for interface compatibility (e.g. a
    caller that already has a loss history on hand) but is not needed to
    compute accuracy, which depends only on the trained model's
    predictions, not the loss values recorded along the way. If train()
    has not been called yet in this process, it is run once with its
    defaults first.

    Args:
        loss_history (list[float]): unused; accepted for interface
            compatibility with callers that pass train()'s return value.

    Returns:
        float: fraction of the toy dataset's four examples correctly
        classified, in [0, 1].
    """
    if _layer is None:
        train()

    predictions = _activation.forward(_layer.forward(_toy_X))
    predicted_labels = (predictions >= 0.5).astype(float)
    return float(np.mean(predicted_labels == _toy_y))


if __name__ == "__main__":
    loss_history = train(epochs=4000, lr=1.0, seed=0)
    for epoch in (0, 999, 1999, 2999, 3999):
        print(f"epoch {epoch + 1:5d}  loss = {loss_history[epoch]:.6f}")
    print(f"final accuracy: {accuracy():.2%}")
