"""Public Stage 8 checks for ``CategoricalCrossEntropyLoss``.


    pytest tests/losses/test_stage8_categorical_cross_entropy.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    8,
    "Categorical cross-entropy loss",
    "nn/losses/categorical_cross_entropy_loss.py",
)


def _cce():
    from nn.losses import CategoricalCrossEntropyLoss

    return CategoricalCrossEntropyLoss()


@pytest.mark.group("A", "Forward")
def test_forward_worked_value():
    a = np.array([[0.0369, 0.1830, 0.7801]])
    y = np.array([[0.0, 0.0, 1.0]])
    loss = _cce().forward(a, y)
    assert isinstance(loss, float)
    assert loss == pytest.approx(-np.log(0.7801), rel=1e-4)


@pytest.mark.group("A", "Forward")
def test_forward_is_batch_average(rng):
    a = rng.dirichlet(np.ones(4), size=6)
    y = np.eye(4)[rng.integers(0, 4, size=6)]
    expected = -np.sum(y * np.log(a)) / 6
    assert _cce().forward(a, y) == pytest.approx(expected)


@pytest.mark.group("B", "Backward")
def test_backward_worked_value():
    a = np.array([[0.0369, 0.1830, 0.7801]])
    y = np.array([[0.0, 0.0, 1.0]])
    cce = _cce()
    cce.forward(a, y)
    grad = cce.backward()
    assert grad.shape == (1, 3)
    assert np.allclose(grad, -(y / a), atol=1e-6)


@pytest.mark.group("C", "Gradient check")
def test_gradient_check(rng, numeric_grad):
    a = rng.dirichlet(np.ones(4), size=5)
    y = np.eye(4)[rng.integers(0, 4, size=5)]
    cce = _cce()
    cce.forward(a, y)
    analytic = cce.backward()
    num = numeric_grad(lambda: _cce().forward(a, y), a)
    assert np.allclose(analytic, num, atol=1e-5)


@pytest.mark.group("E", "Signature")
def test_not_a_module_and_backward_takes_no_argument():
    from nn.losses import CategoricalCrossEntropyLoss
    from nn.module import Module

    assert not issubclass(CategoricalCrossEntropyLoss, Module)
    cce = CategoricalCrossEntropyLoss()
    cce.forward(np.array([[0.3, 0.7]]), np.array([[0.0, 1.0]]))
    cce.backward()
