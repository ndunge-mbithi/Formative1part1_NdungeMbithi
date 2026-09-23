"""Public Stage 6 checks for ``nn.losses.cross_entropy_loss.CrossEntropyLoss``.

A SUBSET. Grading also runs a full gradient check on dL/da and the a - y
shortcut chained through Sigmoid.backward. See the rubric.

    pytest tests/losses/test_stage6_cross_entropy.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    6, "Binary cross-entropy loss", "nn/losses/cross_entropy_loss.py"
)


def _bce():
    from nn.losses import CrossEntropyLoss

    return CrossEntropyLoss()


@pytest.mark.group("A", "Forward")
def test_forward_worked_value():
    loss = _bce().forward(np.array([[0.2142]]), np.array([[1.0]]))
    assert isinstance(loss, float)
    assert loss == pytest.approx(-np.log(0.2142), rel=1e-6)


@pytest.mark.group("A", "Forward")
def test_forward_is_batch_average(rng):
    a = rng.uniform(0.1, 0.9, size=(8, 1))
    y = rng.integers(0, 2, size=(8, 1)).astype(float)
    expected = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
    assert _bce().forward(a, y) == pytest.approx(expected)


@pytest.mark.group("B", "Backward")
def test_backward_worked_value():
    bce = _bce()
    bce.forward(np.array([[0.2142]]), np.array([[1.0]]))
    grad = bce.backward()
    assert grad.shape == (1, 1)
    assert grad[0, 0] == pytest.approx(-1.0 / 0.2142, rel=1e-6)


@pytest.mark.group("C", "Gradient check")
def test_gradient_check(rng, numeric_grad):
    a = rng.uniform(0.05, 0.95, size=(6, 1))
    y = rng.integers(0, 2, size=(6, 1)).astype(float)
    bce = _bce()
    bce.forward(a, y)
    analytic = bce.backward()
    num = numeric_grad(lambda: _bce().forward(a, y), a)
    assert np.allclose(analytic, num, atol=1e-6)


@pytest.mark.group("E", "Signature")
def test_not_a_module_and_backward_takes_no_argument():
    from nn.losses import CrossEntropyLoss
    from nn.module import Module

    assert not issubclass(CrossEntropyLoss, Module)
    bce = CrossEntropyLoss()
    bce.forward(np.array([[0.5]]), np.array([[1.0]]))
    bce.backward()  # no grad_output argument
