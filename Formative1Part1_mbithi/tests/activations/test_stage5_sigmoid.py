"""Public Stage 5 checks for ``nn.activations.sigmoid.Sigmoid``.


    pytest tests/activations/test_stage5_sigmoid.py -v
    ruff check nn/
"""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    5, "Sigmoid forward & backward", "nn/activations/sigmoid.py"
)


def _sigmoid():
    from nn.activations import Sigmoid

    return Sigmoid()


@pytest.mark.group("A", "Forward")
def test_forward_worked_value():
    out = _sigmoid().forward(np.array([[-1.3]]))
    assert np.allclose(out, 0.2142, atol=1e-4)


@pytest.mark.group("A", "Forward")
def test_forward_range_and_shape(rng):
    out = _sigmoid().forward(rng.standard_normal((5, 4)))
    assert out.shape == (5, 4)
    assert np.all(out > 0.0)
    assert np.all(out < 1.0)


@pytest.mark.group("B", "Backward")
def test_backward_worked_value():
    sig = _sigmoid()
    sig.forward(np.array([[-1.3]]))
    assert np.allclose(sig.backward(np.array([[1.0]])), 0.1683, atol=1e-4)


@pytest.mark.group("C", "Gradient check")
def test_gradient_check(rng, numeric_grad):
    x = rng.standard_normal((5, 4))
    g = rng.standard_normal((5, 4))
    sig = _sigmoid()
    sig.forward(x)
    analytic = sig.backward(g)
    num = numeric_grad(lambda: float(np.sum(_sigmoid().forward(x) * g)), x)
    assert np.allclose(analytic, num, atol=1e-6)


@pytest.mark.group("D", "Numerical stability")
def test_large_magnitude_does_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # overflow -> failure
        out = _sigmoid().forward(np.array([[1000.0, -1000.0, 50.0, -50.0]]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.group("E", "Purity & aliasing")
def test_forward_does_not_mutate_input():
    x = np.array([[-1.3, 2.0]])
    before = x.copy()
    _sigmoid().forward(x)
    assert np.array_equal(x, before)
