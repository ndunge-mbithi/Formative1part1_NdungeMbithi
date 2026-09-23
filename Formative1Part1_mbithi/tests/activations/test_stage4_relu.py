"""Public Stage 4 checks for ``nn.activations.relu.ReLU``.


    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(4, "ReLU forward & backward", "nn/activations/relu.py")


@pytest.mark.group("A", "Forward")
def test_forward_zeroes_negatives_keeps_positives():
    from nn.activations import ReLU

    out = ReLU().forward(np.array([[-2.0, 0.0, 3.0, -0.1, 5.0]]))
    assert np.allclose(out, [[0.0, 0.0, 3.0, 0.0, 5.0]])


@pytest.mark.group("A", "Forward")
def test_forward_preserves_shape():
    from nn.activations import ReLU

    assert ReLU().forward(np.zeros((4, 6))).shape == (4, 6)


@pytest.mark.group("B", "Backward")
def test_backward_worked_example():
    from nn.activations import ReLU

    relu = ReLU()
    relu.forward(np.array([[-2.0, 0.0, 3.0]]))
    dz = relu.backward(np.array([[0.5, -0.2, 0.1]]))
    assert np.allclose(dz, [[0.0, 0.0, 0.1]])


@pytest.mark.group("C", "Gradient check")
def test_gradient_check(rng, numeric_grad):
    from nn.activations import ReLU

    x = rng.standard_normal((5, 4))
    g = rng.standard_normal((5, 4))
    relu = ReLU()
    relu.forward(x)
    analytic = relu.backward(g)
    num = numeric_grad(lambda: float(np.sum(ReLU().forward(x) * g)), x)
    assert np.allclose(analytic, num, atol=1e-6)


@pytest.mark.group("E", "Purity & aliasing")
def test_forward_does_not_mutate_input():
    from nn.activations import ReLU

    x = np.array([[-2.0, 3.0]])
    before = x.copy()
    ReLU().forward(x)
    assert np.array_equal(x, before)
