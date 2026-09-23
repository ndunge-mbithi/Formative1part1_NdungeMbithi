"""Public Stage 7 checks for ``Softmax.backward``.

A SUBSET. Grading also runs the gradient check on more shapes, the
row-independence check, and aliasing. A batch loop is allowed here; there is
no no-loop check. See the rubric.

    pytest tests/activations/test_stage7_softmax_backward.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    7, "Softmax.backward", "nn/activations/softmax.py", slug="softmax_backward"
)


@pytest.mark.group("A", "Shape")
def test_backward_preserves_shape(make_softmax, rng):
    sm = make_softmax()
    sm.forward(rng.standard_normal((6, 4)))
    dz = sm.backward(rng.standard_normal((6, 4)))
    assert dz.shape == (6, 4)


@pytest.mark.group("B", "Gradient check")
def test_gradient_check_against_forward(make_softmax, rng, numeric_grad):
    z = rng.standard_normal((5, 4))
    g = rng.standard_normal((5, 4))
    sm = make_softmax()
    sm.forward(z)
    analytic = sm.backward(g)
    num = numeric_grad(lambda: float(np.sum(make_softmax().forward(z) * g)), z)
    assert np.allclose(analytic, num, atol=1e-6)


@pytest.mark.group("C", "a - y shortcut")
def test_chained_with_categorical_cross_entropy_is_a_minus_y(make_softmax):
    from nn.losses import CategoricalCrossEntropyLoss

    z = np.array([[-1.3, 0.3, 1.75]])
    y = np.array([[0.0, 0.0, 1.0]])
    sm = make_softmax()
    a = sm.forward(z)
    cce = CategoricalCrossEntropyLoss()
    cce.forward(a, y)
    dz = sm.backward(cce.backward())
    assert np.allclose(dz, (a - y) / y.shape[0], atol=1e-4)


@pytest.mark.group("E", "Purity & aliasing")
def test_backward_does_not_mutate_grad_output(make_softmax, rng):
    sm = make_softmax()
    sm.forward(rng.standard_normal((4, 3)))
    g = rng.standard_normal((4, 3))
    before = g.copy()
    sm.backward(g)
    assert np.array_equal(g, before)
