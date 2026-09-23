"""Public Stage 3 checks for ``Linear.backward`` and ``Linear.parameters``.


    pytest tests/layers/test_stage3_linear_backward.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    3, "Linear.backward & parameters", "nn/layers/linear.py", slug="linear_backward"
)

# Chapter 3 worked example (single output neuron, batch of 2).
W3 = [[0.5], [1.0], [-0.5]]
B3 = [0.2]
X3 = [[2.0, -1.0, 3.0], [1.0, 0.0, -2.0]]
G3 = [[0.4], [-0.9]]
DW3 = [[-0.1], [-0.4], [3.0]]
DB3 = [-0.5]
DX3 = [[0.2, 0.4, -0.2], [-0.45, -0.9, 0.45]]


def _numeric_grad(loss_fn, param, eps=1e-6):
    grad = np.zeros_like(param, dtype=float)
    for idx in np.ndindex(param.shape):
        orig = param[idx]
        param[idx] = orig + eps
        hi = loss_fn()
        param[idx] = orig - eps
        lo = loss_fn()
        param[idx] = orig
        grad[idx] = (hi - lo) / (2 * eps)
    return grad


@pytest.mark.group("A", "Gradient shapes")
def test_gradient_shapes(make_linear, rng):
    n, c, m = 4, 3, 6
    layer = make_linear(n, c, W=rng.standard_normal((n, c)), b=rng.standard_normal(c))
    layer.forward(rng.standard_normal((m, n)))
    dx = layer.backward(rng.standard_normal((m, c)))
    assert layer.dW.shape == (n, c)
    assert layer.db.shape == (c,)
    assert dx.shape == (m, n)


@pytest.mark.group("B", "Value correctness")
def test_worked_example(make_linear):
    layer = make_linear(3, 1, W=W3, b=B3)
    layer.forward(np.array(X3))
    dx = layer.backward(np.array(G3))
    assert np.allclose(layer.dW, DW3)
    assert np.allclose(layer.db, DB3)
    assert np.allclose(dx, DX3)


@pytest.mark.group("C", "Gradient check")
def test_gradient_check(make_linear, rng):
    n, c, m = 4, 3, 5
    layer = make_linear(n, c, W=rng.standard_normal((n, c)), b=rng.standard_normal(c))
    x = rng.standard_normal((m, n))
    g = rng.standard_normal((m, c))  # upstream gradient; L = sum(Z * g)

    def loss():
        return float(np.sum(layer.forward(x) * g))

    layer.forward(x)
    dx = layer.backward(g)
    assert np.allclose(layer.dW, _numeric_grad(loss, layer.W), atol=1e-6)
    assert np.allclose(layer.db, _numeric_grad(loss, layer.b), atol=1e-6)
    assert np.allclose(dx, _numeric_grad(loss, x), atol=1e-6)


@pytest.mark.group("E", "parameters()")
def test_parameters_returns_param_grad_pairs(make_linear, rng):
    layer = make_linear(4, 3, W=rng.standard_normal((4, 3)), b=rng.standard_normal(3))
    layer.forward(rng.standard_normal((5, 4)))
    layer.backward(rng.standard_normal((5, 3)))
    params = layer.parameters()
    assert len(params) == 2
    (w, dw), (b, db) = params
    assert w is layer.W and dw is layer.dW
    assert b is layer.b and db is layer.db


@pytest.mark.group("F", "Purity & aliasing")
def test_backward_does_not_mutate_grad_output(make_linear, rng):
    layer = make_linear(4, 3, W=rng.standard_normal((4, 3)), b=rng.standard_normal(3))
    layer.forward(rng.standard_normal((5, 4)))
    g = rng.standard_normal((5, 3))
    before = g.copy()
    layer.backward(g)
    assert np.array_equal(g, before)
