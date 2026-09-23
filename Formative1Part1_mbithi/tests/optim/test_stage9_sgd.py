"""Public Stage 9 checks for ``nn.optim.sgd.SGD``.

A SUBSET. Grading also checks in-place updates, a persistent optimizer across
training steps, and updating a real Linear layer. See the rubric.

    pytest tests/optim/test_stage9_sgd.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(9, "SGD step & zero_grad", "nn/optim/sgd.py")


def _sgd(pairs, lr):
    from nn.optim import SGD

    return SGD(pairs, lr)


@pytest.mark.group("A", "step()")
def test_step_moves_param_opposite_gradient():
    w = np.array([1.0, 2.0, 3.0])
    g = np.array([0.1, -0.4, 0.2])
    _sgd([(w, g)], lr=0.5).step()
    assert np.allclose(w, [1.0 - 0.05, 2.0 + 0.2, 3.0 - 0.1])


@pytest.mark.group("A", "step()")
def test_step_uses_the_learning_rate():
    w = np.zeros(2)
    g = np.array([1.0, 1.0])
    _sgd([(w, g)], lr=0.25).step()
    assert np.allclose(w, [-0.25, -0.25])


@pytest.mark.group("B", "step() in place")
def test_step_does_not_rebind_the_param():
    w = np.array([5.0, 5.0])
    same = w
    _sgd([(w, np.ones(2))], lr=1.0).step()
    assert same is w
    assert np.allclose(w, [4.0, 4.0])


@pytest.mark.group("C", "zero_grad()")
def test_zero_grad_zeroes_every_gradient():
    g1 = np.array([1.0, 2.0])
    g2 = np.array([[3.0], [4.0]])
    _sgd([(np.zeros(2), g1), (np.zeros((2, 1)), g2)], lr=0.1).zero_grad()
    assert np.allclose(g1, 0.0)
    assert np.allclose(g2, 0.0)


@pytest.mark.group("E", "Signature")
def test_sgd_is_not_a_module():
    from nn.module import Module
    from nn.optim import SGD

    assert not issubclass(SGD, Module)
