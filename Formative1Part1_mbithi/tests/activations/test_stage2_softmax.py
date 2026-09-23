"""Public Stage 2 checks for ``nn.activations.softmax.Softmax`` (forward only).

A SUBSET. Grading also checks the per-row denominator on a mixed-scale batch,
shift-invariance, ordering, an exact reference on random logits, aliasing, and
vectorisation. ``Softmax.backward`` is a later chapter and is not checked here.
See the assignment rubric.

Run from the submission root, with the iml-formative1 environment active:

    pytest tests/activations/test_stage2_softmax.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    2, "Softmax.forward", "nn/activations/softmax.py", slug="softmax"
)

WORKED_Z = [[-1.3, 0.3, 1.75]]
WORKED_P = [[0.0369, 0.1830, 0.7801]]


@pytest.mark.group("A", "Module contract")
def test_softmax_importable_from_package():
    from nn.activations import Softmax  # noqa: F401  -- Step 2 re-export


@pytest.mark.group("B", "Output shape")
def test_forward_preserves_shape(make_softmax):
    out = make_softmax().forward(np.zeros((6, 4)))
    assert out.shape == (6, 4)


@pytest.mark.group("C", "Value correctness")
def test_forward_worked_example(make_softmax):
    out = make_softmax().forward(np.array(WORKED_Z))
    # matches the guide's printed 4-decimal values
    assert np.allclose(np.round(out, 4), WORKED_P)


@pytest.mark.group("D", "Normalisation")
def test_rows_sum_to_one(make_softmax, rng):
    out = make_softmax().forward(rng.standard_normal((7, 5)))
    assert np.allclose(out.sum(axis=1), 1.0)


@pytest.mark.group("D", "Normalisation")
def test_entries_in_unit_interval(make_softmax, rng):
    out = make_softmax().forward(rng.standard_normal((7, 5)))
    assert np.all(out > 0.0)
    assert np.all(out < 1.0)


@pytest.mark.group("E", "Numerical stability")
def test_large_logit_does_not_overflow(make_softmax):
    out = make_softmax().forward(np.array([[1000.0, 0.0, -3.0]]))
    assert np.all(np.isfinite(out))
    assert np.isclose(out.sum(), 1.0)


@pytest.mark.group("G", "Purity & aliasing")
def test_forward_does_not_mutate_input(make_softmax):
    z = np.array([[-1.3, 0.3, 1.75]])
    before = z.copy()
    make_softmax().forward(z)
    assert np.array_equal(z, before)
