"""Public Stage 1 checks for ``nn.layers.linear.Linear``.

These cover the parts of the Stage 1 rubric that are verified in the open:
construction (parameter shapes and dtypes, zero bias, non-trivial weights) and
the forward pass (output shape and value on the worked examples). They are a
development aid and a SUBSET. Grading runs additional checks that are not
distributed -- exact output shape (a one-row input gives a (1, 1) array, not a
bare float), weight-initialisation scale, weight orientation on non-square
layers, vectorisation (no Python loops), aliasing, and numerical stability.
Passing this file is necessary, not sufficient; see the assignment rubric for
the full list of Stage 1 criteria.

Run from the submission root, with the iml-formative1 environment active:

    pytest tests/layers/test_stage1_vector.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(1, "Linear: __init__ & forward", "nn/layers/linear.py")

WORKED_W = [[0.5], [1.0], [-0.5]]
WORKED_B = [0.2]
WORKED_X = [[2.0, -1.0, 3.0]]
WORKED_Z = [[-1.3]]


@pytest.mark.group("A", "Construction & attributes")
def test_init_parameter_shapes(make_linear):
    layer = make_linear(4, 3)
    assert layer.W.shape == (4, 3)
    assert layer.b.shape == (3,)


@pytest.mark.group("A", "Construction & attributes")
def test_init_parameter_dtypes_are_float(make_linear):
    layer = make_linear(4, 3)
    assert np.issubdtype(layer.W.dtype, np.floating)
    assert np.issubdtype(layer.b.dtype, np.floating)


@pytest.mark.group("A", "Construction & attributes")
def test_init_bias_is_zeros(make_linear):
    assert np.allclose(make_linear(4, 3).b, 0.0)


@pytest.mark.group("A", "Construction & attributes")
def test_init_weights_are_random_not_constant(make_linear):
    w = make_linear(64, 32).W
    assert not np.allclose(w, 0.0), "W must not be all zeros"
    assert w.std() > 0.0, "W must be randomly initialised, not a single constant"


@pytest.mark.group("A", "Construction & attributes")
def test_linear_is_importable_from_package():
    from nn.layers import Linear  # noqa: F401  -- Step 4: the re-export line


@pytest.mark.group("C", "Output shape")
def test_forward_shape_batch_single_neuron(make_linear):
    assert make_linear(4, 1).forward(np.zeros((8, 4))).shape == (8, 1)


@pytest.mark.group("C", "Output shape")
def test_forward_shape_batch_multi_output(make_linear):
    assert make_linear(4, 3).forward(np.zeros((8, 4))).shape == (8, 3)


@pytest.mark.group("D", "Value correctness")
def test_forward_worked_example_single(make_linear):
    layer = make_linear(3, 1, W=WORKED_W, b=WORKED_B)
    out = layer.forward(np.array(WORKED_X))
    assert np.allclose(out, WORKED_Z)


@pytest.mark.group("D", "Value correctness")
def test_forward_worked_example_batch(make_linear):
    layer = make_linear(3, 1, W=WORKED_W, b=WORKED_B)
    x = np.array([[2.0, -1.0, 3.0], [1.0, 0.0, -2.0]])
    assert np.allclose(layer.forward(x), [[-1.3], [1.7]])


@pytest.mark.group("F", "Purity & aliasing")
def test_forward_does_not_mutate_input(make_linear):
    layer = make_linear(3, 1, W=WORKED_W, b=WORKED_B)
    x = np.array(WORKED_X)
    x_before = x.copy()
    layer.forward(x)
    assert np.array_equal(x, x_before), "forward must not modify its input array"
