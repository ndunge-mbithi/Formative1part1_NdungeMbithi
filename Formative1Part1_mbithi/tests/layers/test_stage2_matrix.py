"""Public Stage 2 checks for ``Linear`` with more than one output neuron.

Chapter 2 adds no code to ``linear.py`` -- ``Linear(n, C)`` already models C
neurons because it is built from ``in_features`` and ``out_features`` and
``self.W`` has shape ``(in_features, out_features)``. These checks confirm that,
in the open. They are a SUBSET; grading also checks per-neuron independence in
detail, the initialisation scale for ``(n, C)``, weight orientation on
non-square layers, vectorisation, and aliasing. See the assignment rubric.

Run from the submission root, with the iml-formative1 environment active:

    pytest tests/layers/test_stage2_matrix.py -v
    ruff check nn/
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(
    2, "Linear with C output neurons", "nn/layers/linear.py", slug="matrix"
)

# Chapter 2 worked example: the three class weight vectors are the COLUMNS of W.
W2 = [[0.5, -0.3, 0.1], [1.0, 0.2, -0.6], [-0.5, 0.4, 0.3]]
B2 = [0.2, -0.1, 0.05]
X2 = [[2.0, -1.0, 3.0]]
Z2 = [[-1.3, 0.3, 1.75]]
X2_BATCH = [[2.0, -1.0, 3.0], [1.0, 0.0, -2.0]]
Z2_BATCH = [[-1.3, 0.3, 1.75], [1.7, -1.2, -0.45]]


@pytest.mark.group("A", "Construction & attributes")
def test_init_shapes_for_multi_output(make_linear):
    layer = make_linear(5, 4)
    assert layer.W.shape == (5, 4)
    assert layer.b.shape == (4,)
    assert np.allclose(layer.b, 0.0)


@pytest.mark.group("C", "Output shape")
def test_forward_shape_multi_output(make_linear):
    out = make_linear(5, 4).forward(np.zeros((8, 5)))
    assert out.shape == (8, 4)


@pytest.mark.group("D", "Value correctness")
def test_forward_worked_example_single(make_linear):
    layer = make_linear(3, 3, W=W2, b=B2)
    assert np.allclose(layer.forward(np.array(X2)), Z2)


@pytest.mark.group("D", "Value correctness")
def test_forward_worked_example_batch(make_linear):
    layer = make_linear(3, 3, W=W2, b=B2)
    assert np.allclose(layer.forward(np.array(X2_BATCH)), Z2_BATCH)


@pytest.mark.group("D", "Value correctness")
def test_single_output_case_still_works(make_linear):
    layer = make_linear(3, 1, W=[[0.5], [1.0], [-0.5]], b=[0.2])
    out = layer.forward(np.array(X2))
    assert out.shape == (1, 1)
    assert np.allclose(out, [[-1.3]])


@pytest.mark.group("E", "Per-neuron independence")
def test_output_column_matches_a_single_neuron(make_linear):
    # Column 0 of W is Chapter 1's neuron; output column 0 must be its logit,
    # whatever the other columns hold.
    w = [[0.5, 9.0, -9.0], [1.0, 9.0, -9.0], [-0.5, 9.0, -9.0]]
    layer = make_linear(3, 3, W=w, b=[0.2, 0.0, 0.0])
    out = layer.forward(np.array(X2))
    assert np.isclose(out[0, 0], -1.3)


@pytest.mark.group("G", "Purity & aliasing")
def test_forward_does_not_mutate_input(make_linear):
    layer = make_linear(3, 3, W=W2, b=B2)
    x = np.array(X2_BATCH)
    before = x.copy()
    layer.forward(x)
    assert np.array_equal(x, before)
