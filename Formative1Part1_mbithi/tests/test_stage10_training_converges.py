"""Public Stage 10 checks for ``main.py`` -- the whole pipeline trained.

This is the only test that does not exercise one file in isolation: it runs
main.train() end to end on the provided toy dataset and asserts the loss
decreases and the model reaches the accuracy threshold.

    pytest tests/test_stage10_training_converges.py -v
    ruff check nn/
    ruff check main.py
"""

import pytest

pytestmark = pytest.mark.stage(10, "Training loop converges", "main.py")


@pytest.mark.group("A", "Interface")
def test_required_symbols_exist():
    import main

    assert callable(main.toy_data)
    assert callable(main.train)
    assert callable(main.accuracy)
    x, y = main.toy_data()
    assert x.shape == (4, 2)
    assert y.shape == (4, 1)


@pytest.mark.group("B", "Loss decreases")
def test_train_returns_decreasing_loss_history():
    import main

    history = main.train(epochs=4000, lr=1.0, seed=0)
    assert isinstance(history, list)
    assert len(history) == 4000
    assert history[-1] < history[0]
    assert history[-1] < 0.1


@pytest.mark.group("C", "Accuracy")
def test_final_accuracy_crosses_threshold():
    import main

    main.train(epochs=4000, lr=1.0, seed=0)
    assert main.accuracy() >= 0.99
