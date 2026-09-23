# Building a Neural Network from Scratch in NumPy

A single-layer neural network library implemented in NumPy only: no PyTorch,
TensorFlow, JAX, or automatic differentiation. Every forward pass, backward
pass, and parameter update is derived and implemented by hand. Read
`guide.pdf` for the full chapter-by-chapter derivation this implementation
follows.

## What's here

```
nn/
  module.py                              base Module contract (forward/backward/parameters/zero_grad)
  layers/linear.py                       z = xW + b, Xavier-initialized, with backward and parameters()
  activations/relu.py                    ReLU, elementwise mask
  activations/sigmoid.py                 numerically stable sigmoid, backward reuses a(1-a)
  activations/softmax.py                 row-wise softmax; backward is the exact Jacobian-vector product
  losses/cross_entropy_loss.py           binary cross-entropy (not a Module: no parameters, backward takes no argument)
  losses/categorical_cross_entropy_loss.py   multiclass cross-entropy over one-hot targets
  optim/sgd.py                           param -= lr * grad, in place, for every tracked (param, grad) pair
main.py                                  wires Linear + Sigmoid + CrossEntropyLoss + SGD, trains on toy AND-gate data
```

`Module` defines the shared interface. Every layer and activation subclasses
it and overrides `forward`/`backward`; activations have no learnable weights,
so they rely on `Module`'s default (empty) `parameters()`. The two loss
classes deliberately do **not** subclass `Module` — their `forward` takes
`(predictions, targets)` rather than a single input, and their `backward`
takes no argument, since a loss is where the gradient chain starts rather
than something with an upstream gradient flowing into it. `SGD` isn't a
`Module` either: it only reads gradients other modules already computed and
updates parameters in place.

## Mathematical components

- **Linear**: `Y = XW + b`. Backward: `dW = X^T G`, `db = sum(G, axis=0)`,
  `dX = G W^T`, where `G` is the upstream gradient. `dW`/`db` are batch
  sums (shared parameters, gradients accumulate across the batch); `dX` is
  per-example (each row of `X` gets its own gradient row back). Weights are
  Xavier/Glorot-uniform initialized — `U(-b, b)` with
  `b = sqrt(6 / (in_features + out_features))` — because an all-zero (or
  degenerate) initialization makes every output neuron compute the same
  thing and receive the same gradient, so they never differentiate during
  training (the symmetry problem). Bias is zero-initialized; only weights
  need symmetry-breaking. `dW`/`db` are pre-allocated as `np.zeros_like(...)`
  in `__init__` and written in place (`self.dW[...] = ...`) rather than
  reassigned, so an optimizer that captured a reference to them via
  `parameters()` keeps seeing updates rather than a stale first-epoch array.

- **ReLU**: `max(0, x)` forward; backward passes the upstream gradient
  through unchanged where the forward input was strictly positive, and
  blocks it (multiplies by 0) everywhere else, including exactly at 0.

- **Sigmoid**: `1 / (1 + e^-x)`, implemented as two branches selected by
  boolean indexing (`x >= 0` uses `1/(1+e^-x)`, `x < 0` uses
  `e^x/(1+e^x)`) rather than a single expression evaluated on the whole
  array. A naive `np.where` still evaluates *both* branches on every
  element before selecting between them, so `exp(-x)` would overflow for
  very negative `x` even though that branch is discarded — boolean
  indexing only ever computes `exp` on the subset where its argument is
  safely non-positive. Backward reuses the forward output: `da/dx = a(1-a)`.

- **Softmax**: row-wise `e^(z - max(z)) / sum(e^(z - max(z)))`, with the
  max-subtraction purely for numerical stability (it cancels algebraically,
  so the result is unchanged, but every exponent becomes ≤ 0, so it can't
  overflow). Unlike the other activations, softmax's backward is not a
  simple elementwise multiply: every output in a row depends on every logit
  in that row through the shared denominator, so its derivative is a full
  Jacobian per example, `diag(a) - a a^T`. Rather than building that
  Jacobian explicitly (or looping over the batch), `backward` uses the
  closed-form Jacobian-vector product
  `dz = a * (g - sum(g * a, axis=1, keepdims=True))`, vectorized across the
  whole batch with no Python loop — this is the exact derivative, not an
  approximation, and chaining it with `CategoricalCrossEntropyLoss`
  collapses to the expected `a - y`.

- **Binary cross-entropy**: `-mean(y*log(a) + (1-y)*log(1-a))`, predictions
  clipped to `[1e-12, 1-1e-12]` before any `log` or division, in both
  `forward` and `backward`, so a prediction that rounds to exactly 0 or 1
  can't produce `-inf`/`nan`. Chained with `Sigmoid.backward`, this
  collapses to the `a - y` shortcut.

- **Categorical cross-entropy**: `-sum(y * log(a)) / m` over one-hot
  targets, same clipping discipline, `backward` returns `-y / (m*a)`.
  Chained with `Softmax.backward`, this also collapses to `a - y`.

- **SGD**: `param -= lr * grad` for every `(param, grad)` pair passed in.
  `step()` mutates the parameter arrays in place (`-=`, not reassignment),
  so it keeps working correctly against whatever the caller's own
  references point at. `zero_grad()` zeroes each gradient array in place
  the same way.

## Environment

```bash
conda env create -f environment.yml
conda activate iml-formative1
```

Python 3.11, NumPy 2.x, pytest 8.x, ruff — no deep-learning framework.

## Running the tests

From this directory, with the environment active:

```bash
pytest                # all public checks (tests/ only, per pyproject.toml)
ruff check nn/         # documentation + style for the library
ruff check main.py     # documentation + style for the training script
```

`pytest` reports one grouped block per stage (via the `conftest.py` fixtures
and markers) and writes `stage<N>_report.json` files. As of this submission,
**all 62 public tests pass across Stages 1-10, and both `ruff check` commands
exit 0.**

## Running the training example

```bash
python main.py
```

This trains `Linear(2, 1)` → `Sigmoid` → `CrossEntropyLoss`, optimized with
plain SGD, on the AND-gate toy dataset (`main.toy_data()`) for 4000 epochs at
`lr=1.0`. AND is used instead of the more common XOR toy example specifically
*because* this network is a single linear layer throughout — mathematically a
logistic regression, which cannot represent XOR's non-linearly-separable
boundary no matter how it's trained, but can represent AND's. A real training
run:

```
epoch     1  loss = 0.722179
epoch  1000  loss = 0.017387
epoch  2000  loss = 0.008664
epoch  3000  loss = 0.005757
epoch  4000  loss = 0.004308
final accuracy: 100.00%
```

`train(epochs=4000, lr=1.0, seed=0)` seeds NumPy's global random state before
constructing the layer, so a fixed seed reproduces an identical loss history
run to run; `accuracy()` reads back the model `train()` left in module-level
state (or trains once with defaults if called first) and reports the fraction
of the four AND-gate examples correctly classified.

## How correctness was validated

Beyond the public test suite (gradient checks against central-difference
numerical gradients, worked-value checks against the guide's hand-computed
examples, shape checks, aliasing/mutation checks, and the numerical-stability
cases for sigmoid/softmax/both losses), I additionally:

- Confirmed the `Sigmoid`+`CrossEntropyLoss` and `Softmax`+
  `CategoricalCrossEntropyLoss` gradient shortcuts both collapse to `a - y`
  by direct computation, not just by the provided shortcut tests.
- Ran `main.train()` across several seeds (0, 1, 2, 42) and confirmed loss
  decreases monotonically toward the same final value and accuracy reaches
  100% in every case, not just the seed the public test happens to check.
- Confirmed `train(seed=0)` called twice in the same process produces an
  identical loss history, verifying the reproducibility requirement rather
  than assuming NumPy's global seeding does what's expected.
- Grepped the whole `nn/` package and `main.py` for `torch`/`tensorflow`/
  `keras`/`jax`/`sklearn` to confirm no forbidden dependency crept in.
