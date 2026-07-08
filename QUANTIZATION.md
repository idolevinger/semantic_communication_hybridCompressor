# How Quantization Works

This document explains the **constellation quantizer** — the stage that turns the
autoencoder's continuous bottleneck vector into discrete symbols that can be
"transmitted" over the simulated channel. All the code lives in
[quantizer.py](quantizer.py); it is wired into the pipeline in
[pipeline.py](pipeline.py) and driven during training by [train.py](train.py).

---

## 1. Why we need a quantizer at all

The autoencoder encoder outputs a continuous, real-valued vector `z` of size
`BOTTLENECK_DIM` (e.g. 128). A real communication system cannot send arbitrary
real numbers — it sends **symbols** drawn from a finite, agreed-upon set of points
called a **constellation** (the classic QAM grid). The quantizer's job is to snap
`z` onto that finite grid.

The hard part: snapping to the nearest grid point is a *step function* — it has
zero gradient almost everywhere, so you cannot train an encoder through it with
backpropagation. The quantizer solves this with a **soft-to-hard** scheme that
sends hard symbols forward but lets a smooth gradient flow backward.

---

## 2. The constellation (the grid of legal symbols)

A square **M-QAM** constellation is built by
[`build_square_qam()`](quantizer.py#L30):

- `M = QAM_ORDER` must be a perfect square (4, 16, 64, 256, …).
- With `M = 16`, each axis has `√16 = 4` levels: `[-3, -1, 1, 3]`.
- Taking every (I, Q) combination gives a 4×4 grid of 16 points — a complex plane
  of allowed symbols. Each point carries `log₂(M)` bits (4 bits for 16-QAM).
- The whole grid is then scaled so the **average symbol power** equals `QAM_POWER`
  (default 1.0). Fixing the power is what makes the channel SNR well-defined.

```
Q
 3 │  •     •     •     •
 1 │  •     •     •     •
-1 │  •     •     •     •
-3 │  •     •     •     •
   └─────────────────────  I
     -3    -1    1     3      (16-QAM, before power normalization)
```

By default the grid is **fixed** (stored as a non-trainable buffer). Setting
`LEARNED_CONSTELLATION = True` makes the 16 points trainable parameters that the
optimizer can move around; they are re-normalized back to `QAM_POWER` after every
forward pass so the power budget never drifts.

---

## 3. Real vector → I/Q symbols

A constellation point is a 2-D coordinate `(I, Q)`. So the quantizer treats each
**consecutive pair** of real bottleneck values as one complex symbol:

```
z = [z0, z1, z2, z3, ... , z126, z127]      (BOTTLENECK_DIM = 128 reals)
      └──┘   └──┘              └─────┘
     sym 0  sym 1             sym 63         → 64 I/Q symbols
```

This is the `z.view(batch, num_symbols, 2)` reshape at
[quantizer.py:117](quantizer.py#L117), where `num_symbols = BOTTLENECK_DIM / 2`.
(The bottleneck dim must therefore be even.)

---

## 4. The core trick: soft assignment + hard forward

For each symbol the quantizer computes the **squared distance to every one of the
M constellation points** ([quantizer.py:122-124](quantizer.py#L122)):

```
dist2[b, s, m] = || symbol(b,s) − constellation(m) ||²
```

It then produces two different assignments from those distances.

### Hard assignment (what is actually transmitted)

Pick the single nearest point — an `argmin` over the M points:

```python
idx  = dist2.argmin(dim=-1)      # nearest point index per symbol
hard = self.constellation[idx]   # its (I, Q) coordinates
```

This is exact, discrete, and non-differentiable — a real transmitted symbol.

### Soft assignment (what carries the gradient)

Turn the distances into a probability-like weighting with a temperature-scaled
softmax, then take the **weighted average** of all constellation points:

```python
weights = softmax(−sigma_q · dist2)   # peaks on the closest points
soft    = weights @ constellation     # smooth, differentiable blend
```

`soft` is a smooth function of the input `z`, so gradients flow through it.

### Straight-through estimator (glue)

The forward output uses `hard`, but the gradient is taken from `soft`
([quantizer.py:135](quantizer.py#L135)):

```python
out = hard + (soft - soft.detach())
```

- **Forward:** `soft - soft.detach() == 0`, so `out == hard` — exact grid points
  are transmitted.
- **Backward:** `hard` and `soft.detach()` have no gradient, so
  `d(out)/dz == d(soft)/dz` — the encoder is trained *as if* the smooth soft
  assignment had been used.

This is the **straight-through estimator**: hard where it matters (the channel),
soft where it matters (the gradient).

---

## 5. `sigma_q`: the hardness knob, annealed during training

`sigma_q` is the softmax temperature that controls how *peaked* the soft
assignment is:

- **Small `sigma_q`** → weights are spread across many nearby points → `soft` is a
  soft blend, far from any single grid point. Smooth gradients, but soft and hard
  disagree.
- **Large `sigma_q`** → the softmax collapses onto the single nearest point →
  `soft ≈ hard`. The training-time behavior matches the deployed hard behavior.

We want smooth gradients *early* (to train freely) and hard-like behavior *late*
(so training matches deployment). So `sigma_q` is **annealed upward** once per
optimizer step by [`step_sigma()`](quantizer.py#L92), called from the training
loop at [train.py:111](train.py#L111):

```
sigma_q = min(SOFT_Q_MAX, SOFT_Q_INIT + SOFT_Q_ANNEAL_RATE · step_count)
```

With the defaults (`INIT = 5.0`, `MAX = 100.0`, `ANNEAL_RATE = 5e-3`) it climbs
from 5 to its cap of 100 over roughly 19k optimizer steps — well within a normal
training run. You can watch it print each epoch as `sigma_q` in the training log.

---

## 6. The KL regularizer: use the whole constellation

Left alone, the encoder could cheat by mapping everything onto just two or three
constellation points ("mode collapse"), wasting most of the grid and most of the
bit budget. To prevent that, the quantizer tracks an **empirical usage
distribution** `P̂(C)` — the average soft weight each point received on the last
batch ([quantizer.py:138](quantizer.py#L138)):

```python
self.last_usage = weights.mean over (batch, symbols)   # shape (M,)
```

[`kl_to_uniform()`](quantizer.py#L153) then measures how far that usage is from a
**uniform** distribution over all M points:

```
KL( P̂(C) || Uniform )
```

Phase-1 training adds this term to the loss ([train.py:105-106](train.py#L105)):

```
L₁ = MSE(reconstruction, original) + KL_LAMBDA · KL(usage || uniform)
```

`KL_LAMBDA` (default `0.05`) sets how hard we push toward uniform usage. The KL is
minimized when every constellation point is used roughly equally, which keeps the
code from collapsing and maximizes the information each symbol carries. For very
large constellations (M ≥ 4096) it can be set to 0.

---

## 7. Where the quantizer sits in the pipeline

**Training / eval without LDPC** — the differentiable path
([pipeline.py:70](pipeline.py#L70), [pipeline.py:79](pipeline.py#L79)):

```
z = encoder(x)          # 768 → bottleneck, L2-normalized
z = quantizer(z)        # snap to constellation (hard fwd, soft grad)
z = channel(z)          # + AWGN
x̂ = decoder(z)          # reconstruct 768-D
```

**Inference with LDPC** — the discrete path
([pipeline.py:56-67](pipeline.py#L56)). Here we do not need a gradient, so instead
of the straight-through `forward()` the pipeline calls
[`hard_quantize_indices()`](quantizer.py#L142) to get the integer symbol indices
directly, and hands those to the LDPC codec for bit-level error correction. Note
the LDPC path is **eval-only** and uses the *same* constellation grid, so the two
paths are consistent.

---

## 8. Configuration reference

All knobs live in [config.py](config.py):

| Parameter | Default | What it controls |
|---|---|---|
| `QAM_ORDER` | 16 | Constellation size M (perfect square). Higher M = more bits/symbol but more noise-sensitive. |
| `QAM_POWER` | 1.0 | Target average symbol power. Fixes the SNR definition. |
| `LEARNED_CONSTELLATION` | False | `True` = constellation points are trainable (re-normalized to `QAM_POWER` each step). |
| `SOFT_Q_INIT` | 5.0 | Starting `sigma_q` (hardness). |
| `SOFT_Q_MAX` | 100.0 | Maximum `sigma_q`. |
| `SOFT_Q_ANNEAL_RATE` | 5e-3 | `sigma_q` increment per optimizer step. |
| `KL_LAMBDA` | 0.05 | Weight on the usage-vs-uniform KL term in Phase-1 loss. |

---

## 9. Verifying it yourself

The quantizer has a standalone smoke test — run the module directly:

```bash
python quantizer.py
```

It checks that (1) every output symbol lands exactly on a constellation point
(hard forward works), (2) the straight-through gradient is finite and non-zero
(training signal flows), (3) the average output power matches `QAM_POWER`, (4) the
KL term is finite, and (5) `step_sigma()` increments the hardness.

The architecture tests in [test_pipeline_arch.py](test_pipeline_arch.py) also
assert that the quantizer output lies exactly on the grid and that the
straight-through gradient is non-zero:

```bash
pytest test_pipeline_arch.py -v
```
