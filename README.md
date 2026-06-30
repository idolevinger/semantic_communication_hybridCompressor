# Semantic Communication over a Noisy Channel

Text classification where the **full transmission pipeline is learned end-to-end**.
A sentence is BERT-encoded, compressed by a learned autoencoder, quantized onto a
QAM constellation, sent through a simulated AWGN channel, reconstructed, and
classified — all as a single differentiable system trained on AG News (4 classes:
World, Sports, Business, Sci/Tech).

The project also implements an optional **rate-1/2 LDPC** block (inference-only) to
run the core experiment of modern semantic communication research: does **joint
source-channel coding** (one learned encoder that handles everything) beat the
classical **separation approach** (a clean compressor + a classical channel code)?

---

## Full pipeline

```
                         ┌─────────────── trained end-to-end ────────────────┐
                         │                                                    │
Text ──► BERT ──► AE Encoder ──► Quantizer ──► AWGN Channel ──► AE Decoder ──► Classifier ──► Class
         (frozen)  768 → 128      soft/hard       y = x + n      128 → 768       768 → 4
                   + L2 norm      16-QAM grid     (64 I/Q syms)
                         │                                                    │
                         └────────────────────────────────────────────────────┘
```

**BERT** is frozen and pre-cached. Everything after it is trained jointly.

The **AE Encoder** (768→128) compresses the sentence embedding and L2-normalizes
the output so every transmitted vector has the same total power — required for
a stable SNR definition at the quantizer.

The **Quantizer** maps the 128-D real vector (64 I/Q pairs) onto the nearest point of
a fixed 16-QAM grid. This is a hard, non-differentiable operation in the forward pass;
gradients flow through a soft (softmax-weighted) assignment using the
straight-through estimator. A hardness parameter `sigma_q` is annealed upward
during training so the soft assignment converges to the hard one.

The **AWGN Channel** adds Gaussian noise: `y = x + n`, `n ~ N(0, σ²I)`.
Because the symbols are power-normalized, `σ` directly controls the SNR:

```
SNR (dB) = 10 · log₁₀(QAM_POWER / σ²)
```

The **AE Decoder** (128→768) reconstructs the original BERT embedding from the noisy
received vector. Classification runs on this reconstruction, not on the compressed
bottleneck — the system learns to recover enough semantic content to classify correctly.

The **Classifier** is a residual MLP (768 → 128 → 128 → 64 → 4). It reads the
768-D reconstruction.

### Training loss

```
L = CrossEntropy(logits, y)
  + 0.1 · MSE(reconstructed_embedding, original_embedding)
  + 0.05 · KL(constellation_usage || Uniform)
```

- **CE** — primary classification objective.
- **MSE** — forces the AE decoder to reconstruct the original BERT embedding, keeping
  the bottleneck semantically meaningful rather than collapsing to a pure classification
  shortcut.
- **KL** — regularizer that pushes the quantizer to use all constellation points roughly
  equally, preventing mode collapse where most symbols cluster onto a few points.

---

## The core experiment: joint vs. separate source-channel coding

Shannon's separation theorem says that for point-to-point channels with infinite block
lengths, you can design compression and error correction independently without losing
performance. With **finite block lengths and learned encoders**, joint design can
outperform separation. This is the central claim of semantic communication research.

The experiment tests it directly with a **fixed transmission budget of 128 real
dimensions**:

| | Option A — Joint | Option B — Separate |
|---|---|---|
| Design philosophy | One encoder learns everything | Clean compressor + classical LDPC |
| `BOTTLENECK_DIM` | 128 | 64 |
| Training noise | **ON** | **OFF** (clean channel) |
| LDPC at test | **OFF** | **ON** (rate-1/2) |
| Info bits | 256 | 128 info → 256 coded |
| Transmitted real dims | 128 | 128 (LDPC doubles the bits) |

**Option A** trains with noise so the encoder learns noise-robust representations.
No external error correction is needed.

**Option B** trains on a clean channel so the encoder focuses entirely on
compression. Rate-1/2 LDPC is bolted on at test time to provide error protection.
The encoder never sees the LDPC; they are designed independently (separation).

**Baseline C** uses Option B's checkpoint but removes the LDPC at test time. Since
this encoder was never trained with noise and has no error correction, it degrades
rapidly — confirming that LDPC is doing real work in Option B.

**Expected result:**

```
Accuracy (%)
 │
 │  B wins here         A wins here
 │ (LDPC corrects       (wider bottleneck,
 │  channel errors)      more semantic info)
 │      ╲                     ╱
 │       ╲──── crossover ────╱
 │
 │  C (baseline): fast degradation without LDPC
 │
 └──────────────────────────────────── SNR (dB)
   0 dB                           20 dB
```

The **crossover SNR** is the main finding: the point at which learned noise
robustness stops being sufficient and classical error correction becomes necessary.

---

## Repository layout

```
project/
├── config.py               all hyperparameters and paths (single source of truth)
├── pipeline.py             build_pipeline() factory — entry point for every script
│
├── bert_encoder.py         frozen BERT (bert-base-uncased), CLS pooling
├── build_embeddings.py     pre-cache BERT embeddings to data/  (run once)
├── autoencoder.py          AEEncoder (768→bottleneck) + AEDecoder (bottleneck→768)
├── quantizer.py            ConstellationQuantizer: soft/hard QAM, straight-through
├── channel.py              AWGNChannel and IdentityChannel
├── decoder.py              TaskDecoder: residual MLP classifier (768→4)
├── ldpc_codec.py           LDPCCodec: rate-1/2 LDPC + Gray-coded 16-QAM (eval only)
│
├── train.py                training loop with early stopping
├── test.py                 evaluation with overall + per-class breakdown
├── main.py                 run clean and noisy experiments back-to-back
├── predict.py              single-sentence CLI inference
├── app.py                  Flask web interface
│
├── test_pipeline_arch.py   architecture unit tests (pytest, no trained weights needed)
├── requirements.txt
│
├── data/                   cached embeddings written by build_embeddings.py
├── results/                checkpoints and plots written by test scripts
│
├── Tests/                  scientific evaluation scripts (see Tests/README.md)
│   ├── dim_sweep.py        sweep bottleneck dimensions at constant SNR
│   ├── snr_sweep.py        sweep channel SNR with dedicated models
│   ├── mismatch_sweep.py   evaluate SNR mismatch robustness
│   ├── plot_reconstruction.py cosine similarity of reconstruction vs BERT
│   └── baseline_sim.py     statistical baseline calculations
│
└── Docs/                   experiment reports and final outputs
    ├── experiment_results.md comprehensive analysis report
    ├── results_report.html   printable RTL report with graphs
    ├── export_html.py        HTML report generation script
    └── rtl_style.css         RTL styling support
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step-by-step: full flow from scratch to results

### Step 1 — Cache BERT embeddings (run once)

BERT is slow and frozen, so all 768-D embeddings are pre-computed and saved.
This step downloads `bert-base-uncased` and the AG News dataset (~50k train +
7.6k test), then writes `data/stage1_embeddings.pt`.

```bash
python build_embeddings.py
```

Everything after this step reads from the cache; BERT is never touched again.

---

### Step 2 — Train Option A (joint source-channel coding)

In `config.py` set:

```python
BOTTLENECK_DIM      = 128
NOISE_APPLY_TRAIN   = True    # encoder sees AWGN during training
```

Then train:

```bash
python train.py
cp results/trained_pipeline.pt results/model_joint_128.pt
```

The encoder learns to produce representations that survive channel noise.
No LDPC is used at any point.

---

### Step 3 — Train Option B (separate source + channel coding)

In `config.py` set:

```python
BOTTLENECK_DIM      = 64
NOISE_APPLY_TRAIN   = False   # encoder trains on a clean channel
```

Then train:

```bash
python train.py
cp results/trained_pipeline.pt results/model_separate_64.pt
```

The encoder only learns to compress faithfully. LDPC provides all channel
protection at test time. The encoder and the channel code are designed
independently — this is the classical separation approach.

---

### Step 4 — Run Scientific Evaluations (Sweeps)

The `Tests/` directory contains all scientific evaluations for the project. Each script isolates a specific variable (bottleneck dimension, channel noise, or SNR mismatch). 
For detailed explanations of each test and its exact parameters, see `Tests/README.md`.

```bash
python Tests/dim_sweep.py
python Tests/snr_sweep.py
python Tests/mismatch_sweep.py
```

These scripts automatically handle training (or loading) the required models, evaluate them, and plot the comparative results directly into their respective subfolders inside the `results/` directory.

---

### Step 5 — Evaluate a single checkpoint in detail

```bash
python test.py
```

Reads `results/trained_pipeline.pt` (whatever was last trained) and prints
overall accuracy + loss plus a per-class breakdown. Noise is controlled by
`NOISE_APPLY_EVAL` in `config.py`.

---

### Step 6 — Classify a sentence

```bash
# single sentence, no noise
python predict.py "NASA launches a new telescope to study distant galaxies"

# single sentence, with AWGN at the configured noise level
python predict.py --noise "Stocks rallied after the rate cut"

# interactive mode
python predict.py
python predict.py --noise
```

BERT is loaded fresh each time. The pipeline uses `results/trained_pipeline.pt`.

---

### Step 7 — Web interface

```bash
python app.py
# open http://localhost:5000
```

---

### Alternative: run both clean and noisy experiments back-to-back

```bash
python main.py
```

Trains and evaluates twice: once without noise, once with. Useful for a quick
side-by-side comparison without manually changing `config.py`.

---

## Configuration reference

All settings live in `config.py`. Nothing should be redefined in other modules.

### Architecture

| Parameter | Default | Notes |
|---|---|---|
| `BOTTLENECK_DIM` | 128 | AE encoder output dim. Each pair of real dims is one I/Q symbol, so `BOTTLENECK_DIM/2` symbols are transmitted. **Changing this requires retraining — old checkpoints are incompatible.** |
| `BERT_DIM` | 768 | Fixed by BERT. Do not change. |
| `HIDDEN_DIM` | 128 | Classifier hidden size. |
| `NUM_CLASSES` | 4 | AG News has 4 classes. |

### Constellation / quantizer

| Parameter | Default | Notes |
|---|---|---|
| `QAM_ORDER` | 16 | Constellation size M (must be a perfect square: 4, 16, 64, 256…). Higher M = more bits per symbol = finer resolution, but more noise-sensitive. |
| `QAM_POWER` | 1.0 | Target average symbol power. The encoder L2-normalizes to this. |
| `LEARNED_CONSTELLATION` | False | `True` = constellation points are trainable parameters, renormalized to `QAM_POWER` after each step. |
| `SOFT_Q_INIT` | 5.0 | Initial value of `sigma_q` (quantizer hardness). |
| `SOFT_Q_MAX` | 100.0 | `sigma_q` is clamped to this maximum. |
| `SOFT_Q_ANNEAL_RATE` | 5e-3 | `sigma_q` increment per optimizer step. Reaches MAX at ~19k steps. |
| `KL_LAMBDA` | 0.05 | Weight on the constellation-uniformity KL regularizer. Set to 0 for very large M (≥4096). |

### Channel

| Parameter | Default | Notes |
|---|---|---|
| `NOISE_STD` | 0.3 | AWGN standard deviation. With `QAM_POWER=1.0` this is ~10.5 dB SNR. |
| `NOISE_APPLY_TRAIN` | True | Inject AWGN during training. Set `False` for Option B (clean-trained encoder). |
| `NOISE_APPLY_EVAL` | True | Inject AWGN during evaluation. |

SNR conversion: `SNR (dB) = 10 · log₁₀(QAM_POWER / NOISE_STD²)`

### LDPC (inference / evaluation only)

| Parameter | Default | Notes |
|---|---|---|
| `USE_LDPC` | False | Enable LDPC coding at test time. Never active during training. |
| `LDPC_CODE_RATE` | 0.5 | Rate-1/2: doubles the number of transmitted bits. |
| `LDPC_MAX_ITER` | 50 | Belief-propagation decoding iterations. |

### Training

| Parameter | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 128 | |
| `LR` | 3e-4 | AdamW learning rate. |
| `EPOCHS` | 50 | Maximum epochs before early stopping fires. |
| `EARLY_STOP_PATIENCE` | 8 | Stop if val accuracy does not improve for this many epochs. |
| `RECON_LOSS_WEIGHT` | 0.1 | Weight on the AE reconstruction MSE term. |
| `LABEL_SMOOTHING` | 0.05 | Cross-entropy label smoothing. |
| `CLASS_WEIGHTS` | [1, 1, 1.15, 1.15] | Per-class CE weights (Business and Sci/Tech are slightly upweighted). |

---

## LDPC codec internals

`ldpc_codec.py` wraps `pyldpc`. It is **never called during training** — only in the
`pipeline.forward()` eval branch when `USE_LDPC=True`.

Encode path (info bits → transmitted symbols):

```
(batch, n_info_symbols) indices           hard-quantized QAM symbol indices
        │
        ▼  symbols_to_bits()
(batch, n_info_bits) binary               Gray-coded bits per symbol
        │
        ▼  encode()                       GF(2) matrix multiply with generator G
(batch, n_coded_bits) binary              rate-1/2 LDPC codeword
        │
        ▼  map_to_symbols()               Gray → symbol index → constellation coords
(batch, 2·n_tx_symbols) float            QAM I/Q coordinates, ready to transmit
```

Decode path (received symbols → decoded info bits):

```
(batch, 2·n_tx_symbols) float            noisy received I/Q coordinates
        │
        ▼  demap_to_llrs(noise_std)       soft demapping: log P(bit=0)/P(bit=1)
(batch, n_coded_bits) LLR                log-likelihood ratios
        │
        ▼  decode()                       belief-propagation with H matrix
(batch, n_coded_bits) binary             decoded codeword
        │
        ▼  bits_to_symbols()             bits → constellation coords (same grid as quantizer)
(batch, 2·n_info_symbols) float          feeds into AE decoder
```

For `BOTTLENECK_DIM=64`, `QAM_ORDER=16`:
- 64 info dims → 32 I/Q symbols → 128 info bits
- Rate-1/2 LDPC → 256 coded bits → 64 QAM symbols → **128 transmitted real dims**

This matches Option A's 128 real dims, making the bandwidth comparison fair.

---

## Checkpoint format

```python
{
    "autoencoder": state_dict,   # AEEncoder + AEDecoder weights
    "quantizer":   state_dict,   # sigma_q, step counter, constellation points
    "decoder":     state_dict,   # classifier weights
    "epoch":       int,
    "val_acc":     float,
}
```

Checkpoints are **not portable across different `BOTTLENECK_DIM` values**.
If you change that setting, delete the old checkpoint and retrain.
The channel (`AWGNChannel` / `IdentityChannel`) is reconstructed at load time
from config, not saved in the checkpoint.

---

## Architecture tests

```bash
pytest test_pipeline_arch.py -v
```

Verifies (without requiring trained weights or a GPU):
- Classifier input is 768-D (the AE reconstruction), not 128-D (the bottleneck).
- `forward_train` returns `(B, 4)` logits and `(B, 768)` reconstruction.
- `forward` (inference) returns `(B, 4)` logits.
- The reconstruction changes when noise is present — confirming the AE decoder is
  in the signal path and not bypassed.
- Quantizer output lies exactly on constellation points (hard forward pass works).
- Straight-through gradient is non-zero and finite (training signal flows).

---

## Smoke tests

Every module has a `__main__` block that can be run standalone:

```bash
python autoencoder.py        # shape + norm check
python quantizer.py          # constellation, gradient, power, KL
python channel.py            # empirical noise mean/std
python decoder.py            # output shape
python ldpc_codec.py         # round-trip BER at high SNR
python bert_encoder.py       # embedding shape + cosine similarity sanity
python pipeline.py           # full forward pass shapes
```


