# The Pipeline Flow When Using LDPC

This document explains exactly what happens, step by step, when the system runs
with **LDPC channel coding enabled** (`USE_LDPC = True`). The code lives in
[ldpc_codec.py](ldpc_codec.py); it is wired into the forward pass in
[pipeline.py](pipeline.py).

---

## 1. When is LDPC active?

LDPC is the **"separate source–channel coding"** configuration (Option B in the
experiment). Two hard rules:

- **Inference / evaluation only.** It is *never* used during training. In
  [pipeline.py:56](pipeline.py#L56) the LDPC branch runs only when
  `self.ldpc_codec is not None and not self.training`.
- **The codec is not differentiable.** It operates on hard bits, GF(2) matrices,
  and belief propagation — there is no gradient through it, which is why it can
  only appear at test time on an already-trained autoencoder.

The philosophy: the autoencoder is trained on a **clean channel** to focus purely
on compression; a classical LDPC code is then "bolted on" at test time to provide
all the channel error protection. Encoder and channel code are designed
independently — the classical *separation* approach.

Enable it in [config.py](config.py):

```python
USE_LDPC        = True
LDPC_CODE_RATE  = 0.5    # rate-1/2
LDPC_MAX_ITER   = 50     # belief-propagation iterations
```

The codec is constructed in [pipeline.py:142](pipeline.py#L142), sized from the
bottleneck: `n_info_bits = (BOTTLENECK_DIM // 2) * log2(QAM_ORDER)`.

---

## 2. Where LDPC sits in the pipeline

Compare the two forward paths. **Without** LDPC (training or plain eval), the
quantizer output goes straight through the channel:

```
BERT → AE encoder → quantizer (soft/hard) → AWGN → AE decoder → classifier
```

**With** LDPC, the differentiable quantizer is replaced by a hard-index →
bit-level pipeline that wraps the channel:

```
BERT → AE encoder → hard-quantize to indices
        → [ symbols→bits → LDPC encode → QAM map ] → AWGN
        → [ QAM soft-demap→LLRs → LDPC decode → bits→symbols ]
        → AE decoder → classifier
```

Everything in `[ ... ]` is the LDPC codec. The AE encoder, AE decoder, and
classifier are unchanged — LDPC only protects the symbols in transit.

---

## 3. Concrete dimensions (Option B: `BOTTLENECK_DIM = 64`, `QAM_ORDER = 16`)

These are the numbers the tensors actually take, at rate-1/2. `bps` = bits per
symbol = `log2(16) = 4`.

| Quantity | Formula | Value |
|---|---|---|
| Info I/Q symbols | `BOTTLENECK_DIM / 2` | **32** |
| Bits per symbol (`bps`) | `log2(QAM_ORDER)` | **4** |
| Info bits | `n_info_symbols · bps` | **128** |
| Coded bits | `n_info_bits / code_rate` | **256** |
| Transmitted QAM symbols | `n_coded_bits / bps` | **64** |
| **Transmitted real dims** | `n_tx_symbols · 2` | **128** |

The final 128 real dimensions match the joint model's transmission budget — that
is what makes the joint-vs-separate comparison a fair bandwidth comparison. (The
codec docstring uses `BOTTLENECK_DIM = 128` → 256/512/128 for its example; the
formulas are identical, only the numbers scale.)

---

## 4. Encode path — info bits → transmitted symbols

Starting from the AE encoder's normalized bottleneck `z` (shape
`(batch, BOTTLENECK_DIM)`):

### Step E0 — hard quantize to indices
[pipeline.py:59](pipeline.py#L59) → [quantizer.py:142](quantizer.py#L142)

Each I/Q pair is snapped to the nearest constellation point; we keep the **integer
index** $j^\star_k = \arg\min_j \lVert s_k - c_j\rVert^2$, not the coordinates.

```
z (batch, 64) ──► indices (batch, 32)   ∈ {0,…,15}
```

### Step E1 — symbols → Gray bits
[`symbols_to_bits`](ldpc_codec.py#L96)

Each index becomes its `bps`-bit **Gray code** via a lookup table
(`bit_table`, built in [`_build_bit_table`](ldpc_codec.py#L72)). Gray coding means
adjacent constellation points differ by exactly one bit, minimizing bit errors
when noise pushes a symbol to a neighbor.

```
indices (batch, 32) ──► info_bits (batch, 128)   binary
```

### Step E2 — LDPC encode (GF(2))
[`encode`](ldpc_codec.py#L104)

The info bits are zero-padded to the code dimension `_k` and multiplied by the
generator matrix $G$ modulo 2:

$$
\mathbf{x} = (G\,\mathbf{u}) \bmod 2 .
$$

This adds structured redundancy (rate-1/2 → twice as many bits).

```
info_bits (batch, 128) ──► coded_bits (batch, 256)   binary
```

> Note on `_k`: `pyldpc` may build a code whose true dimension `_k` is slightly
> larger than `n_info_bits`. The extra positions are zero-padded on encode and
> discarded on decode ([ldpc_codec.py:110](ldpc_codec.py#L110)).

### Step E3 — map coded bits → QAM symbols
[`map_to_symbols`](ldpc_codec.py#L116)

Coded bits are grouped into `bps`-bit chunks, converted from Gray back to a
constellation index (`gray_to_idx`), and looked up as **(I, Q) coordinates** on
the *same* grid the quantizer uses ([`build_square_qam`](quantizer.py#L30)).

```
coded_bits (batch, 256) ──► tx (batch, 128)   float I/Q coords  (= 64 symbols × 2)
```

---

## 5. The channel

[pipeline.py:63](pipeline.py#L63)

The transmitted coordinates pass through the AWGN channel, exactly as in the
non-LDPC path:

$$
\mathbf{r} = \mathbf{x} + \mathbf{n}, \qquad \mathbf{n} \sim \mathcal{N}(0, \sigma_n^2 I).
$$

The noise std $\sigma_n$ (`channel.std`) is read back out at
[pipeline.py:64](pipeline.py#L64) so the decoder can compute correct likelihoods.

```
tx (batch, 128) ──► rx (batch, 128)   noisy float I/Q coords
```

---

## 6. Decode path — received symbols → info bits

### Step D1 — soft demap to LLRs
[`demap_to_llrs`](ldpc_codec.py#L133)

For each received symbol the codec computes a Gaussian log-likelihood for every
constellation point, $\log p(y\mid c_j) = -\lVert y - c_j\rVert^2 / (2\sigma_n^2)$,
then produces a **log-likelihood ratio** for each of the `bps` bits by summing
(log-sum-exp) over the points where that bit is 0 vs. 1:

$$
\Lambda_k = \log\!\!\sum_{s:\,b_k(s)=0}\!\! p(y\mid s)\;-\;\log\!\!\sum_{s:\,b_k(s)=1}\!\! p(y\mid s) .
$$

Positive $\Lambda_k$ ⇒ bit likely 0 (matching `pyldpc`'s convention). This is
"soft" information: not a hard 0/1, but a confidence.

```
rx (batch, 128) + σ_n ──► llrs (batch, 256)   real-valued
```

### Step D2 — LDPC decode (belief propagation)
[`decode`](ldpc_codec.py#L157)

The LLRs are fed to `pyldpc.decode`, which runs **belief propagation** on the
parity-check matrix $H$ for up to `LDPC_MAX_ITER` iterations, then extracts the
message bits. The identity $y = \Lambda/2$ with `snr=0` makes `pyldpc`'s internal
channel LLR equal exactly to our $\Lambda$.

At low SNR, BP may not converge — this is expected and handled gracefully: it
returns its best estimate after `max_iter` iterations (the warning is suppressed
at [ldpc_codec.py:172](ldpc_codec.py#L172)). That graceful degradation is exactly
the error-correction behavior we want to measure.

```
llrs (batch, 256) ──► dec_bits (batch, 128)   binary (corrected info bits)
```

### Step D3 — bits → constellation coordinates
[`bits_to_symbols`](ldpc_codec.py#L185)

The decoded info bits are converted (Gray → index → coordinates) back to I/Q
points on the same grid, producing the received bottleneck vector the AE decoder
expects.

```
dec_bits (batch, 128) ──► z_received (batch, 64)   float
```

---

## 7. Back into the neural decoder

[pipeline.py:73](pipeline.py#L73)

From here the flow rejoins the standard path: the recovered bottleneck feeds the
AE decoder and then the classifier.

$$
\hat e = g_\phi(\mathbf{z}_{\text{received}}), \qquad
\ell = h_\psi(\hat e), \qquad
\hat y = \arg\max_c \ell_c .
$$

```
z_received (batch, 64) ──► AE decoder ──► x̂ (batch, 768) ──► classifier ──► logits (batch, 4)
```

---

## 8. End-to-end diagram

```
                       ┌─────────────────────── LDPC codec (eval only) ───────────────────────┐
                       │                                                                       │
Text → BERT → AE enc → │ hard-quantize → symbols→bits → LDPC encode → QAM map │→ AWGN →│ soft-demap→LLR → BP decode → bits→symbols │ → AE dec → classifier → class
        (768)   (64)   │   indices(32)     bits(128)     bits(256)   syms(64)  │  chan  │    LLR(256)       bits(128)     coords(64)  │   (768)        (4)
                       └───────────────────────────────────────────────────────────────────────┘
                                        │  encode path  │           │        │       decode path         │
```

Key invariants:
- Same constellation grid ([`build_square_qam`](quantizer.py#L30)) is used to
  transmit, demap, and reconstruct — the codec and the quantizer never disagree
  on the geometry.
- The AE encoder/decoder and classifier weights are identical to the non-LDPC
  run; only the transport between them changes.
- Rate-1/2 doubles the transmitted bits, so a `BOTTLENECK_DIM=64` LDPC model and
  a `BOTTLENECK_DIM=128` joint model both put **128 real dimensions** on the wire.

---

## 9. Verifying the round trip

The codec has a standalone smoke test that runs a high-SNR encode→channel→decode
loop and reports bit-error rate:

```bash
python ldpc_codec.py
```

Expected output: `n_info_bits=…, n_coded_bits=…, _k=…` and a `BER … (should be
~0)` near zero at `noise_std=0.05` — confirming the encode and decode paths are
mutually consistent.
