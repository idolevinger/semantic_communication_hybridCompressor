"""
ldpc_codec.py
Rate-1/2 LDPC channel coding block — used at inference/evaluation only.

Pipeline position (when USE_LDPC=True):
    hard-quantize indices → bits → LDPC encode → 16-QAM map
    → AWGN channel → 16-QAM soft-demap (LLRs) → LDPC decode → bits → constellation coords

install dependency:  pip install pyldpc
"""

import math
import numpy as np
import torch

from quantizer import build_square_qam
from config import QAM_POWER


class LDPCCodec:
    """
    LDPC codec wrapping pyldpc.  All heavy work is numpy/CPU; call .to_device()
    on the returned tensors before passing them back into PyTorch layers.

    Attributes set at construction time (all read-only after __init__):
        n_info_bits   -- info bits per sentence (from quantizer, e.g. 256)
        n_coded_bits  -- LDPC codeword length (e.g. 512)
        _k            -- actual LDPC code dimension (≥ n_info_bits; zero-padded)
        bps           -- bits per QAM symbol (4 for 16-QAM)
        n_info_symbols-- original I/Q symbol count (n_info_bits // bps = 64)
        n_tx_symbols  -- transmitted QAM symbols after LDPC (n_coded_bits // bps = 128)
    """

    def __init__(self, n_info_bits: int, code_rate: float, qam_order: int, max_iter: int):
        import pyldpc

        self.n_info_bits = n_info_bits                          # 256
        self.n_coded_bits = round(n_info_bits / code_rate)      # 512
        self.qam_order = qam_order                              # 16
        self.bps = int(math.log2(qam_order))                   # 4
        self.max_iter = max_iter
        self.n_info_symbols = n_info_bits // self.bps           # 64
        self.n_tx_symbols = self.n_coded_bits // self.bps       # 128

        # Build LDPC parity-check (H) and generator (G) matrices.
        # pyldpc may produce _k slightly > n_info_bits; we zero-pad info bits to _k.
        d_v = 4
        d_c = round(d_v / code_rate)                           # 8 for rate-1/2
        H, G = pyldpc.make_ldpc(
            self.n_coded_bits, d_v, d_c, systematic=True, sparse=True
        )
        self._H = H                       # (m, n_coded) parity-check matrix
        self._G = G                       # (n_coded, _k)  pyldpc calls this "tG"
        self._k = G.shape[1]             # actual LDPC code dimension

        # 16-QAM constellation — identical grid to ConstellationQuantizer
        self.constellation = build_square_qam(qam_order, QAM_POWER).numpy()  # (M, 2)

        # Gray-coded bit table: bit_table[idx] = bps-bit array for symbol index idx
        self.bit_table = self._build_bit_table()   # (M, bps) int
        # Inverse: integer of Gray bits → constellation index
        weights = 2 ** np.arange(self.bps - 1, -1, -1, dtype=np.int64)
        gray_ints = (self.bit_table * weights).sum(axis=1)
        self.gray_to_idx = np.zeros(qam_order, dtype=np.int32)
        for idx, g in enumerate(gray_ints):
            self.gray_to_idx[g] = idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bit_table(self) -> np.ndarray:
        """Gray-coded bit assignment for square QAM. Returns (M, bps) int array."""
        n = int(math.sqrt(self.qam_order))   # 4 for 16-QAM
        bpd = self.bps // 2                  # bits per I or Q dimension

        def gray_bits(i):
            g = i ^ (i >> 1)
            return [(g >> (bpd - 1 - b)) & 1 for b in range(bpd)]

        table = np.zeros((self.qam_order, self.bps), dtype=np.int32)
        for idx in range(self.qam_order):
            table[idx] = gray_bits(idx // n) + gray_bits(idx % n)
        return table

    @staticmethod
    def _logsumexp(x: np.ndarray) -> np.ndarray:
        """Numerically stable log-sum-exp over last axis."""
        c = x.max(axis=-1, keepdims=True)
        return np.log(np.exp(x - c).sum(axis=-1)) + c[..., 0]

    # ------------------------------------------------------------------
    # Encode path: info bits → transmitted QAM symbols
    # ------------------------------------------------------------------

    def symbols_to_bits(self, indices: torch.Tensor) -> np.ndarray:
        """
        (batch, n_info_symbols) int tensor → (batch, n_info_bits) binary numpy.
        Converts hard-quantized symbol indices to Gray-coded bits.
        """
        idx = indices.cpu().numpy()          # (batch, 64)
        return self.bit_table[idx].reshape(idx.shape[0], self.n_info_bits)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        (batch, n_info_bits) binary → (batch, n_coded_bits) binary via GF(2).
        Zero-pads info_bits to self._k if needed (pyldpc may produce _k > n_info_bits).
        """
        batch = info_bits.shape[0]
        padded = np.zeros((batch, self._k), dtype=np.float64)
        padded[:, : self.n_info_bits] = info_bits
        # G is (n_coded, _k); codeword = G @ v for each v → batch form:
        # (batch, _k) @ (_k, n_coded) = (batch, n_coded)
        return ((padded @ self._G.T) % 2).astype(np.int32)

    def map_to_symbols(self, coded_bits: np.ndarray) -> torch.Tensor:
        """
        (batch, n_coded_bits) binary → (batch, 2*n_tx_symbols) float tensor.
        Groups coded bits into bps-bit chunks, looks up Gray-coded QAM coordinates.
        """
        batch = coded_bits.shape[0]
        bits_3d = coded_bits.reshape(batch, self.n_tx_symbols, self.bps)
        weights = 2 ** np.arange(self.bps - 1, -1, -1, dtype=np.int64)
        gray_ints = (bits_3d * weights).sum(axis=-1).astype(np.int32)   # (batch, n_tx)
        sym_idx = self.gray_to_idx[gray_ints]                            # (batch, n_tx)
        coords = self.constellation[sym_idx]                             # (batch, n_tx, 2)
        return torch.tensor(coords.reshape(batch, -1), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Decode path: received QAM symbols → decoded info bits
    # ------------------------------------------------------------------

    def demap_to_llrs(self, received: torch.Tensor, noise_std: float) -> np.ndarray:
        """
        (batch, 2*n_tx_symbols) float tensor → (batch, n_coded_bits) LLR numpy.

        LLR_k = log Σ_{s: bit_k=0} p(y|s) − log Σ_{s: bit_k=1} p(y|s)
        Positive LLR → bit likely 0 (matches pyldpc's convention).
        """
        rx = received.cpu().numpy().reshape(-1, self.n_tx_symbols, 2)
        sigma2 = max(noise_std ** 2, 1e-10)

        cons = self.constellation[np.newaxis, np.newaxis]                # (1, 1, M, 2)
        diff = rx[:, :, np.newaxis] - cons                               # (batch, n_tx, M, 2)
        dist2 = (diff ** 2).sum(axis=-1)                                 # (batch, n_tx, M)
        log_p = -dist2 / (2 * sigma2)

        llrs = np.zeros((rx.shape[0], self.n_tx_symbols, self.bps))
        for k in range(self.bps):
            m0 = self.bit_table[:, k] == 0
            m1 = self.bit_table[:, k] == 1
            llrs[:, :, k] = (
                self._logsumexp(log_p[:, :, m0]) - self._logsumexp(log_p[:, :, m1])
            )
        return llrs.reshape(rx.shape[0], self.n_coded_bits)

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """
        (batch, n_coded_bits) LLR → (batch, n_info_bits) decoded binary.

        LLRs are fed to pyldpc via the identity:
            pyldpc computes Lc = 2*y / (10^(−snr/10))
        Passing y = llr/2 with snr=0 (var=1) gives Lc = llr exactly.
        """
        import warnings
        import pyldpc

        # pyldpc.decode supports batch: y shape (n_coded, batch)
        y_hack = (llrs / 2.0).T                                 # (n_coded, batch)
        # Non-convergence at low SNR is expected — BP returns its best estimate
        # after maxiter iterations, which is the correct graceful-degradation behavior.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Decoding stopped before convergence")
            x_dec = pyldpc.decode(self._H, y_hack, snr=0, maxiter=self.max_iter)
        if x_dec.ndim == 1:
            x_dec = x_dec[:, np.newaxis]                        # (n_coded, 1)

        batch = llrs.shape[0]
        decoded = np.zeros((batch, self.n_info_bits), dtype=np.int32)
        for i in range(batch):
            msg = pyldpc.get_message(self._G, x_dec[:, i])
            decoded[i] = msg[: self.n_info_bits]
        return decoded

    def bits_to_symbols(self, decoded_bits: np.ndarray) -> torch.Tensor:
        """
        (batch, n_info_bits) binary → (batch, 2*n_info_symbols) float tensor.
        Converts decoded info bits back to constellation point I/Q coordinates,
        matching the grid used by the upstream ConstellationQuantizer.
        """
        batch = decoded_bits.shape[0]
        bits_3d = decoded_bits.reshape(batch, self.n_info_symbols, self.bps)
        weights = 2 ** np.arange(self.bps - 1, -1, -1, dtype=np.int64)
        gray_ints = (bits_3d * weights).sum(axis=-1).astype(np.int32)
        sym_idx = self.gray_to_idx[gray_ints]
        coords = self.constellation[sym_idx]
        return torch.tensor(coords.reshape(batch, -1), dtype=torch.float32)


if __name__ == "__main__":
    from config import BOTTLENECK_DIM, QAM_ORDER, LDPC_CODE_RATE, LDPC_MAX_ITER
    import math as _math

    n_info = BOTTLENECK_DIM // 2 * int(_math.log2(QAM_ORDER))
    codec = LDPCCodec(n_info, LDPC_CODE_RATE, QAM_ORDER, LDPC_MAX_ITER)
    print(f"n_info_bits={codec.n_info_bits}, n_coded_bits={codec.n_coded_bits}, _k={codec._k}")

    rng = np.random.default_rng(0)
    batch = 4
    # Simulate a high-SNR round-trip
    indices = torch.randint(0, QAM_ORDER, (batch, codec.n_info_symbols))
    info_bits = codec.symbols_to_bits(indices)
    coded = codec.encode(info_bits)
    tx = codec.map_to_symbols(coded)

    noise_std = 0.05
    rx = tx + torch.randn_like(tx) * noise_std
    llrs = codec.demap_to_llrs(rx, noise_std)
    dec_bits = codec.decode(llrs)
    recovered = codec.bits_to_symbols(dec_bits)

    ber = np.mean(dec_bits != info_bits)
    print(f"BER at noise_std={noise_std}: {ber:.4f}  (should be ~0)")
    print(f"tx shape: {tuple(tx.shape)}, rx shape: {tuple(rx.shape)}")
    print(f"recovered shape: {tuple(recovered.shape)}")
