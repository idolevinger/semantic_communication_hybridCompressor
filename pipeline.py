"""
pipeline.py
Wires all stages into one model and provides a SINGLE construction function used
by train.py, test.py, predict.py and app.py. Nothing else should hand-assemble
the pipeline -- add a new stage here once, and every script picks it up.

    (BERT embeddings) -> AE Encoder -> Quantizer -> Channel -> AE Decoder -> TaskDecoder -> logits

The classifier (TaskDecoder) reads the 768-D reconstruction produced by the AE
Decoder, not the compressed bottleneck directly. The AE Decoder is therefore part
of the inference path, not just a training-time regularizer.

BERT itself is frozen and embeddings are pre-cached, so it is not part of the
trainable forward pass. predict.py / app.py call BERTEncoder once on raw text
before entering this pipeline.
"""

import math
import torch
import torch.nn as nn

from autoencoder import Autoencoder
from quantizer import ConstellationQuantizer
from channel import IdentityChannel, AWGNChannel
from decoder import TaskDecoder
from config import (
    BERT_DIM,
    BOTTLENECK_DIM,
    QAM_ORDER,
    QAM_POWER,
    LEARNED_CONSTELLATION,
    SOFT_Q_INIT,
    SOFT_Q_MAX,
    SOFT_Q_ANNEAL_RATE,
    NOISE_MEAN,
    NOISE_STD,
    MODEL_FILE,
    USE_LDPC,
    LDPC_CODE_RATE,
    LDPC_MAX_ITER,
)


class SemanticPipeline(nn.Module):
    def __init__(self, autoencoder, quantizer, channel, decoder, ldpc_codec=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.quantizer = quantizer
        self.channel = channel
        self.decoder = decoder
        self.ldpc_codec = ldpc_codec   # LDPCCodec instance or None

    def _received_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Push the encoded bottleneck through the channel and return what the
        receiver recovers. Uses the LDPC branch (eval/inference only) when a codec
        is attached, otherwise the standard soft-quantizer + AWGN path."""
        if self.ldpc_codec is not None and not self.training:
            # LDPC path (eval/inference only — non-differentiable)
            device = z.device
            indices = self.quantizer.hard_quantize_indices(z)         # (batch, num_symbols)
            info_bits = self.ldpc_codec.symbols_to_bits(indices)      # (batch, n_info_bits)
            coded_bits = self.ldpc_codec.encode(info_bits)            # (batch, n_coded_bits)
            tx = self.ldpc_codec.map_to_symbols(coded_bits).to(device)# (batch, 2*n_tx_symbols)
            rx = self.channel(tx)                                     # + AWGN
            noise_std = getattr(self.channel, 'std', 0.0)
            llrs = self.ldpc_codec.demap_to_llrs(rx, noise_std)       # (batch, n_coded_bits)
            dec_bits = self.ldpc_codec.decode(llrs)                   # (batch, n_info_bits)
            return self.ldpc_codec.bits_to_symbols(dec_bits).to(device)
        # Standard path (training, or eval without LDPC)
        z = self.quantizer(z)
        return self.channel(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_recon(x)[0]

    def forward_with_recon(self, x: torch.Tensor):
        """Like forward(), but also returns the 768-D reconstruction x_hat so
        callers can measure semantic distortion (MSE) alongside classification.
        Covers both the LDPC and standard channel paths."""
        z = self.autoencoder.encode(x)              # 768 -> bottleneck (normalized)
        z_received = self._received_latent(z)       # through channel (+ LDPC if attached)
        x_hat = self.autoencoder.recon_decoder(z_received)  # bottleneck -> 768-D
        logits = self.decoder(x_hat)                # 768-D -> logits
        return logits, x_hat

    def forward_train(self, x: torch.Tensor):
        """Training path that also returns the AE reconstruction for the MSE regularizer."""
        z = self.autoencoder.encode(x)              # 768 -> bottleneck (normalized)
        zq = self.quantizer(z)                      # snap to constellation
        zc = self.channel(zq)                       # + AWGN (received bottleneck)
        x_hat = self.autoencoder.recon_decoder(zc)  # reconstruct 768-D from received signal
        logits = self.decoder(x_hat)                # classify on 768-D reconstruction
        return logits, x_hat

    # ---- checkpoint I/O (single source of truth for the format) ----
    def save_checkpoint(self, path: str, epoch: int = -1, val_acc: float = -1.0):
        torch.save(
            {
                "autoencoder": self.autoencoder.state_dict(),
                "quantizer": self.quantizer.state_dict(),
                "decoder": self.decoder.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            },
            path,
        )

    def load_checkpoint(self, path: str, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.quantizer.load_state_dict(ckpt["quantizer"])
        self.decoder.load_state_dict(ckpt["decoder"])
        return ckpt


def build_pipeline(
    use_noise: bool,
    device,
    checkpoint_path: str = None,
    bottleneck_dim: int = None,
    use_ldpc: bool = None,
    noise_std: float = None,
) -> SemanticPipeline:
    """
    Single factory for the whole model. Set checkpoint_path to load trained weights.

    bottleneck_dim / use_ldpc / noise_std override the corresponding config values
    when provided — used by snr_sweep.py to evaluate models with different settings
    without touching config.py.
    """
    _bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else BOTTLENECK_DIM
    _use_ldpc = use_ldpc if use_ldpc is not None else USE_LDPC
    _noise_std = noise_std if noise_std is not None else NOISE_STD

    autoencoder = Autoencoder(
        input_dim=BERT_DIM, bottleneck_dim=_bottleneck_dim, power=QAM_POWER, normalize=True
    )
    quantizer = ConstellationQuantizer(
        bottleneck_dim=_bottleneck_dim,
        order=QAM_ORDER,
        power=QAM_POWER,
        sigma_init=SOFT_Q_INIT,
        sigma_max=SOFT_Q_MAX,
        anneal_rate=SOFT_Q_ANNEAL_RATE,
        learnable=LEARNED_CONSTELLATION,
    )
    channel = AWGNChannel(mean=NOISE_MEAN, std=_noise_std) if use_noise else IdentityChannel()

    decoder = TaskDecoder(input_dim=BERT_DIM)

    ldpc_codec = None
    if _use_ldpc:
        from ldpc_codec import LDPCCodec
        n_info_bits = _bottleneck_dim // 2 * int(math.log2(QAM_ORDER))
        ldpc_codec = LDPCCodec(
            n_info_bits=n_info_bits,
            code_rate=LDPC_CODE_RATE,
            qam_order=QAM_ORDER,
            max_iter=LDPC_MAX_ITER,
        )

    pipeline = SemanticPipeline(autoencoder, quantizer, channel, decoder, ldpc_codec).to(device)

    if checkpoint_path:
        pipeline.load_checkpoint(checkpoint_path, device=device)

    return pipeline


if __name__ == "__main__":
    device = torch.device("cpu")
    p = build_pipeline(use_noise=True, device=device)
    p.train()  # BatchNorm needs batch > 1
    x = torch.randn(8, BERT_DIM)
    logits, x_hat = p.forward_train(x)
    print("logits shape:", tuple(logits.shape))      # (8, NUM_CLASSES)
    print("reconstruction shape:", tuple(x_hat.shape))  # (8, 768)
    p.eval()
    print("eval logits shape:", tuple(p(x).shape))
