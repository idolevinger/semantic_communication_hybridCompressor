"""
autoencoder.py
Learned compressor for the semantic pipeline.

    AEEncoder : 768 -> 512 -> 256 -> bottleneck   (+ fixed-power output normalization)
    AEDecoder : bottleneck -> 256 -> 512 -> 768   (train-only reconstruction head)

The encoder output is L2-normalized to a fixed total power so that the downstream
quantizer sees vectors at a known, constant scale (otherwise "nearest constellation
point" is ill-defined and the effective SNR drifts). The per-symbol average power is
set to QAM_POWER, with the number of complex symbols = bottleneck_dim // 2.

The AE decoder reconstructs the 768-D embedding from the received signal. The
classifier reads this reconstruction, so the AE decoder is part of the inference
path (not discarded after training).
"""

import math

import torch
import torch.nn as nn

from config import BERT_DIM, BOTTLENECK_DIM, QAM_POWER


class AEEncoder(nn.Module):
    def __init__(self, input_dim: int = BERT_DIM, bottleneck_dim: int = BOTTLENECK_DIM,
                 power: float = QAM_POWER, normalize: bool = True):
        super().__init__()
        if bottleneck_dim % 2 != 0:
            raise ValueError("bottleneck_dim must be even (real dims pair into I/Q symbols)")
        self.normalize = normalize
        # total target vector energy = (num complex symbols) * per-symbol power
        num_symbols = bottleneck_dim // 2
        self.target_norm = math.sqrt(num_symbols * power)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            # scale each vector to a fixed L2 norm -> fixed average per-symbol power
            norm = z.norm(dim=1, keepdim=True).clamp_min(1e-8)
            z = z / norm * self.target_norm
        return z


class AEDecoder(nn.Module):
    def __init__(self, bottleneck_dim: int = BOTTLENECK_DIM, output_dim: int = BERT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Bundles encoder + reconstruction decoder.
    - encode(x)       : returns normalized bottleneck z
    - recon_decoder(z): reconstructs 768-D embedding from z; used at both training
                        (MSE regularizer) and inference (feeds the classifier)
    - forward(x)      : training convenience wrapper -> (z, x_hat)
    """

    def __init__(self, input_dim: int = BERT_DIM, bottleneck_dim: int = BOTTLENECK_DIM,
                 power: float = QAM_POWER, normalize: bool = True):
        super().__init__()
        self.encoder = AEEncoder(input_dim, bottleneck_dim, power, normalize)
        self.recon_decoder = AEDecoder(bottleneck_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.recon_decoder(z)
        return z, x_hat


if __name__ == "__main__":
    ae = AEEncoder()
    x = torch.randn(8, BERT_DIM)
    # BatchNorm needs train mode + batch > 1 for this smoke test
    z = ae(x)
    print("z shape:", tuple(z.shape))
    print("per-vector norm (should all equal target_norm=%.4f):" % ae.target_norm)
    print(z.norm(dim=1))
