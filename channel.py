"""
channel.py
Channel models applied to the quantized constellation symbols.

    IdentityChannel : passthrough (clean-channel baseline)
    AWGNChannel     : additive white Gaussian noise, y = x + n

Because upstream symbols are quantized to a unit-power constellation, the noise
std maps directly to an SNR (see config.snr_db_from_std).
"""

import torch
import torch.nn as nn


class IdentityChannel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AWGNChannel(nn.Module):
    def __init__(self, mean: float, std: float, enabled: bool = True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

    def enable(self, flag: bool):
        self.enabled = flag


if __name__ == "__main__":
    ch = AWGNChannel(mean=0.0, std=0.3)
    x = torch.zeros(10000)
    y = ch(x)
    n = y - x
    print(f"empirical noise mean={n.mean().item():.4f} (target 0.0)")
    print(f"empirical noise std ={n.std().item():.4f} (target 0.3)")
