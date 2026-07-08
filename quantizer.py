"""
quantizer.py
Constellation-constrained quantization stage (the genuinely new pipeline piece).

Implements the soft-to-hard quantizer used in constellation-constrained deep
joint source-channel coding:

  - The real bottleneck vector (dim D) is reshaped into D/2 complex (I/Q) symbols.
  - Each symbol is snapped to the NEAREST point of a fixed M-QAM constellation
    in the forward pass (this is what is "transmitted").
  - The gradient flows through a SOFT (softmax-weighted) assignment, via the
    straight-through trick:  out = hard + (soft - soft.detach())
  - A "hardness" parameter sigma_q controls how peaked the soft assignment is,
    and is annealed upward during training.
  - An empirical constellation-usage distribution is tracked so training can add
    a KL-vs-uniform regularizer encouraging all points to be used.

Set learnable=True to make the constellation points trainable parameters
(re-normalized to fixed power after each forward use); default is a fixed
square-QAM grid.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_square_qam(order: int, power: float = 1.0) -> torch.Tensor:
    """
    Build a square M-QAM constellation, normalized to a given AVERAGE symbol power.

    Returns a (M, 2) tensor of (I, Q) coordinates.
    `order` must be a perfect square (4, 16, 64, 256, ...).
    """
    levels_per_axis = int(round(math.sqrt(order)))
    if levels_per_axis ** 2 != order:
        raise ValueError(f"QAM order {order} is not a perfect square")

    # symmetric odd-integer PAM levels: e.g. L=4 -> [-3, -1, 1, 3]
    axis = torch.arange(levels_per_axis, dtype=torch.float32)
    axis = 2 * axis - (levels_per_axis - 1)  # center around 0

    I, Q = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([I.reshape(-1), Q.reshape(-1)], dim=1)  # (M, 2)

    # normalize to target average power: mean(|c|^2) == power
    avg_power = points.pow(2).sum(dim=1).mean()
    points = points * math.sqrt(power / avg_power.item())
    return points


class ConstellationQuantizer(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int,
        order: int = 16,
        power: float = 1.0,
        sigma_init: float = 5.0,
        sigma_max: float = 100.0,
        anneal_rate: float = 5.0 / 10000.0,
        learnable: bool = False,
    ):
        super().__init__()
        if bottleneck_dim % 2 != 0:
            raise ValueError("bottleneck_dim must be even for I/Q pairing")
        self.bottleneck_dim = bottleneck_dim
        self.num_symbols = bottleneck_dim // 2
        self.order = order
        self.power = power
        self.sigma_max = sigma_max
        self.anneal_rate = anneal_rate
        self.learnable = learnable

        self.sigma_init = float(sigma_init)

        points = build_square_qam(order, power)  # (M, 2)
        if learnable:
            self.constellation = nn.Parameter(points.clone())
        else:
            self.register_buffer("constellation", points)

        # hardness sigma_q and a step counter, kept as buffers so they persist
        self.register_buffer("sigma_q", torch.tensor(float(sigma_init)))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

        # most recent empirical usage distribution P_hat(C), shape (M,)
        self.register_buffer("last_usage", torch.full((order,), 1.0 / order))

    # ---- hardness annealing (call once per optimizer step) ----
    def step_sigma(self):
        self.step_count += 1
        t = self.step_count.item()
        new_sigma = min(self.sigma_max, self.sigma_init + self.anneal_rate * t)
        self.sigma_q.fill_(new_sigma)

    # ---- power re-normalization for the learnable case ----
    def _renormalize_constellation(self):
        with torch.no_grad():
            pts = self.constellation
            avg_power = pts.pow(2).sum(dim=1).mean().clamp_min(1e-8)
            scale = math.sqrt(self.power) / avg_power.sqrt()
            self.constellation.mul_(scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch, bottleneck_dim) real, power-normalized.
        returns quantized tensor of identical shape (forward = hard symbols,
        gradient = soft assignment).
        """
        if self.learnable and self.training:
            self._renormalize_constellation()

        batch = z.shape[0]
        # reshape to (batch, num_symbols, 2) -> I/Q pairs
        sym = z.view(batch, self.num_symbols, 2)

        # squared distances to every constellation point
        # sym:  (batch, num_symbols, 1, 2)
        # cons: (1,     1,           M, 2)
        cons = self.constellation.view(1, 1, self.order, 2)
        diff = sym.unsqueeze(2) - cons                    # (batch, num_symbols, M, 2)
        dist2 = diff.pow(2).sum(dim=-1)                   # (batch, num_symbols, M)

        # soft assignment: softmax over -sigma * distance
        weights = F.softmax(-self.sigma_q * dist2, dim=-1)  # (batch, num_symbols, M)
        soft = weights @ self.constellation                 # (batch, num_symbols, 2)

        # hard assignment: nearest point
        idx = dist2.argmin(dim=-1)                           # (batch, num_symbols)
        hard = self.constellation[idx]                       # (batch, num_symbols, 2)

        # straight-through: forward uses hard, backward uses soft gradient
        out = hard + (soft - soft.detach())

        # track empirical usage P_hat(C) = average soft weight per point.
        # Keep a graph-connected copy for the KL loss (so its gradient actually
        # reaches the encoder/constellation) and a detached copy for logging.
        usage = weights.mean(dim=(0, 1))         # (M,) -- still attached to the graph
        self._usage_for_loss = usage
        self.last_usage = usage.detach()         # buffer, for inspection/checkpointing

        return out.reshape(batch, self.bottleneck_dim)

    def hard_quantize_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Return nearest constellation point index for each I/Q pair.
        z: (batch, bottleneck_dim) → returns (batch, num_symbols) int indices in [0, M-1].
        """
        batch = z.shape[0]
        sym = z.view(batch, self.num_symbols, 2)
        cons = self.constellation.view(1, 1, self.order, 2)
        diff = sym.unsqueeze(2) - cons
        dist2 = diff.pow(2).sum(dim=-1)       # (batch, num_symbols, M)
        return dist2.argmin(dim=-1)           # (batch, num_symbols)

    def kl_to_uniform(self) -> torch.Tensor:
        """KL( P_hat(C) || Uniform ) from the most recent forward pass.

        Uses the graph-connected usage tensor so the KL term contributes a real
        gradient; falls back to the detached buffer if no forward has run yet.
        """
        usage = getattr(self, "_usage_for_loss", None)
        if usage is None:
            usage = self.last_usage
        p = usage.clamp_min(1e-12)
        p = p / p.sum()
        uniform = torch.full_like(p, 1.0 / self.order)
        return (p * (p / uniform).log()).sum()


if __name__ == "__main__":
    # standalone smoke test (no other modules required)
    torch.manual_seed(0)
    D = 64
    q = ConstellationQuantizer(bottleneck_dim=D, order=16, power=1.0)

    z = torch.randn(32, D, requires_grad=True)
    out = q(z)

    print("output shape:", tuple(out.shape))

    # 1) every output symbol must land exactly on a constellation point
    sym = out.view(-1, 2)
    cons = q.constellation
    d = torch.cdist(sym, cons)            # (num_sym, M)
    on_grid = d.min(dim=1).values.max().item()
    print(f"max distance of any output symbol to nearest grid point: {on_grid:.2e}  (should be ~0)")

    # 2) gradient must flow (straight-through)
    out.sum().backward()
    print("z.grad is not None:", z.grad is not None)
    print("z.grad finite:", bool(torch.isfinite(z.grad).all()))
    print("z.grad nonzero:", bool((z.grad != 0).any()))

    # 3) average output power ~ configured power
    avg_p = out.view(-1, 2).pow(2).sum(dim=1).mean().item()
    print(f"average output symbol power: {avg_p:.4f}  (target {q.power})")

    # 4) KL term is finite and non-negative
    print("KL to uniform:", q.kl_to_uniform().item())

    # 5) annealing increments sigma
    s0 = q.sigma_q.item()
    for _ in range(100):
        q.step_sigma()
    print(f"sigma_q before={s0}  after 100 steps={q.sigma_q.item():.4f}")
