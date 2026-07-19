"""
mse_sweep.py
Measures the semantic reconstruction distortion (MSE between the input BERT
embedding x and the recovered embedding x_hat) across a 2-D grid of:
    * compression level  -> bottleneck dimension  (fewer dims = harder compression)
    * channel quality    -> SNR in dB             (lower SNR = noisier channel)

For every (dim, SNR) cell it trains a model (or reuses an existing checkpoint),
evaluates the test-set MSE, and produces a graph of MSE vs SNR with one line per
compression level. A companion heatmap gives the full grid at a glance.

Run from the project root or the Tests directory, e.g.:
    python Tests/mse_sweep.py --dims 64 32 16 --snrs 20 10 0 -10 --epochs 10
"""
import argparse
import os
import math
import sys

# Ensure imports work from the Tests directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import train
from torch.utils.data import DataLoader, TensorDataset


def _load_test_loader() -> DataLoader:
    data = torch.load(config.CACHE_EMBEDDINGS_FILE, map_location="cpu", weights_only=False)
    ds = TensorDataset(data["test"]["embeddings"], data["test"]["labels"])
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)


def _snr_to_std(snr_db: float) -> float:
    """AWGN std for a target SNR (dB) at the constellation's average symbol power."""
    return math.sqrt(config.QAM_POWER / (10 ** (snr_db / 10)))


@torch.no_grad()
def evaluate_mse(checkpoint: str, bottleneck_dim: int, noise_std: float, device, loader: DataLoader) -> float:
    """Test-set MSE between the input embedding and its channel-corrupted reconstruction."""
    from pipeline import build_pipeline
    pipeline = build_pipeline(
        use_noise=True,
        device=device,
        checkpoint_path=checkpoint,
        bottleneck_dim=bottleneck_dim,
        use_ldpc=False,
        noise_std=noise_std,
    )
    pipeline.eval()
    mse_criterion = nn.MSELoss(reduction="sum")
    sq_err_sum, elem_count = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        _, x_hat = pipeline.forward_with_recon(x)
        sq_err_sum += mse_criterion(x_hat, x).item()
        elem_count += x.numel()
    return sq_err_sum / elem_count


def main():
    parser = argparse.ArgumentParser(
        description="Sweep reconstruction MSE over compression level x SNR."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 32, 16, 8],
                        help="Bottleneck dimensions (compression levels) to sweep")
    parser.add_argument("--snrs", type=float, nargs="+", default=[20.0, 10.0, 0.0, -10.0],
                        help="SNR values in dB to sweep")
    parser.add_argument("--use_existing", type=int, choices=[0, 1], default=1,
                        help="1 to reuse existing checkpoints, 0 to force retrain")
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    loader = _load_test_loader()

    # Sort so the axes read naturally: dims high->low (light->heavy compression),
    # SNR high->low (clean->noisy).
    dims = sorted(args.dims, reverse=True)
    snrs = sorted(args.snrs, reverse=True)

    out_dir = os.path.join(config.RESULTS_DIR, "mse_sweep")
    os.makedirs(out_dir, exist_ok=True)

    # mse_grid[i][j] = MSE for dims[i] at snrs[j]
    mse_grid = [[None] * len(snrs) for _ in range(len(dims))]

    for i, dim in enumerate(dims):
        for j, snr in enumerate(snrs):
            print(f"\n{'='*60}\n[INFO] Cell dim={dim}, SNR={snr} dB\n{'='*60}")
            noise_std = _snr_to_std(snr)
            checkpoint_path = os.path.join(out_dir, f"model_dim_{dim}_snr_{int(snr)}.pt")

            # 1. Train if needed (each cell is trained at its own dim + noise level)
            if not os.path.exists(checkpoint_path) or args.use_existing == 0:
                print(f"[INFO] Training model dim={dim}, SNR={snr} dB (std={noise_std:.4f})...")
                train.main(
                    noise_apply=True,
                    bottleneck_dim=dim,
                    save_path=checkpoint_path,
                    noise_std=noise_std,
                )
            else:
                print(f"[INFO] Found existing checkpoint for dim={dim}, SNR={snr} dB. Skipping training.")

            # 2. Evaluate MSE
            mse = evaluate_mse(checkpoint_path, dim, noise_std, device, loader)
            mse_grid[i][j] = mse
            print(f"[RESULT] dim={dim}, SNR={snr} dB -> MSE = {mse:.6f}")

    # ---------------- Plotting ----------------
    x_labels = [str(s) for s in snrs]
    colors = ["#2f6fd6", "#2ca25f", "#f28e2b", "#d62728", "#9467bd", "#8c564b", "#17becf"]

    # (A) Line plot: MSE vs SNR, one line per compression level
    line_path = os.path.join(out_dir, "mse_sweep_lines.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, dim in enumerate(dims):
        ax.plot(
            x_labels,
            mse_grid[i],
            marker="o",
            linewidth=2,
            label=f"dim={dim}",
            color=colors[i % len(colors)],
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Semantic Reconstruction MSE vs Channel SNR and Compression Level")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Compression\n(bottleneck dim)", loc="upper right")
    ax.text(0.02, 0.02, "Cleaner Channel ->", transform=ax.transAxes, color="gray", fontsize=9)
    ax.text(0.72, 0.02, "-> Noisier Channel", transform=ax.transAxes, color="gray", fontsize=9)
    plt.tight_layout()
    plt.savefig(line_path)
    plt.close(fig)
    print(f"\n[DONE] Line plot saved -> {line_path}")

    # (B) Heatmap: full compression x SNR grid
    heat_path = os.path.join(out_dir, "mse_sweep_heatmap.png")
    fig, ax = plt.subplots(figsize=(1.6 * len(snrs) + 2, 1.0 * len(dims) + 2))
    grid = [[mse_grid[i][j] for j in range(len(snrs))] for i in range(len(dims))]
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([str(d) for d in dims])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bottleneck Dimension (Compression Level)")
    ax.set_title("Reconstruction MSE Grid")
    for i in range(len(dims)):
        for j in range(len(snrs)):
            ax.text(j, i, f"{grid[i][j]:.4f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()
    plt.savefig(heat_path)
    plt.close(fig)
    print(f"[DONE] Heatmap saved -> {heat_path}")

    # Console summary table
    print("\n[SUMMARY] Reconstruction MSE (rows=dim, cols=SNR dB)")
    header = "dim\\snr | " + " | ".join(f"{s:>8}" for s in snrs)
    print(header)
    print("-" * len(header))
    for i, dim in enumerate(dims):
        row = f"{dim:>7} | " + " | ".join(f"{mse_grid[i][j]:8.5f}" for j in range(len(snrs)))
        print(row)


if __name__ == "__main__":
    main()
