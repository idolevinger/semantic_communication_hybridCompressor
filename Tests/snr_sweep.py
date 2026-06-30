"""
snr_sweep.py
Trains a separate semantic pipeline model for different SNR values,
and evaluates the accuracy to show the effect of noise at a fixed compression level.
"""
import argparse
import os
import math
import sys

# Ensure imports work from the Tests directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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

@torch.no_grad()
def evaluate(checkpoint: str, bottleneck_dim: int, noise_std: float, device, loader: DataLoader) -> tuple:
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
    correct = torch.zeros(config.NUM_CLASSES, dtype=torch.long)
    count = torch.zeros(config.NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = pipeline(x).argmax(1)
        for c in range(config.NUM_CLASSES):
            mask = y == c
            correct[c] += (preds[mask] == c).sum().item()
            count[c] += mask.sum().item()
    per_class = (correct.float() / count.float().clamp(min=1)).tolist()
    overall = (correct.sum().float() / count.sum().float()).item()
    return overall, per_class

def main():
    parser = argparse.ArgumentParser(description="Sweep training over varying SNR values.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--snrs", type=float, nargs="+", default=[20.0, 10.0, 0.0, -10.0], help="SNR values to sweep")
    parser.add_argument("--dim", type=int, default=64, help="Fixed Bottleneck dimension to use (default: 64)")
    parser.add_argument("--use_existing", type=int, choices=[0, 1], default=1, help="1 to use existing checkpoints, 0 to force retrain")
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    loader = _load_test_loader()

    # Sort decreasingly: 20, 10, 0, -10
    snrs = sorted(args.snrs, reverse=True) 
    overall_accs = []
    per_class_accs = []

    out_dir = os.path.join(config.RESULTS_DIR, "snr_sweep", f"dim_{args.dim}")
    os.makedirs(out_dir, exist_ok=True)

    for snr in snrs:
        print(f"\n{'='*50}\n[INFO] Starting sweep for SNR: {snr} dB\n{'='*50}")
        checkpoint_path = os.path.join(out_dir, f"model_snr_{int(snr)}.pt")
        noise_std = math.sqrt(config.QAM_POWER / (10 ** (snr / 10)))
        
        # 1. Train if needed
        if not os.path.exists(checkpoint_path) or args.use_existing == 0:
            print(f"[INFO] Training model with SNR={snr} dB...")
            train.main(
                noise_apply=True,
                bottleneck_dim=args.dim,
                save_path=checkpoint_path,
                noise_std=noise_std
            )
        else:
            print(f"[INFO] Found existing checkpoint for SNR={snr} dB. Skipping training.")

        # 2. Evaluate
        print(f"[INFO] Evaluating model with SNR={snr} dB...")
        total_acc, per_acc = evaluate(checkpoint_path, args.dim, noise_std, device, loader)
        overall_accs.append(total_acc)
        per_class_accs.append(per_acc)
        print(f"[RESULT] SNR {snr} dB Accuracy: {total_acc * 100:.2f}%")

    # 3. Plotting
    plot_path = os.path.join(out_dir, "snr_sweep.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    x_labels = [str(s) for s in snrs]

    # Overall accuracy
    ax1.plot(x_labels, [v * 100 for v in overall_accs], marker="s", linewidth=2, color="crimson")
    
    arrow_props = dict(arrowstyle="->", color='crimson', shrinkA=0, shrinkB=5)
    for i, val in enumerate(overall_accs):
        y_offset = 25 if (val * 100) < 40 else -25
        ax1.annotate(f"{val*100:.1f}%", (x_labels[i], val*100), textcoords="offset points", xytext=(0, y_offset), ha='center', fontsize=9, fontweight='bold', color='crimson', arrowprops=arrow_props)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Impact of Channel Noise (Bottleneck Dim = {args.dim})")
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.05, 'Cleaner Channel ->', transform=ax1.transAxes, color='gray', fontsize=10)
    ax1.text(0.75, 0.05, '-> Noisier Channel', transform=ax1.transAxes, color='gray', fontsize=10)

    # Per-class accuracy
    colors = ["#2f6fd6", "#2ca25f", "#f28e2b", "#d62728"]
    for idx, name in enumerate(config.CATEGORY_NAMES):
        ax2.plot(
            x_labels,
            [row[idx] * 100 for row in per_class_accs],
            marker="o",
            linewidth=1.8,
            label=name,
            color=colors[idx % len(colors)],
        )

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"\n[DONE] Plot saved -> {plot_path}")

if __name__ == "__main__":
    main()
