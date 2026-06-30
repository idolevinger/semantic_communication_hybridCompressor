"""
dim_sweep.py
Trains a separate semantic pipeline model for different bottleneck dimensions (compression levels),
and evaluates the accuracy at a fixed clean SNR (20 dB) to show the effect of aggressive compression.
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
from snr_sweep import evaluate, _load_test_loader

def main():
    parser = argparse.ArgumentParser(description="Sweep training over varying bottleneck dimensions.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 64, 32, 16, 8, 4, 2], help="Bottleneck dimensions to sweep")
    parser.add_argument("--snr", type=float, default=10.0, help="Fixed SNR to use (default: 10dB)")
    parser.add_argument("--use_existing", type=int, choices=[0, 1], default=1, help="1 to use existing checkpoints, 0 to force retrain")
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Use a clean SNR (e.g. 20 dB) to isolate the effect of compression from noise
    noise_std = math.sqrt(config.QAM_POWER / (10 ** (args.snr / 10)))
    loader = _load_test_loader()

    # Sort decreasingly: 128, 64, 32, 16, 8
    dims = sorted(args.dims, reverse=True) 
    overall_accs = []
    per_class_accs = []

    for dim in dims:
        print(f"\n{'='*50}\n[INFO] Starting sweep for Bottleneck Dimension: {dim}\n{'='*50}")
        out_dir = os.path.join(config.RESULTS_DIR, "dim_sweep", f"snr_{int(args.snr)}")
        os.makedirs(out_dir, exist_ok=True)
        checkpoint_path = os.path.join(out_dir, f"model_dim_{dim}.pt")
        
        # 1. Train if needed
        if not os.path.exists(checkpoint_path) or args.use_existing == 0:
            print(f"[INFO] Training model with bottleneck={dim}...")
            train.main(
                noise_apply=True,
                bottleneck_dim=dim,
                save_path=checkpoint_path,
                noise_std=noise_std
            )
        else:
            print(f"[INFO] Found existing checkpoint for dim={dim}. Skipping training.")

        # 2. Evaluate
        print(f"[INFO] Evaluating model with bottleneck={dim}...")
        total_acc, per_acc = evaluate(checkpoint_path, dim, noise_std, device, loader)
        overall_accs.append(total_acc)
        per_class_accs.append(per_acc)
        print(f"[RESULT] Dim {dim} Accuracy: {total_acc * 100:.2f}%")

    # 3. Plotting
    plot_path = os.path.join(out_dir, "dim_sweep.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Strings so matplotlib treats them as categorical variables maintaining our sorted order
    x_labels = [str(d) for d in dims]

    # Overall accuracy
    ax1.plot(x_labels, [v * 100 for v in overall_accs], marker="s", linewidth=2, color="purple")
    
    arrow_props = dict(arrowstyle="->", color='purple', shrinkA=0, shrinkB=5)
    for i, val in enumerate(overall_accs):
        y_offset = 25 if (val * 100) < 40 else -25
        ax1.annotate(f"{val*100:.1f}%", (x_labels[i], val*100), textcoords="offset points", xytext=(0, y_offset), ha='center', fontsize=9, fontweight='bold', color='purple', arrowprops=arrow_props)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Impact of Semantic Compression (SNR = {args.snr} dB)")
    ax1.set_ylim([0, 100]) # Absolute percentage scale
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.05, 'Less Compression ->', transform=ax1.transAxes, color='gray', fontsize=10)
    ax1.text(0.75, 0.05, '-> More Aggressive', transform=ax1.transAxes, color='gray', fontsize=10)

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

    ax2.set_xlabel("Bottleneck Dimension (Compression Level)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 100]) # Absolute percentage scale
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"\n[DONE] Plot saved -> {plot_path}")

if __name__ == "__main__":
    main()
