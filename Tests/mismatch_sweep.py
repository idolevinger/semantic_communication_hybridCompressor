"""
mismatch_sweep.py
Trains a single semantic pipeline model at a fixed SNR (e.g. 10 dB) and Bottleneck (64),
and evaluates its accuracy across varying channel SNRs (-10 to 20 dB).
This shows the effect of SNR mismatch between training and deployment.
"""
import argparse
import os
import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import train
from snr_sweep import evaluate, _load_test_loader

def main():
    parser = argparse.ArgumentParser(description="Sweep evaluation SNR for a fixed trained model.")
    parser.add_argument("--train_snr", type=float, default=10.0, help="SNR used for training (default: 10dB)")
    parser.add_argument("--eval_snrs", type=float, nargs="+", default=[20.0, 10.0, 0.0, -10.0], help="SNR values for evaluation")
    parser.add_argument("--dim", type=int, default=64, help="Fixed Bottleneck dimension to use (default: 64)")
    parser.add_argument("--use_existing", type=int, choices=[0, 1], default=1, help="1 to use existing checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    loader = _load_test_loader()
    
    out_dir = os.path.join(config.RESULTS_DIR, "mismatch_sweep", f"train_snr_{int(args.train_snr)}_dim_{args.dim}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Train ONE model at train_snr
    checkpoint_path = os.path.join(out_dir, f"model_train_snr_{int(args.train_snr)}.pt")
    train_noise_std = math.sqrt(config.QAM_POWER / (10 ** (args.train_snr / 10)))
    
    if not os.path.exists(checkpoint_path) or args.use_existing == 0:
        print(f"[INFO] Training BASE model with SNR={args.train_snr} dB...")
        train.main(
            noise_apply=True,
            bottleneck_dim=args.dim,
            save_path=checkpoint_path,
            noise_std=train_noise_std
        )
    else:
        print(f"[INFO] Found existing base model for SNR={args.train_snr} dB.")

    # 2. Evaluate across all eval_snrs
    snrs = sorted(args.eval_snrs, reverse=True) 
    overall_accs = []
    per_class_accs = []

    for snr in snrs:
        print(f"\n{'='*50}\n[INFO] Evaluating at Test SNR: {snr} dB\n{'='*50}")
        eval_noise_std = math.sqrt(config.QAM_POWER / (10 ** (snr / 10)))
        total_acc, per_acc = evaluate(checkpoint_path, args.dim, eval_noise_std, device, loader)
        overall_accs.append(total_acc)
        per_class_accs.append(per_acc)
        print(f"[RESULT] Eval SNR {snr} dB Accuracy: {total_acc * 100:.2f}%")

    # 3. Plotting
    plot_path = os.path.join(out_dir, "mismatch_sweep.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    x_labels = [str(s) for s in snrs]

    # Overall accuracy
    ax1.plot(x_labels, [v * 100 for v in overall_accs], marker="s", linewidth=2, color="teal")
    
    arrow_props = dict(arrowstyle="->", color='teal', shrinkA=0, shrinkB=5)
    for i, val in enumerate(overall_accs):
        y_offset = 25 if (val * 100) < 40 else -25
        ax1.annotate(f"{val*100:.1f}%", (x_labels[i], val*100), textcoords="offset points", xytext=(0, y_offset), ha='center', fontsize=9, fontweight='bold', color='teal', arrowprops=arrow_props)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Impact of SNR Mismatch (Trained @ {args.train_snr}dB, Dim = {args.dim})")
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3)
    
    # Mark the training point with a vertical line
    if str(args.train_snr) in x_labels:
        ax1.axvline(x=x_labels.index(str(args.train_snr)), color='gray', linestyle='--', alpha=0.5, label="Training Point")
        ax1.legend(loc="upper right")

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

    ax2.set_xlabel("Evaluation SNR (dB)")
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
