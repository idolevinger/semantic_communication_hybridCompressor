"""
ldpc_test.py
Head-to-head comparison of three source-channel coding strategies at a single,
fixed channel noise level (the "same as the main experiment", i.e. config.NOISE_STD
by default). This is the direct test of joint vs. separate source-channel coding.

    Option A — JOINT (D=64)
        Trained WITH channel noise. Evaluated through the noisy channel, NO LDPC.
        The encoder itself learned noise-robust representations.

    Option B — SEPARATE + LDPC (D=32)
        Trained on a CLEAN channel (no noise) — the encoder only learns to
        compress. Evaluated through the noisy channel WITH rate-1/2 LDPC error
        correction bolted on at test time (the classical separation approach).

    Option C — BASELINE (D=32)
        The SAME clean-trained checkpoint as Option B, but evaluated through the
        noisy channel WITHOUT LDPC. Isolates how much work the LDPC is doing:
        a clean-trained encoder with no error correction should degrade fast.

Bandwidth is matched: Option A transmits 64/2 = 32 QAM symbols; Option B's 32
dims give 16 symbols -> rate-1/2 LDPC doubles the bits back to 32 symbols. Both
put the same number of real dimensions on the wire, so the comparison is fair.

Usage:
    python Tests/ldpc_test.py                 # noise = config.NOISE_STD (~10.5 dB)
    python Tests/ldpc_test.py --snr 6         # fix the channel at 6 dB instead
    python Tests/ldpc_test.py --epochs 20     # cap training epochs per phase
    python Tests/ldpc_test.py --use_existing 0  # force retrain both models
"""
import argparse
import math
import os
import sys

# Ensure imports resolve when run from the Tests/ directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import config
import train


DIM_JOINT = 64      # Option A bottleneck (joint, trained with noise)
DIM_SEPARATE = 32   # Option B/C bottleneck (separate, trained clean)


def _load_test_loader() -> DataLoader:
    data = torch.load(config.CACHE_EMBEDDINGS_FILE, map_location="cpu", weights_only=False)
    ds = TensorDataset(data["test"]["embeddings"], data["test"]["labels"])
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)


@torch.no_grad()
def evaluate(checkpoint: str, bottleneck_dim: int, noise_std: float,
             use_ldpc: bool, device, loader: DataLoader) -> tuple:
    """Evaluate one checkpoint through a noisy channel, with or without LDPC.

    Returns (overall_accuracy, [per_class_accuracy, ...]).
    """
    from pipeline import build_pipeline
    pipeline = build_pipeline(
        use_noise=True,
        device=device,
        checkpoint_path=checkpoint,
        bottleneck_dim=bottleneck_dim,
        use_ldpc=use_ldpc,
        noise_std=noise_std,
    )
    pipeline.eval()  # required: the LDPC branch only runs when not training

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


def _plot(results, noise_std, snr_db, out_dir):
    """results: list of dicts with keys name, overall, per_class, color."""
    plot_path = os.path.join(out_dir, "ldpc_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["name"] for r in results]
    colors = [r["color"] for r in results]

    # ---- left: overall accuracy bars ----
    overalls = [r["overall"] * 100 for r in results]
    bars = ax1.bar(names, overalls, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, overalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%",
                 ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Overall Accuracy (%)")
    ax1.set_ylim([0, 100])
    ax1.set_title(f"Joint vs. Separate+LDPC vs. Baseline\n"
                  f"Channel: noise_std={noise_std:.3f}  (SNR ≈ {snr_db:.1f} dB)")
    ax1.grid(True, axis="y", alpha=0.3)

    # ---- right: per-class grouped bars ----
    n_opt = len(results)
    n_cls = config.NUM_CLASSES
    width = 0.8 / n_opt
    x = list(range(n_cls))
    for i, r in enumerate(results):
        offsets = [c + (i - (n_opt - 1) / 2) * width for c in x]
        ax2.bar(offsets, [v * 100 for v in r["per_class"]], width=width,
                label=r["name"], color=r["color"], edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(config.CATEGORY_NAMES)
    ax2.set_ylabel("Per-Class Accuracy (%)")
    ax2.set_ylim([0, 100])
    ax2.set_title("Per-Class Breakdown")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare Joint vs. Separate+LDPC vs. Baseline at a fixed SNR.")
    parser.add_argument("--snr", type=float, default=None,
                        help="Channel SNR in dB. Default: derived from config.NOISE_STD.")
    parser.add_argument("--noise_std", type=float, default=None,
                        help="Channel noise std (overrides --snr). Default: config.NOISE_STD.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Cap training epochs per phase.")
    parser.add_argument("--use_existing", type=int, choices=[0, 1], default=1,
                        help="1 = reuse existing checkpoints, 0 = force retrain.")
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    # Resolve the single channel noise level used for ALL three evaluations.
    if args.noise_std is not None:
        noise_std = args.noise_std
    elif args.snr is not None:
        noise_std = math.sqrt(config.QAM_POWER / (10 ** (args.snr / 10)))
    else:
        noise_std = config.NOISE_STD
    snr_db = config.snr_db_from_std(noise_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Channel: noise_std={noise_std:.4f}  (SNR ≈ {snr_db:.2f} dB)")

    out_dir = os.path.join(config.RESULTS_DIR, "ldpc_test")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_joint = os.path.join(out_dir, f"model_joint_{DIM_JOINT}.pt")
    ckpt_separate = os.path.join(out_dir, f"model_separate_{DIM_SEPARATE}.pt")

    loader = _load_test_loader()

    # ---------------------------------------------------------------
    # 1. Train Option A — JOINT, D=64, WITH noise
    # ---------------------------------------------------------------
    print(f"\n{'='*60}\nOPTION A — JOINT  (D={DIM_JOINT}, trained WITH noise)\n{'='*60}")
    if not os.path.exists(ckpt_joint) or args.use_existing == 0:
        train.main(noise_apply=True, bottleneck_dim=DIM_JOINT,
                   save_path=ckpt_joint, noise_std=noise_std)
    else:
        print(f"[INFO] Reusing existing checkpoint: {ckpt_joint}")

    # ---------------------------------------------------------------
    # 2. Train Option B/C — SEPARATE, D=32, WITHOUT noise (clean)
    #    One clean-trained model serves both B (with LDPC) and C (without).
    # ---------------------------------------------------------------
    print(f"\n{'='*60}\nOPTION B/C — SEPARATE  (D={DIM_SEPARATE}, trained on CLEAN channel)\n{'='*60}")
    if not os.path.exists(ckpt_separate) or args.use_existing == 0:
        train.main(noise_apply=False, bottleneck_dim=DIM_SEPARATE,
                   save_path=ckpt_separate)
    else:
        print(f"[INFO] Reusing existing checkpoint: {ckpt_separate}")

    # ---------------------------------------------------------------
    # 3. Evaluate all three through the SAME noisy channel
    # ---------------------------------------------------------------
    print(f"\n{'='*60}\nEVALUATION  (all through noise_std={noise_std:.4f})\n{'='*60}")

    print("[INFO] A: joint, no LDPC ...")
    a_overall, a_pc = evaluate(ckpt_joint, DIM_JOINT, noise_std,
                               use_ldpc=False, device=device, loader=loader)
    print(f"[RESULT] A (joint)          : {a_overall*100:.2f}%")

    print("[INFO] B: separate + LDPC ... (belief propagation, may take a while)")
    b_overall, b_pc = evaluate(ckpt_separate, DIM_SEPARATE, noise_std,
                               use_ldpc=True, device=device, loader=loader)
    print(f"[RESULT] B (separate + LDPC): {b_overall*100:.2f}%")

    print("[INFO] C: separate, no LDPC (baseline) ...")
    c_overall, c_pc = evaluate(ckpt_separate, DIM_SEPARATE, noise_std,
                               use_ldpc=False, device=device, loader=loader)
    print(f"[RESULT] C (baseline)       : {c_overall*100:.2f}%")

    # ---------------------------------------------------------------
    # 4. Report + plot
    # ---------------------------------------------------------------
    results = [
        {"name": f"A: Joint (D={DIM_JOINT})",          "overall": a_overall, "per_class": a_pc, "color": "#2f6fd6"},
        {"name": f"B: Separate+LDPC (D={DIM_SEPARATE})", "overall": b_overall, "per_class": b_pc, "color": "#2ca25f"},
        {"name": f"C: Baseline (D={DIM_SEPARATE})",      "overall": c_overall, "per_class": c_pc, "color": "#d62728"},
    ]

    print(f"\n{'='*60}\nSUMMARY  (SNR ≈ {snr_db:.2f} dB)\n{'='*60}")
    print(f"{'Option':<28}{'Overall':>10}")
    for r in results:
        print(f"{r['name']:<28}{r['overall']*100:>9.2f}%")
    print(f"\n  LDPC gain (B - C): {(b_overall - c_overall)*100:+.2f} pts")
    print(f"  Joint vs LDPC (A - B): {(a_overall - b_overall)*100:+.2f} pts")

    plot_path = _plot(results, noise_std, snr_db, out_dir)
    print(f"\n[DONE] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
