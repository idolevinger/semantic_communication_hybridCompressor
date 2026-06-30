"""
main.py
Runs the full joint-vs-separate experiment in three steps:

  1. Train Option A (joint source-channel coding):
       BOTTLENECK_DIM=64, trained WITH AWGN noise, no LDPC.
       The encoder learns noise-robust representations directly.

  2. Train Option B (separate source + channel coding):
       BOTTLENECK_DIM=32, trained WITHOUT noise (clean channel).
       The encoder focuses on compression only; LDPC handles
       error protection at test time (separation approach).

  3. SNR sweep:
       Evaluates both checkpoints across SNR -6–20 dB and plots
       three curves: A (joint), B (separate + LDPC), C (baseline).

Each training run saves to its own checkpoint file — no manual
config editing or cp commands needed.
"""

import os

import train
import snr_sweep
from config import RESULTS_DIR

CHECKPOINT_A = os.path.join(RESULTS_DIR, "model_joint_64.pt")
CHECKPOINT_B = os.path.join(RESULTS_DIR, "model_separate_32.pt")


def main():
    print("=" * 70)
    print("OPTION A: Joint source-channel coding")
    print("  BOTTLENECK_DIM=64 | training WITH noise | no LDPC")
    print("=" * 70)
    train.main(noise_apply=True, bottleneck_dim=64, save_path=CHECKPOINT_A)

    print()
    print("=" * 70)
    print("OPTION B: Separate source + channel coding")
    print("  BOTTLENECK_DIM=32 | training WITHOUT noise | LDPC at test")
    print("=" * 70)
    train.main(noise_apply=False, bottleneck_dim=32, save_path=CHECKPOINT_B)

    print()
    print("=" * 70)
    print("SNR SWEEP: Option A vs Option B vs Baseline C")
    print("=" * 70)
    snr_sweep.main()


if __name__ == "__main__":
    main()
