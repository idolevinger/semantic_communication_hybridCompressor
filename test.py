"""
test.py
Evaluate a trained pipeline on cached BERT embeddings.
Prints overall and per-class accuracy + loss. Noise controlled via config / argument.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline import build_pipeline
from config import (
    CACHE_EMBEDDINGS_FILE,
    MODEL_FILE,
    BATCH_SIZE,
    NUM_CLASSES,
    CATEGORY_NAMES,
    NOISE_APPLY_EVAL,
    NOISE_STD,
    snr_db_from_std,
    BOTTLENECK_DIM,
)


def load_test_dataset():
    data = torch.load(CACHE_EMBEDDINGS_FILE)
    return TensorDataset(data["test"]["embeddings"], data["test"]["labels"])


@torch.no_grad()
def evaluate_detailed(pipeline, loader, device, num_classes):
    pipeline.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss, total_correct, total_count = 0.0, 0, 0
    class_loss = torch.zeros(num_classes, dtype=torch.float64)
    class_correct = torch.zeros(num_classes, dtype=torch.int64)
    class_count = torch.zeros(num_classes, dtype=torch.int64)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = pipeline(x)
        losses = criterion(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += losses.sum().item()
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                class_loss[c] += losses[mask].sum().item()
                class_correct[c] += (preds[mask] == y[mask]).sum().item()
                class_count[c] += mask.sum().item()

    overall_loss = total_loss / max(1, total_count)
    overall_acc = total_correct / max(1, total_count)
    safe = class_count.clamp(min=1).to(torch.float64)
    return (
        overall_loss,
        overall_acc,
        (class_loss / safe).tolist(),
        (class_correct.to(torch.float64) / safe).tolist(),
        class_count.tolist(),
    )


def print_report(overall_loss, overall_acc, per_loss, per_acc, per_support):
    print("\n" + "=" * 72)
    print(f"[OVERALL] Loss: {overall_loss:.4f} | Accuracy: {overall_acc:.4f}")
    print("=" * 72)
    print(f"{'Label':<10} {'Support':>10} {'Acc':>10} {'Loss':>12}")
    print("-" * 72)
    for i in range(len(per_loss)):
        name = CATEGORY_NAMES[i] if i < len(CATEGORY_NAMES) else f"Class_{i}"
        print(f"{name:<10} {per_support[i]:>10} {per_acc[i]:>10.4f} {per_loss[i]:>12.4f}")
    print("=" * 72 + "\n")


def main(noise_apply: bool = None, bottleneck_dim: int = None):
    if noise_apply is None:
        noise_apply = NOISE_APPLY_EVAL
    _bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else BOTTLENECK_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print(f"[INFO] BOTTLENECK_DIM={_bottleneck_dim}")
    if noise_apply:
        print(f"[INFO] Testing WITH noise | std={NOISE_STD} (~{snr_db_from_std(NOISE_STD):.1f} dB SNR)")
    else:
        print("[INFO] Testing WITHOUT noise")

    pipeline = build_pipeline(
        use_noise=noise_apply,
        device=device,
        checkpoint_path=MODEL_FILE,
        bottleneck_dim=_bottleneck_dim,
    )

    test_loader = DataLoader(load_test_dataset(), batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate_detailed(pipeline, test_loader, device, NUM_CLASSES)
    print_report(*results)


if __name__ == "__main__":
    main()
