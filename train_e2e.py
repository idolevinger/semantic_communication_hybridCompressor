"""
train_e2e.py
End-to-End Training for Semantic Compression (Alternative Approach).

Unlike `train.py` which forces unsupervised compression, this script trains the
Autoencoder, Quantizer, and TaskDecoder in a SINGLE phase. The classification
loss (CrossEntropy) directly updates the Autoencoder's weights, causing the 
bottleneck to learn task-specific features rather than pure semantics.

WARNING: This violates the project's core methodology of unsupervised semantic 
compression, but is provided as an additional baseline/option.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pipeline import build_pipeline
from config import (
    CACHE_EMBEDDINGS_FILE,
    MODEL_FILE,
    BOTTLENECK_DIM,
    BATCH_SIZE,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    KL_LAMBDA,
    LABEL_SMOOTHING,
    CLASS_WEIGHTS,
    NOISE_APPLY_TRAIN,
    RECON_LOSS_WEIGHT,
)

def load_datasets():
    data = torch.load(CACHE_EMBEDDINGS_FILE)
    train_ds = TensorDataset(data["train"]["embeddings"], data["train"]["labels"])
    val_ds = TensorDataset(data["val"]["embeddings"], data["val"]["labels"])
    return train_ds, val_ds

@torch.no_grad()
def evaluate_e2e(pipeline, loader, device, ce_criterion, mse_criterion):
    pipeline.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, x_hat = pipeline.forward_train(x)
        
        loss = ce_criterion(logits, y) + RECON_LOSS_WEIGHT * mse_criterion(x_hat, x)
        
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    pipeline.train()
    return loss_sum / total, correct / total

def main(noise_apply: bool = None, bottleneck_dim: int = None, save_path: str = None, noise_std: float = None):
    if noise_apply is None:
        noise_apply = NOISE_APPLY_TRAIN
    _bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else BOTTLENECK_DIM
    _save_path = save_path if save_path is not None else MODEL_FILE.replace('.pt', '_e2e.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] BOTTLENECK_DIM={_bottleneck_dim}")
    if noise_std is not None:
        print(f"[INFO] Training WITH channel noise std={noise_std}")
    else:
        print(f"[INFO] Training {'WITH' if noise_apply else 'WITHOUT'} channel noise")

    train_ds, val_ds = load_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    pipeline = build_pipeline(
        use_noise=noise_apply, device=device, bottleneck_dim=_bottleneck_dim, noise_std=noise_std
    )
    pipeline.train()

    print("\n--- End-to-End Training (AE + Quantizer + Classifier) ---")
    all_params = list(pipeline.autoencoder.parameters()) + list(pipeline.quantizer.parameters()) + list(pipeline.decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR)
    
    mse_criterion = nn.MSELoss()
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            
            logits, x_hat = pipeline.forward_train(x)
            
            mse_loss = mse_criterion(x_hat, x)
            kl_loss = pipeline.quantizer.kl_to_uniform()
            ce_loss = ce_criterion(logits, y)
            
            loss = ce_loss + RECON_LOSS_WEIGHT * mse_loss + KL_LAMBDA * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pipeline.quantizer.step_sigma()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_e2e(pipeline, val_loader, device, ce_criterion, mse_criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | sigma_q {pipeline.quantizer.sigma_q.item():.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            pipeline.save_checkpoint(_save_path, epoch=epoch, val_acc=val_acc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping after {epoch+1} epochs")
                break

    print(f"[DONE] Best val_acc={best_val_acc:.4f}. Model -> {_save_path}")

    # --- Print Summary (like test.py) ---
    print("\n[INFO] Loading best model for final evaluation on Test set...")
    pipeline.load_checkpoint(_save_path, device=device)
    
    from test import evaluate_detailed, print_report, load_test_dataset
    from config import NUM_CLASSES
    test_loader = DataLoader(load_test_dataset(), batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate_detailed(pipeline, test_loader, device, NUM_CLASSES)
    print_report(*results)

if __name__ == "__main__":
    main()
