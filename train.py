"""
train.py
Two-Phase Training for Unsupervised Semantic Compression.

Phase 1: Train Autoencoder + Quantizer to minimize MSE + KL (no classification).
         This ensures the bottleneck compresses semantics purely, without label leakage.
Phase 2: Freeze AE, Train TaskDecoder (Classifier) on reconstructed embeddings.
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
)

def load_datasets():
    data = torch.load(CACHE_EMBEDDINGS_FILE)
    train_ds = TensorDataset(data["train"]["embeddings"], data["train"]["labels"])
    val_ds = TensorDataset(data["val"]["embeddings"], data["val"]["labels"])
    return train_ds, val_ds

@torch.no_grad()
def evaluate_ae(pipeline, loader, device):
    pipeline.eval()
    mse_criterion = nn.MSELoss()
    loss_sum, total = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        _, x_hat = pipeline.forward_train(x)
        loss = mse_criterion(x_hat, x)
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
    pipeline.train()
    return loss_sum / total

@torch.no_grad()
def evaluate_clf(pipeline, loader, device):
    pipeline.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = pipeline(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    pipeline.train()
    return loss_sum / total, correct / total

def main(noise_apply: bool = None, bottleneck_dim: int = None, save_path: str = None, noise_std: float = None):
    if noise_apply is None:
        noise_apply = NOISE_APPLY_TRAIN
    _bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else BOTTLENECK_DIM
    _save_path = save_path if save_path is not None else MODEL_FILE

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

    # PHASE 1: Unsupervised AE Training
    print("\n--- PHASE 1: Unsupervised Autoencoder Training ---")
    ae_params = list(pipeline.autoencoder.parameters()) + list(pipeline.quantizer.parameters())
    optimizer_ae = torch.optim.AdamW(ae_params, lr=LR)
    mse_criterion = nn.MSELoss()

    best_val_mse = float('inf')
    epochs_no_improve = 0
    AE_EPOCHS = EPOCHS

    for epoch in range(AE_EPOCHS):
        loss_sum, total = 0.0, 0
        for x, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{AE_EPOCHS}"):
            x = x.to(device)
            _, x_hat = pipeline.forward_train(x)
            
            mse_loss = mse_criterion(x_hat, x)
            kl_loss = pipeline.quantizer.kl_to_uniform()
            loss = mse_loss + KL_LAMBDA * kl_loss

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            pipeline.quantizer.step_sigma()

            loss_sum += mse_loss.item() * x.size(0)
            total += x.size(0)

        train_mse = loss_sum / total
        val_mse = evaluate_ae(pipeline, val_loader, device)

        print(f"AE Epoch {epoch+1}/{AE_EPOCHS} | Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f} | sigma_q {pipeline.quantizer.sigma_q.item():.2f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            pipeline.save_checkpoint(_save_path, epoch=epoch, val_acc=-val_mse) # negative so it doesn't look like accuracy
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] AE Early stopping after {epoch+1} epochs")
                break

    # Load best AE
    print(f"\n[INFO] Loading best AE model (val_mse={best_val_mse:.4f})")
    pipeline.load_checkpoint(_save_path, device=device)

    # PHASE 2: Classifier Training
    print("\n--- PHASE 2: Classifier Training (AE Frozen) ---")
    for param in ae_params:
        param.requires_grad = False

    optimizer_clf = torch.optim.AdamW(pipeline.decoder.parameters(), lr=LR)
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    epochs_no_improve = 0
    CLF_EPOCHS = EPOCHS

    for epoch in range(CLF_EPOCHS):
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"CLF Epoch {epoch+1}/{CLF_EPOCHS}"):
            x, y = x.to(device), y.to(device)
            
            # Forward pass without gradients for AE
            with torch.no_grad():
                z = pipeline.autoencoder.encode(x)
                zq = pipeline.quantizer(z)
                zc = pipeline.channel(zq)
                x_hat = pipeline.autoencoder.recon_decoder(zc)
            
            # Forward pass with gradients for Classifier
            logits = pipeline.decoder(x_hat)
            loss = ce_criterion(logits, y)

            optimizer_clf.zero_grad()
            loss.backward()
            optimizer_clf.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_clf(pipeline, val_loader, device)

        print(f"CLF Epoch {epoch+1}/{CLF_EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            pipeline.save_checkpoint(_save_path, epoch=epoch, val_acc=val_acc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] CLF Early stopping after {epoch+1} epochs")
                break

    print(f"[DONE] Best val_acc={best_val_acc:.4f}. Model -> {_save_path}")

if __name__ == "__main__":
    main()
