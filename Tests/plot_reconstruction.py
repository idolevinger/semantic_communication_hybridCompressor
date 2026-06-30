"""
plot_reconstruction.py
Measures and plots the Cosine Similarity (reconstruction accuracy) of the Autoencoder
for different bottleneck dimensions.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure imports work from the Tests directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import build_pipeline
import config
from train import load_datasets

@torch.no_grad()
def evaluate_reconstruction(pipeline, loader, device):
    pipeline.eval()
    total_cos_sim = 0.0
    total_samples = 0
    for x, _ in loader:
        x = x.to(device)
        _, x_hat = pipeline.forward_train(x)
        # Calculate cosine similarity between original and reconstructed
        cos_sim = F.cosine_similarity(x, x_hat, dim=-1).sum().item()
        total_cos_sim += cos_sim
        total_samples += x.size(0)
    return total_cos_sim / total_samples

def main():
    dims = [128, 64, 32, 16, 8, 4, 2]
    snr = 20.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_ds = load_datasets()
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    similarities = []
    
    print(f"[INFO] Evaluating Reconstruction Accuracy at SNR = {snr} dB")
    for dim in dims:
        model_path = os.path.join(config.RESULTS_DIR, "dim_sweep", f"snr_{int(snr)}", f"model_dim_{dim}.pt")
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            similarities.append(0.0)
            continue
            
        noise_std = 10 ** (-snr / 20.0)
        pipeline = build_pipeline(use_noise=True, device=device, checkpoint_path=model_path, bottleneck_dim=dim, noise_std=noise_std)
        
        sim = evaluate_reconstruction(pipeline, val_loader, device)
        similarities.append(sim * 100.0)
        print(f"Dim {dim}: Cosine Similarity = {sim*100:.2f}%")
        
    plt.figure(figsize=(10, 6))
    x_labels = [str(d) for d in dims]
    plt.plot(x_labels, similarities, marker='o', linestyle='-', linewidth=2, markersize=8, color='indigo')
    
    # Annotate first and last points with arrows below
    arrow_props = dict(arrowstyle="->", color='indigo', shrinkA=0, shrinkB=5)
    plt.annotate(f"{similarities[0]:.2f}%", (x_labels[0], similarities[0]), textcoords="offset points", xytext=(0,-30), ha='center', fontsize=10, fontweight='bold', color='indigo', arrowprops=arrow_props)
    plt.annotate(f"{similarities[-1]:.2f}%", (x_labels[-1], similarities[-1]), textcoords="offset points", xytext=(0,-30), ha='center', fontsize=10, fontweight='bold', color='indigo', arrowprops=arrow_props)

    plt.title(f'Semantic Reconstruction Quality vs Bottleneck Dimension\n(Unsupervised AE, SNR = {snr} dB)', fontsize=14)
    plt.xlabel('Bottleneck Dimension (number of real values)', fontsize=12)
    plt.ylabel('Cosine Similarity (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = os.path.join(config.RESULTS_DIR, "dim_sweep", f"snr_{int(snr)}", "reconstruction_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved plot to {out_path}")

if __name__ == "__main__":
    main()
