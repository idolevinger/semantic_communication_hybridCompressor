import torch
import torch.nn.functional as F
import math
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snr_sweep import _load_test_loader
from pipeline import build_pipeline
from config import QAM_POWER, RESULTS_DIR

def evaluate_reconstruction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=10.0, help="SNR value matching the trained models folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _load_test_loader()

    snr = args.snr
    noise_std = math.sqrt(QAM_POWER / (10 ** (snr / 10)))
    dims = [128, 64, 32, 16, 8, 4, 2]
    
    print("\n--- RECONSTRUCTION FIDELITY VS COMPRESSION ---")
    print("Testing if aggressive compression ruins the vector reconstruction (Cosine Similarity).")
    
    for dim in dims:
        ckpt = os.path.join(RESULTS_DIR, "dim_sweep", f"snr_{int(snr)}", f"model_dim_{dim}.pt")
        pipeline = build_pipeline(
            use_noise=True,
            device=device,
            checkpoint_path=ckpt,
            bottleneck_dim=dim,
            use_ldpc=False,
            noise_std=noise_std
        )
        pipeline.eval()
        
        total_cos_sim = 0.0
        samples = 0
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                z = pipeline.autoencoder.encode(x)
                z = pipeline.quantizer(z)
                z_rx = pipeline.channel(z)
                x_hat = pipeline.autoencoder.recon_decoder(z_rx)
                
                cos_sim = F.cosine_similarity(x_hat, x, dim=-1).sum().item()
                total_cos_sim += cos_sim
                samples += x.size(0)
                
        avg_cos = total_cos_sim / samples
        print(f"Bottleneck: {dim:>3d} | Cosine Similarity (x vs x_hat): {avg_cos:.4f}")

if __name__ == "__main__":
    evaluate_reconstruction()
