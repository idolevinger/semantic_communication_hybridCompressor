import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train import load_datasets

def main():
    _, val_ds = load_datasets()
    loader = DataLoader(val_ds, batch_size=7600, shuffle=False)
    x, _ = next(iter(loader)) # x is [7600, 768]
    
    # 1. Similarity to Global Mean
    global_mean = x.mean(dim=0, keepdim=True) # [1, 768]
    sim_to_mean = F.cosine_similarity(x, global_mean, dim=-1).mean().item()
    print(f"Average Cosine Similarity to Global Mean: {sim_to_mean * 100:.2f}%")
    
    # 2. Similarity between random pairs
    x_shifted = torch.roll(x, shifts=1, dims=0)
    sim_random = F.cosine_similarity(x, x_shifted, dim=-1).mean().item()
    print(f"Average Cosine Similarity between random pairs: {sim_random * 100:.2f}%")

if __name__ == "__main__":
    main()
