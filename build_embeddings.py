"""
build_embeddings.py
Run ONCE. Encodes every AG News split with frozen BERT and caches 768-D vectors.

    text (data.py) -> BERT (bert_encoder.py) -> cached tensors (config.CACHE_EMBEDDINGS_FILE)
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DatasetLoader
from bert_encoder import BERTEncoder
from config import CACHE_EMBEDDINGS_FILE, BATCH_SIZE


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def encode_split(encoder, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    all_emb, all_lbl = [], []
    for texts, labels in tqdm(loader, desc="Encoding"):
        emb = encoder.encode_texts(texts)
        all_emb.append(emb.cpu())
        all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


def main():
    print("[INFO] Building BERT embeddings (slow -- run once)")
    datasets = DatasetLoader().load()
    encoder = BERTEncoder()

    data = {}
    for split in ["train", "val", "test"]:
        emb, lbl = encode_split(encoder, datasets[split])
        data[split] = {"embeddings": emb, "labels": lbl}
        print(f"[INFO] {split}: embeddings {tuple(emb.shape)}, labels {tuple(lbl.shape)}")

    torch.save(data, CACHE_EMBEDDINGS_FILE)
    print(f"[DONE] Saved -> {CACHE_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
