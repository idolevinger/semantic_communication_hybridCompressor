"""
data.py
Load AG News (raw text + integer labels) with a train/val/test split and a CSV cache.

All configuration is imported from config.py -- nothing is redefined here.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import (
    DATASET_NAME,
    TRAIN_SIZE,
    TEST_SIZE,
    VAL_RATIO,
    RANDOM_SEED,
    CACHE_TEXTS_FILE,
    USE_CACHED,
)


class TextDataset(Dataset):
    """Returns (text, label) pairs."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class DatasetLoader:
    def _download(self):
        print(f"[DATA] Downloading {DATASET_NAME} from Hugging Face")
        ds = load_dataset(DATASET_NAME)
        train_split = ds["train"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SIZE))
        test_split = ds["test"].shuffle(seed=RANDOM_SEED).select(range(TEST_SIZE))
        print(f"[DATA] train={len(train_split)}  test={len(test_split)}")
        return train_split, test_split

    def _save_cache(self, train_t, val_t, test_t, train_y, val_y, test_y):
        df = pd.concat(
            [
                pd.DataFrame({"split": "train", "text": train_t, "label": train_y}),
                pd.DataFrame({"split": "val", "text": val_t, "label": val_y}),
                pd.DataFrame({"split": "test", "text": test_t, "label": test_y}),
            ],
            ignore_index=True,
        )
        df.to_csv(CACHE_TEXTS_FILE, index=False)
        print(f"[CACHE] Saved text cache -> {CACHE_TEXTS_FILE}")

    def _load_cache(self):
        print(f"[CACHE] Loading text cache <- {CACHE_TEXTS_FILE}")
        df = pd.read_csv(CACHE_TEXTS_FILE)

        def pick(split):
            part = df[df["split"] == split]
            return part["text"].tolist(), part["label"].tolist()

        train_t, train_y = pick("train")
        val_t, val_y = pick("val")
        test_t, test_y = pick("test")
        return train_t, val_t, test_t, train_y, val_y, test_y

    def load(self):
        if USE_CACHED and os.path.exists(CACHE_TEXTS_FILE):
            train_t, val_t, test_t, train_y, val_y, test_y = self._load_cache()
        else:
            train_split, test_split = self._download()

            full_train_t = list(train_split["text"])
            full_train_y = list(train_split["label"])

            train_t, val_t, train_y, val_y = train_test_split(
                full_train_t,
                full_train_y,
                test_size=VAL_RATIO,
                stratify=full_train_y,
                random_state=RANDOM_SEED,
            )

            test_t = list(test_split["text"])
            test_y = list(test_split["label"])

            self._save_cache(train_t, val_t, test_t, train_y, val_y, test_y)

        return {
            "train": TextDataset(train_t, train_y),
            "val": TextDataset(val_t, val_y),
            "test": TextDataset(test_t, test_y),
        }


if __name__ == "__main__":
    ds = DatasetLoader().load()
    print("Sizes:")
    print("  train:", len(ds["train"]))
    print("  val:  ", len(ds["val"]))
    print("  test: ", len(ds["test"]))
    x, y = ds["train"][0]
    print("\nExample:")
    print("  label:", int(y))
    print("  text :", x[:150], "...")
