"""
bert_encoder.py
Frozen BERT encoder: text -> 768-D CLS embeddings.

The CLS embedding is the average of the last two hidden layers' [CLS] tokens
(0.5 * last + 0.5 * second-to-last). This blend is a deliberate choice -- the
second-to-last layer tends to carry slightly more general (less task-specialised)
features than the final layer, and averaging the two empirically gives a more
stable sentence representation for downstream classification than the last
layer alone. Change `_pool` if you want last-layer-only.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from config import BERT_MODEL_NAME, MAX_SEQ_LEN


class BERTEncoder:
    def __init__(self, model_name: str = BERT_MODEL_NAME, max_length: int = MAX_SEQ_LEN):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # frozen

    @staticmethod
    def _pool(outputs) -> torch.Tensor:
        """Blend the [CLS] token of the last two hidden layers."""
        last_hidden = outputs.hidden_states[-1][:, 0, :]
        second_last = outputs.hidden_states[-2][:, 0, :]
        return 0.5 * last_hidden + 0.5 * second_last

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Return (len(texts), 768) CLS embeddings on CPU."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            cls = self._pool(outputs)
            all_embeddings.append(cls.cpu())
        return torch.cat(all_embeddings, dim=0)


if __name__ == "__main__":
    enc = BERTEncoder()
    sample = [
        "NASA launches a new telescope to study distant galaxies.",
        "The home team won the championship in overtime.",
        "Stocks rallied after the central bank cut interest rates.",
    ]
    emb = enc.encode_texts(sample)
    print("embeddings shape:", tuple(emb.shape))  # (3, 768)

    # quick sanity: similar topics should be more similar than dissimilar ones
    import torch.nn.functional as F
    e = F.normalize(emb, dim=1)
    sim = e @ e.t()
    print("cosine similarity matrix:\n", sim)
