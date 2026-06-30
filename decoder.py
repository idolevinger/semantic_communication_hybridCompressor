"""
decoder.py
Task decoder: (noisy) bottleneck vector -> class logits.

    bottleneck -> HIDDEN -> [residual] -> [residual] -> HIDDEN/2 -> NUM_CLASSES
"""

import torch
import torch.nn as nn

from config import BOTTLENECK_DIM, HIDDEN_DIM, NUM_CLASSES


class TaskDecoder(nn.Module):
    def __init__(self, input_dim: int = BOTTLENECK_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.15)

        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_norm(self.input_proj(x)))

        identity = x
        out = self.dropout1(torch.relu(self.norm1(self.layer1(x))))
        out = out + identity

        identity = out
        out = self.dropout2(torch.relu(self.norm2(self.layer2(out))))
        out = out + identity

        features = self.dropout3(torch.relu(self.norm3(self.layer3(out))))
        return self.classifier(features)


if __name__ == "__main__":
    from config import BERT_DIM
    dec = TaskDecoder(input_dim=BERT_DIM)
    x = torch.randn(4, BERT_DIM)
    print("logits shape:", tuple(dec(x).shape))  # (4, NUM_CLASSES)
