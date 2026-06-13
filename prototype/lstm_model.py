"""
LSTMClassifier — verbatim copy of cell 49 in the training notebook.
The class definition must match training exactly for torch.load state_dict
to work without key errors.
"""
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64,
                 hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h = h.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
        last = h[-1]
        pooled = torch.cat([last[0], last[1]], dim=1)
        return self.fc(self.dropout(pooled)).squeeze(1)
