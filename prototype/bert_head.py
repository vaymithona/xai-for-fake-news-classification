"""
ClassifierHead — verbatim copy of cell 64 in the training notebook.
The class definition must match training exactly for torch.load state_dict
to work without key errors.
"""
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)
