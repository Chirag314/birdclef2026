"""
perch_mlp.py — lightweight MLP classifier on top of frozen Perch v2 embeddings.

Architecture: Linear(1536→512) + BN + ReLU + Dropout → Linear(512→256) + BN + ReLU + Dropout → Linear(256→n_classes)
"""

import torch
import torch.nn as nn


class PerchMLP(nn.Module):
    def __init__(self,
                 n_classes: int,
                 in_dim: int = 1536,
                 hidden: tuple = (512, 256),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h, bias=False),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
