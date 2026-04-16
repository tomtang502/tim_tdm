"""Crossing intent classifier: bi-LSTM over pedestrian trajectory features."""

from __future__ import annotations

import torch
import torch.nn as nn

SEQ_LEN: int = 15       # number of frames fed into the classifier
FEATURE_DIM: int = 8    # cx, cy, w, h, dx, dy, speed, bbox_area


class CrossingIntentLSTM(nn.Module):
    """Lightweight bi-directional LSTM for pedestrian crossing intent classification."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.classifier(last)
