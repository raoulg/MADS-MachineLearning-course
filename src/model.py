from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    base_channels: int = 16   # bepaalt model "breedte"
    fc_units: int = 64        # hidden units in classifier
    dropout: float = 0.0


class SmallCNN(nn.Module):
    """Kleine CNN voor 32x32 RGB (CIFAR-10)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.base_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16

            nn.Conv2d(c, 2 * c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((2 * c) * 8 * 8, cfg.fc_units),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.fc_units, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
