# src_rnn/models.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RNNConfig:
    input_size: int = 3
    hidden_size: int = 64
    num_layers: int = 1
    output_size: int = 20
    dropout: float = 0.0


class GRULastStep(nn.Module):
    """
    Baseline: gebruikt de laatste timestep (van de gepaddede sequentie).
    Dit is bewust 'naïef' en dient als referentie.
    """

    def __init__(self, config: RNNConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        h, _ = self.rnn(x)               # (B, T, H)
        last = h[:, -1, :]               # (B, H)
        logits = self.classifier(last)   # (B, output_size)
        return logits


class GRUMeanPool(nn.Module):
    """
    Pooling fix: gebruikt mean pooling over tijd (B, T, H) -> (B, H).
    Dit is robuuster dan 'last step' bij variabele sequentielengtes.
    (Nog zonder masking; dat kan later als aparte verbetering.)
    """

    def __init__(self, config: RNNConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        h, _ = self.rnn(x)                # (B, T, H)
        pooled = h.mean(dim=1)            # (B, H)
        logits = self.classifier(pooled)  # (B, output_size)
        return logits
    
class Conv1DGRUMeanPool(nn.Module):
    """
    Conv1D vóór GRU:
    - input:  (B, T, C)
    - Conv1D werkt op (B, C, T), dus we permuten
    - output: (B, T, H) -> mean pooling -> classifier
    """

    def __init__(
        self,
        config: RNNConfig,
        conv_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.config = config

        # Conv over tijd (C=input channels = 3)
        self.conv = nn.Conv1d(
            in_channels=config.input_size,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # houdt T gelijk
        )
        self.act = nn.ReLU()

        # GRU verwacht input_size = conv_channels
        self.rnn = nn.GRU(
            input_size=conv_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.act(self.conv(x))     # (B, conv_channels, T)
        x = x.transpose(1, 2)          # (B, T, conv_channels)

        h, _ = self.rnn(x)             # (B, T, H)
        pooled = h.mean(dim=1)         # (B, H)
        logits = self.classifier(pooled)
        return logits    


def build_model(model_name: str, config: RNNConfig) -> nn.Module:
    """
    Kleine factory zodat notebooks alleen een modelnaam hoeven door te geven.
    """
    name = model_name.lower().strip()
    if name in {"gru_last", "last", "grulast"}:
        return GRULastStep(config)
    if name in {"gru_mean", "mean", "grumean"}:
        return GRUMeanPool(config)
    if name in {"conv1d_gru_mean", "conv_gru_mean", "conv1d"}:
        return Conv1DGRUMeanPool(config)
    raise ValueError(
        f"Unknown model_name='{model_name}'. Use 'gru_last', 'gru_mean', or 'conv1d_gru_mean'."
    )

