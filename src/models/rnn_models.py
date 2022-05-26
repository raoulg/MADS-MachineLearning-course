from typing import Dict

import gin
import torch
from torch import nn

Tensor = torch.Tensor


class BaseModel(nn.Module):
    def __init__(self, observations: int, horizon: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()  # we have 3d data, the linear model wants 2D
        self.linear = nn.Linear(observations, horizon)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


@gin.configurable
class BaseRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, horizon: int
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(hidden_size, horizon)
        self.horizon = horizon

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


@gin.configurable
class GRUmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionGRU(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class NLPmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionNLP(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
