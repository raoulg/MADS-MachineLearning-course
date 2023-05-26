from typing import Dict

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    """encoder"""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.encode = nn.Sequential(
            nn.Linear(config["insize"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["latent"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        latent = self.encode(x)
        return latent


class Decoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(config["latent"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["insize"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decode(x)
        x = x.reshape((-1, 28, 28, 1))
        return x


class RecostructionLoss:
    def __call__(self, y, yhat):
        sqe = (y - yhat) ** 2
        summed = sqe.sum(dim=(1, 2, 3))
        return summed.mean()


class AutoEncoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x


def sample_range(encoder, stream, k: int = 10):
    minmax_list = []
    for _ in range(10):
        X, _ = next(stream)
        y = encoder(X).detach().numpy()
        minmax_list.append(y.min())
        minmax_list.append(y.max())
    minmax = np.array(minmax_list)
    return minmax.min(), minmax.max()


def build_latent_grid(decoder, minimum: int, maximum: int, k: int = 20):
    x = np.linspace(minimum, maximum, k)
    y = np.linspace(minimum, maximum, k)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    img = decoder(torch.tensor(grid, dtype=torch.float32))
    return img.detach().numpy()


def select_n_random(data, labels, n=300):
    """
    Selects n random datapoints and their corresponding labels from a dataset
    """
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
