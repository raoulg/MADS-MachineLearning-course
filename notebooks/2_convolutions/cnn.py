"""The CNN from 01_convolutions.ipynb, in one place.

`02_pretrained.ipynb` and `03_mlflow.ipynb` both compare a from-scratch CNN against a head on
cached vectors. If each notebook pasted in its own copy of the model, the two comparisons would
drift apart the first time anyone edited one of them -- and you would not notice, because both
would still run. One definition, imported twice.

The model is the one from that notebook, with one edit: `num_classes` is an argument instead of a
hardcoded 10, because flowers has 102 of them. The `_conv_test` trick is
what makes it reusable: it discovers the activation-map size by running a dummy tensor through
the convolutions, so `AvgPool2d` always reduces to `(B, filters, 1, 1)` and the same class
accepts 28x28 greyscale and 64x64 colour without an edit.
"""

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        filters: int,
        units1: int,
        units2: int,
        input_size: tuple,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters

        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        activation_map_size = self._conv_test(input_size)
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
        )

    def _conv_test(self, input_size: tuple) -> tuple[int, int]:
        x = torch.ones(input_size, dtype=torch.float32)
        height, width = self.convolutions(x).shape[-2:]
        return int(height), int(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.agg(self.convolutions(x)))
