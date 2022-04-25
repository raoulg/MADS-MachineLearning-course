from pathlib import Path
from typing import Tuple

import gin
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def build_grid(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """creates a grid on [0,1] domain
    with a granularity k

    Args:
        k (int): granularity

    Returns:
        np.ndarray: 2D coordinate grid on [0,1]
    """
    x = np.linspace(0, 1, k)
    y = np.linspace(0, 1, k)
    xv, yv = np.meshgrid(x, y)
    grid = np.c_[xv.ravel(), yv.ravel()]
    return grid, x, y


@gin.configurable
def get_MNIST(  # noqa: N802
    data_dir: Path, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
