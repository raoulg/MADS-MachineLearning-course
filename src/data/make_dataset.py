from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gin
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

Tensor = torch.Tensor


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


def get_flowers(data_dir: Path) -> Path:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  # noqa: E501
    image_folder = tf.keras.utils.get_file(
        "flower_photos", origin=dataset_url, untar=True, cache_dir=data_dir
    )

    image_folder = Path(image_folder)
    logger.info(f"Data is downloaded to {image_folder}.")
    return image_folder


def get_sunspots(datadir: Path) -> pd.DataFrame:
    """loads sunspot data since 1749, selects year and monthly mean"""
    file = datadir / "sunspots.csv"
    if file.exists():
        logger.info(f"Found {file}, load from disk")
        data = pd.read_csv(file)
    else:
        logger.info(f"{file} does not exist, retrieving")
        spots = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php", stream=True)
        spots_ = np.genfromtxt(spots.raw, delimiter=";")
        data = pd.DataFrame(spots_[:, 2:4], columns=["year", "MonthlyMean"])  # type: ignore # noqa: E501
        data.to_csv(file, index=False)
    return data


class Datagenerator:
    def __init__(self, paths: List[Path], batchsize: int) -> None:
        self.dataset = paths
        self.size = len(self.dataset)
        self.batchsize = batchsize

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x_ = np.genfromtxt(self.dataset[idx])[:, 3:]
        x = torch.tensor(x_).type(torch.float32)
        y = self.dataset[idx].parent.name
        return x, int(y) - 1

    def __iter__(self) -> Datagenerator:
        self.index = 0
        self.index_list = torch.randperm(self.size)
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X = []  # noqa N806
            Y = []  # noqa N806
            for _ in range(self.batchsize):
                x, y = self[int(self.index_list[self.index])]
                X.append(x)
                Y.append(y)
                self.index += 1
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(Y)
        else:
            raise StopIteration


def get_gestures(datadir: Path) -> None:
    pass
