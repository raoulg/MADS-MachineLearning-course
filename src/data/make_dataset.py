from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import gin
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.data import data_tools
from src.data.data_tools import PaddedDatagenerator, TSDataset

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
    image_folder = Path(data_dir) / "datasets/flower_photos"
    if not image_folder.exists():
        image_folder = tf.keras.utils.get_file(
            "flower_photos", origin=dataset_url, untar=True, cache_dir=data_dir
        )
        image_folder = Path(image_folder)
        logger.info(f"Data is downloaded to {image_folder}.")
    else:
        logger.info(f"Dataset already exists at {image_folder}")
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


def get_imdb_data(cache_dir: str = ".") -> Tuple[List[Path], List[Path]]:
    datapath = Path(cache_dir) / "aclImdb"
    if datapath.exists():
        logger.info(f"{datapath} already exists, skipping download")
    else:
        logger.info(f"{datapath} not found on disk, downloading")

        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        _ = tf.keras.utils.get_file(
            "aclImdb_v1.tar.gz", url, untar=True, cache_dir=cache_dir, cache_subdir=""
        )
    testdir = datapath / "test"
    traindir = datapath / "train"
    keep_subdirs_only(testdir)
    keep_subdirs_only(traindir)
    unsup = traindir / "unsup"
    if unsup.exists():
        shutil.rmtree(traindir / "unsup")
    formats = [".txt"]
    testpaths = [
        path for path in data_tools.walk_dir(testdir) if path.suffix in formats
    ]
    trainpaths = [
        path for path in data_tools.walk_dir(traindir) if path.suffix in formats
    ]
    return trainpaths, testpaths


def get_gestures(
    data_dir: Path, split: float, batchsize: int
) -> Tuple[PaddedDatagenerator, PaddedDatagenerator]:
    formats = [".txt"]
    paths = [path for path in data_tools.walk_dir(data_dir) if path.suffix in formats]

    # make a train-test split
    idx = int(len(paths) * split)
    trainpaths = paths[:idx]
    testpaths = paths[idx:]
    traindataset = TSDataset(trainpaths)
    testdataset = TSDataset(testpaths)
    trainloader = PaddedDatagenerator(traindataset, batchsize=32)
    testloader = PaddedDatagenerator(testdataset, batchsize=32)
    return trainloader, testloader


def keep_subdirs_only(path: Path) -> None:
    files = [file for file in path.iterdir() if file.is_file()]
    for file in files:
        file.unlink()
