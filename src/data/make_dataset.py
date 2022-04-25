from pathlib import Path
from typing import Tuple

import gin
import numpy as np
import tensorflow as tf
from loguru import logger
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


def get_flowers(data_dir: Path) -> Path:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  # noqa: E501
    image_folder = tf.keras.utils.get_file(
        "flower_photos", origin=dataset_url, untar=True, cache_dir=data_dir
    )

    image_folder = Path(image_folder)
    logger.info(f"Data is downloaded to {image_folder}.")
    return image_folder
