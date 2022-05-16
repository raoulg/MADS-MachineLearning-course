import random
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

Tensor = torch.Tensor


def walk_dir(path: Path) -> Iterator:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop trough. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
            continue
        # resolve works like .absolute(), but it removes the "../.." parts
        # of the location, so it is cleaner
        yield p.resolve()


def iter_valid_paths(path: Path, formats: List[str]) -> Tuple[Iterator, List[str]]:
    """
    Gets all paths in folders and subfolders
    strips the classnames assuming that the subfolders are the classnames
    Keeps only paths with the right suffix


    Args:
        path (Path): image folder
        formats (List[str]): suffices to keep.

    Returns:
        Tuple[Iterator, List[str]]: _description_
    """
    # gets all files in folder and subfolders
    walk = walk_dir(path)
    # retrieves foldernames as classnames
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    # keeps only specified formats
    paths = (path for path in walk if path.suffix in formats)
    return paths, class_names


class Dataloader:
    """Point the dataloader to a directory.
    It will load all files, taking the subfolders as classes.
    Only files in the formatslist are kept.
    The .datagenerator returns an Iterator of batched images and labels
    """

    def __init__(
        self,
        path: Path,
        split: float,
        formats: List[str] = [".png", ".jpg"],
    ) -> None:
        """
        Initializes the class

        Args:
            path (Path): location of the images
            formats (List[str], optional): Formats to keep. Defaults to [".png", ".jpg"]
        """

        # get all paths
        self.paths, self.class_names = iter_valid_paths(path, formats)
        # make a dictionary mapping class names to an integer
        self.class_dict: Dict[str, int] = {
            k: v for k, v in zip(self.class_names, range(len(self.class_names)))
        }
        # unpack generator
        self.valid_files = [*self.paths]
        self.data_size = len(self.valid_files)
        self.index_list = [*range(self.data_size)]

        random.shuffle(self.index_list)

        n_train = int(self.data_size * split)
        self.train = self.index_list[:n_train]
        self.test = self.index_list[n_train:]

    def data_generator(
        self,
        batch_size: int,
        image_size: Tuple[int, int],
        channels: int,
        channel_first: bool,
        mode: str,
        shuffle: bool = True,
    ) -> Iterator:
        """
        Builds batches of images

        Args:
            batch_size (int): _description_
            image_size (Tuple[int, int]): _description_
            channels (int): _description_
            shuffle (bool, optional): _description_. Defaults to True.

        Yields:
            Iterator: _description_
        """
        if mode == "train":
            data_size = len(self.train)
            index_list = self.train
        else:
            data_size = len(self.test)
            index_list = self.test

        index = 0
        while True:
            # prepare empty matrices
            X = np.zeros(  # noqa: N806
                (batch_size, image_size[0], image_size[1], channels)
            )  # noqa: N806
            Y = np.zeros(batch_size)  # noqa: N806

            for i in range(batch_size):
                if index >= data_size:
                    index = 0
                    if shuffle:
                        random.shuffle(index_list)
                # get path
                file = self.valid_files[index_list[index]]
                # get image from disk
                X[i] = self.load_image(file, image_size, channels)
                # map parent directory name to integer
                Y[i] = self.class_dict[file.parent.name]
                index += 1

            if channel_first:
                X = np.moveaxis(X, 3, 1)  # noqa: N806

            yield ((X, Y))

    def load_image(
        self, path: Path, image_size: Tuple[int, int], channels: int
    ) -> np.ndarray:
        # load file
        img_ = tf.io.read_file(str(path))
        # decode as image
        img = tf.image.decode_image(img_, channels=channels)
        # resize with bilinear algorithm
        img_resize = tf.image.resize(img, image_size, method="bilinear")
        # add correct shape with channels-last convention
        img_resize.set_shape((image_size[0], image_size[1], channels))
        # cast to numpy
        return img_resize.numpy()


def clean_dir(dir: Union[str, Path]) -> None:
    dir = Path(dir)
    if dir.exists():
        logger.info(f"Clean out {dir}")
        shutil.rmtree(dir)
    else:
        dir.mkdir(parents=True)


def window(x: Tensor, n_time: int) -> Tensor:
    """
    Generates and index that can be used to window a timeseries.
    E.g. the single series [0, 1, 2, 3, 4, 5] can be windowed into 4 timeseries with
    length 3 like this:

    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]

    We now can feed 4 different timeseries into the model, instead of 1, all
    with the same length.
    """
    n_window = len(x) - n_time + 1
    time = torch.arange(0, n_time).reshape(1, -1)
    window = torch.arange(0, n_window).reshape(-1, 1)
    idx = time + window
    return idx
