from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Iterator
import random
import torch
import numpy as np
import tensorflow as tf

from src.data import data_tools

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
        self.paths, self.class_names = data_tools.iter_valid_paths(path, formats)
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

    def __len__(self) -> int:
        return len(self.valid_files)

    def data_generator(
        self,
        batch_size: int,
        image_size: Tuple[int, int],
        channels: int,
        channel_first: bool,
        mode: str,
        shuffle: bool = True,
        transforms: Optional[Callable] = None,
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
            X = torch.zeros(  # noqa: N806
                    (batch_size, image_size[0], image_size[1], channels)
                )  # noqa: N806
            Y = torch.zeros(batch_size, dtype=torch.long)  # noqa: N806

            for i in range(batch_size):
                if index >= data_size:
                    index = 0
                    if shuffle:
                        random.shuffle(index_list)
                # get path
                file = self.valid_files[index_list[index]]
                # get image from disk
                if transforms is not None:
                    img = self.load_image(file, image_size, channels)
                    X[i] = transforms(img)
                else:
                    X[i] = torch.tensor(self.load_image(file, image_size, channels))
                # map parent directory name to integer
                Y[i] = self.class_dict[file.parent.name]
                index += 1

            if channel_first:
                X = torch.permute(X, (0, 3, 1, 2))  # noqa N806

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
        return img_resize.numpy().astype(np.uint8)