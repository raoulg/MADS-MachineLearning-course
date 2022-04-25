from asyncio import shield
import random
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
import shutil
from loguru import logger


def walk_dir(path: Path) -> Generator:
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


def iter_valid_paths(path: Path, formats: List[str]) -> Tuple[Generator, List[str]]:
    # gets all files in folder and subfolders
    walk = walk_dir(path)
    # retrieves foldernames as classnames
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    # keeps only specified formats
    paths = (path for path in walk if path.suffix in formats)
    return paths, class_names


def data_generator(
    batch_size: int,
    path: Path,
    image_size: Tuple[int, int],
    channels: int,
    shuffle: bool = True,
    formats: List[str] = [".png", ".jpg"],
) -> Generator:

    # get all paths
    paths, class_names = iter_valid_paths(path, formats)
    # make a dictionary mapping class names to an integer
    class_dict: Dict[str, int] = {
        k: v for k, v in zip(class_names, range(len(class_names)))
    }

    # unpack generator
    valid_files = [*paths]

    data_size = len(valid_files)
    index_list = [*range(data_size)]

    if shuffle:
        random.shuffle(index_list)

    index = 0
    while True:
        # prepare empty matrices
        X = np.zeros((batch_size, image_size[0], image_size[1], channels))  # noqa: N806
        Y = np.zeros(batch_size)  # noqa: N806

        for i in range(batch_size):
            if index >= data_size:
                index = 0
                if shuffle:
                    random.shuffle(index_list)
            # get path
            file = valid_files[index_list[index]]
            # get image from disk
            X[i] = load_image(file, image_size, channels)
            # map parent directory name to integer
            Y[i] = class_dict[file.parent.name]
            index += 1
        yield ((X, Y))


def load_image(path: Path, image_size: Tuple[int, int], channels: int) -> np.ndarray:
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

def clean_dir(dir: str) -> None:
    dir = Path(dir)
    if dir.exists():
        logger.info("Clean out {dir}")
        shutil.rmtree(dir)

