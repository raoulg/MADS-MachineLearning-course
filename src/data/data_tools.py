import random
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import tensorflow as tf


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
        yield p.resolve()


def iter_valid_paths(path: Path, formats: List[str]) -> Tuple[Generator, List[str]]:
    walk = walk_dir(path)
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    paths = (path for path in walk if path.suffix in formats)
    return paths, class_names


def data_generator(
    batch_size: int,
    path: Path,
    image_size: Tuple[int, int],
    channels: int,
    shuffle: bool = True,
    formats: List[str] = [".png", ".jpg"]
) -> Generator:

    paths, class_names = iter_valid_paths(path, formats)
    class_dict = {k: v for k, v in zip(class_names, range(len(class_names)))}
    valid_files = [*paths]

    data_size = len(valid_files)
    index_list = [*range(data_size)]

    if shuffle:
        random.shuffle(index_list)

    index = 0
    while True:
        # X = [0] * batch_size  # noqa: N806
        # Y = [0] * batch_size  # noqa: N806
        X = np.zeros((batch_size, image_size[0], image_size[1], channels))  # noqa: N806
        Y = np.zeros(batch_size)  # noqa: N806

        for i in range(batch_size):
            if index >= data_size:
                index = 0
                if shuffle:
                    random.shuffle(index_list)
            file = valid_files[index_list[index]]
            X[i] = load_image(file, image_size, channels)
            Y[i] = class_dict[file.parent.name]
            index += 1
        # X = np.array(X)  # noqa: N806
        # Y = np.array(Y)  # noqa: N806
        yield ((X, Y))


def load_image(path: Path, image_size: Tuple[int, int], channels: int) -> np.ndarray:
    img_ = tf.io.read_file(str(path))
    img = tf.image.decode_image(img_, channels=channels)
    img_resize = tf.image.resize(img, image_size, method="bilinear")
    img_resize.set_shape((image_size[0], image_size[1], channels))
    return img_resize.numpy()
