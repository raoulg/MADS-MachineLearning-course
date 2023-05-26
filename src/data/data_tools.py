from __future__ import annotations

import random
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
import torch
from loguru import logger
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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
        formats (List[str]): suffices to keep, eg [".jpg", ".png"]

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


def get_file(
    data_dir: Path,
    filename: Path,
    url: str,
    unzip: bool = True,
    overwrite: bool = False,
) -> Path:
    path = data_dir / filename
    if path.exists() and not overwrite:
        logger.info(f"File {path} already exists, skip download")
        return path
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 2**10
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    logger.info(f"Downloading {path}")
    with open(path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if unzip:
        extract(path)
    return path


def extract(path: Path) -> None:
    if path.suffix in [".zip"]:
        logger.info(f"Unzipping {path}")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)
    if path.suffix in [".tgz", ".tar.gz", ".gz"]:
        logger.info(f"Unzipping {path}")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=path.parent)


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


def dir_add_timestamp(log_dir: Optional[Path] = None) -> Path:
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = log_dir / timestamp
    logger.info(f"Logging to {log_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    return log_dir


class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: List = []
        self.process_data()

    def process_data(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]


class FacesDataset(BaseDataset):
    def __init__(self, paths: List[Path]) -> None:
        super().__init__(paths)

    def process_data(self) -> None:
        for path in self.paths:
            img = self.load_image(path)
            self.dataset.append((img, path.name))

    def load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        return img


class ImgDataset(BaseDataset):
    def __init__(
        self, paths: List[Path], class_names: List[str], img_size: Tuple[int, int]
    ) -> None:
        self.img_size = img_size
        self.class_names = class_names
        super().__init__(paths)

    def process_data(self) -> None:
        for file in self.paths:
            img = self.load_image(file, self.img_size)
            x = np.reshape(img, (1,) + img.shape)
            y = self.class_names.index(file.parent.name)
            self.dataset.append((x, y))

    def load_image(self, path: Path, image_size: Tuple[int, int]) -> np.ndarray:
        # load file
        img_ = Image.open(path).resize(image_size, Image.LANCZOS)
        return np.asarray(img_)


class TSDataset(BaseDataset):
    """This assume a txt file with numeric data
    Dropping the first columns is hardcoded
    y label is name-1, because the names start with 1

    Args:
        BaseDataset (_type_): _description_
    """

    def process_data(self) -> None:
        for file in tqdm(self.paths):
            x_ = np.genfromtxt(file)[:, 3:]
            x = torch.tensor(x_).type(torch.float32)
            y = torch.tensor(int(file.parent.name) - 1)
            self.dataset.append((x, y))


class TensorDataset:
    """The main responsibility of the Dataset class is to
    offer a __len__ method and a __getitem__ method
    """

    def __init__(self, data: Tensor, targets: Tensor) -> None:
        self.data = data
        self.targets = targets
        assert len(data) == len(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx], self.targets[idx]


class TextDataset(BaseDataset):
    """This assumes textual data, one line per file

    Args:
        BaseDataset (_type_): _description_
    """

    def process_data(self) -> None:
        for file in tqdm(self.paths):
            with open(file) as f:
                x = f.readline()
            y = file.parent.name
            self.dataset.append((x, y))


class BaseDataIterator:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: BaseDataset, batchsize: int) -> None:
        self.dataset = dataset
        self.batchsize = batchsize

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def __iter__(self) -> BaseDataIterator:
        self.index = 0
        self.index_list = torch.randperm(len(self.dataset))
        return self

    def batchloop(self) -> Tuple[List, List]:
        X = []  # noqa N806
        Y = []  # noqa N806
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            X.append(x)
            Y.append(y)
            self.index += 1
        return X, Y

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            return torch.tensor(X), torch.tensor(Y)
        else:
            raise StopIteration


class PaddedDatagenerator(BaseDataIterator):
    """Iterator with additional padding of X

    Args:
        BaseDataIterator (_type_): _description_
    """

    def __init__(self, dataset: BaseDataset, batchsize: int) -> None:
        super().__init__(dataset, batchsize)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(Y)
        else:
            raise StopIteration


class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: BaseDataset,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize
        self.preprocessor = preprocessor
        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            batch.append((x, y))
            self.index += 1
        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            if self.preprocessor is not None:
                X, Y = self.preprocessor(batch)  # noqa N806
            else:
                X, Y = zip(*batch)  # noqa N806
            yield X, Y


class VAEstreamer(BaseDatastreamer):
    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            # we throw away the Y
            X_, _ = zip(*batch)  # noqa N806
            X = torch.stack(X_)  # noqa N806
            # change the channel to channel-last
            X = torch.moveaxis(X, 1, 3)  # noqa N806
            # and yield X, X
            yield X, X
