from __future__ import annotations

import random
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from loguru import logger
from PIL import Image
from sklearn import datasets as sk_datasets
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.data import data_tools
from src.data.data_tools import get_file, iter_valid_paths, keep_subdirs_only
from src.models import tokenizer
from src.settings import (
    DatasetSettings,
    ImgDatasetSettings,
    TextDatasetSettings,
    WindowedDatasetSettings,
    fashionmnistsettings,
    flowersdatasetsettings,
    gesturesdatasetsettings,
    imdbdatasetsettings,
    sunspotsettings,
)

Tensor = torch.Tensor


class DatasetType(Enum):
    FLOWERS = 1
    IMDB = 2
    GESTURES = 3
    FASHION = 4
    SUNSPOTS = 5


class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple:
        ...


class ProcessingDatasetProtocol(DatasetProtocol):
    def process_data(self) -> None:
        ...


class DatastreamerProtocol(Protocol):
    def stream(self) -> Iterator:
        ...


T = TypeVar("T", bound=DatasetSettings)


class AbstractDatasetFactory(ABC, Generic[T]):
    def __init__(self, settings: T):
        self._settings = settings

    @property
    def settings(self) -> T:
        return self._settings

    def download_data(self) -> None:
        url = self._settings.dataset_url
        filename = self._settings.filename
        datadir = self._settings.data_dir
        self.subfolder = Path(datadir) / self.settings.name
        if not self.subfolder.exists():
            logger.info("Start download...")
            self.subfolder.mkdir(parents=True)
            self.filepath = get_file(self.subfolder, filename, url=url, overwrite=False)
        else:
            logger.info(f"Dataset already exists at {self.subfolder}")
            self.filepath = self.subfolder / filename

    @abstractmethod
    def create_dataset(self, *args, **kwargs) -> Mapping[str, DatasetProtocol]:
        raise NotImplementedError

    @abstractmethod
    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        raise NotImplementedError


class SunspotsDatasetFactory(AbstractDatasetFactory[WindowedDatasetSettings]):
    """
    Data from https://www.sidc.be/SILSO/datafiles
    """

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        spots = np.genfromtxt(str(self.filepath), usecols=(3))
        tensors = torch.from_numpy(spots).type(torch.float32)

        split = kwargs.pop("split", 0.8)
        idx = int(len(tensors) * split)
        train = tensors[:idx]
        valid = tensors[idx:]

        norm = max(train)
        train = train / norm
        valid = valid / norm

        window_size = self.settings.window_size
        horizon = self.settings.horizon
        trainset = SunspotDataset(self._window(train, window_size), horizon)
        validset = SunspotDataset(self._window(valid, window_size), horizon)
        return {"train": trainset, "valid": validset}

    def _window(self, data: Tensor, window_size: int) -> Tensor:
        idx = data_tools.window(data, window_size)
        dataset = data[idx]
        dataset = dataset[..., None]
        return dataset

    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]

        def _preprocess(batch: List[Tuple]):
            X, y = zip(*batch)
            return torch.stack(X), torch.stack(y)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=_preprocess
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=_preprocess
        )
        return {"train": trainstreamer, "valid": validstreamer}


class FashionDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        data_dir = self.settings.data_dir

        training_data = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        valid_data = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return {"train": training_data, "valid": valid_data}

    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]

        def _preprocess(batch: List[Tuple]):
            X, y = zip(*batch)
            return torch.stack(X), torch.tensor(y)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=_preprocess
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=_preprocess
        )
        return {"train": trainstreamer, "valid": validstreamer}


class GesturesDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def __init__(self, settings: DatasetSettings) -> None:
        super().__init__(settings)
        self._created = False
        self.datasets: Mapping[str, DatasetProtocol]

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()

        if self._created:
            return self.datasets
        formats = [f.value for f in self._settings.formats]
        datadir = self.subfolder / "gestures-dataset"
        img = datadir / "gestures.png"
        if img.exists():
            shutil.move(img, datadir.parent / "gestures.png")
        keep_subdirs_only(datadir)
        paths = [
            path for path in data_tools.walk_dir(datadir) if path.suffix in formats
        ]
        random.shuffle(paths)

        split = kwargs.pop("split", 0.8)
        idx = int(len(paths) * split)
        trainpaths = paths[:idx]
        validpaths = paths[idx:]

        traindataset = TSDataset(trainpaths)
        validdataset = TSDataset(validpaths)
        datasets = {
            "train": traindataset,
            "valid": validdataset,
        }
        self.datasets = datasets  # type: ignore
        self._created = True
        return datasets

    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]

        def _preprocess(batch: List[Tuple]):
            X, y = zip(*batch)
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(y)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=_preprocess
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=_preprocess
        )
        return {"train": trainstreamer, "valid": validstreamer}


class IMDBDatasetFactory(AbstractDatasetFactory[TextDatasetSettings]):
    def __init__(self, settings: TextDatasetSettings) -> None:
        super().__init__(settings)

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        testdir = self.subfolder / "aclImdb/test"
        traindir = self.subfolder / "aclImdb/train"
        keep_subdirs_only(testdir)
        keep_subdirs_only(traindir)

        # remove dir with unlabeled reviews
        unsup = traindir / "unsup"
        if unsup.exists():
            shutil.rmtree(traindir / "unsup")

        formats = [f.value for f in self._settings.formats]
        trainpaths = [
            path for path in data_tools.walk_dir(traindir) if path.suffix in formats
        ]
        testpaths = [
            path for path in data_tools.walk_dir(testdir) if path.suffix in formats
        ]
        logger.info(
            f"Creating TextDatasets from {len(trainpaths)} trainfiles and {len(testpaths)} testfiles."
        )

        traindataset = TextDataset(paths=trainpaths)
        testdataset = TextDataset(paths=testpaths)
        return {"train": traindataset, "valid": testdataset}

    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        testdataset = datasets["valid"]
        corpus = []
        clean_fn = self.settings.clean_fn

        for i in range(len(traindataset)):
            x = clean_fn(traindataset[i][0])
            corpus.append(x)
        v = tokenizer.build_vocab(corpus, max=self.settings.maxvocab)

        logger.info(f"Vocab length is {len(v)}.")
        preprocessor = tokenizer.Preprocessor(
            max=self.settings.maxtokens,
            vocab=v,
            clean=clean_fn,
        )
        trainstreamer = BaseDatastreamer(
            traindataset, batchsize, preprocessor=preprocessor
        )
        teststreamer = BaseDatastreamer(
            testdataset, batchsize, preprocessor=preprocessor
        )
        return {"train": trainstreamer, "valid": teststreamer}


class FlowersDatasetFactory(AbstractDatasetFactory[ImgDatasetSettings]):
    def __init__(self, settings: ImgDatasetSettings) -> None:
        super().__init__(settings)

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        formats = self._settings.formats
        paths_, class_names = iter_valid_paths(
            self.subfolder / "flower_photos", formats=formats
        )
        paths = [*paths_]
        random.shuffle([paths])
        trainidx = int(len(paths) * self._settings.trainfrac)
        train = paths[:trainidx]
        valid = paths[trainidx:]
        traindataset = ImgDataset(train, class_names, img_size=self._settings.img_size)
        validdataset = ImgDataset(valid, class_names, img_size=self._settings.img_size)
        return {"train": traindataset, "valid": validdataset}

    def create_datastreamer(self, batchsize: int) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(10),
            ]
        )

        def _preprocess(batch: List[Tuple]):
            X, y = zip(*batch)
            X = torch.stack([transformations(x.squeeze()) for x in X])
            return X, torch.tensor(y)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=_preprocess
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=_preprocess
        )
        return {"train": trainstreamer, "valid": validstreamer}


class DatasetFactoryProvider:
    @staticmethod
    def get_factory(dataset_type: DatasetType) -> AbstractDatasetFactory:
        if dataset_type == DatasetType.FLOWERS:
            return FlowersDatasetFactory(flowersdatasetsettings)
        if dataset_type == DatasetType.IMDB:
            return IMDBDatasetFactory(imdbdatasetsettings)
        if dataset_type == DatasetType.GESTURES:
            return GesturesDatasetFactory(gesturesdatasetsettings)
        if dataset_type == DatasetType.FASHION:
            return FashionDatasetFactory(fashionmnistsettings)
        if dataset_type == DatasetType.SUNSPOTS:
            return SunspotsDatasetFactory(sunspotsettings)

        raise ValueError(f"Invalid dataset type: {dataset_type}")


class SunspotDataset(DatasetProtocol):
    def __init__(self, data: Tensor, horizon: int) -> None:
        self.data = data
        self.size = len(data)
        self.horizon = horizon

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # get a single item
        item = self.data[idx]
        # slice off the horizon
        x = item[: -self.horizon, :]
        y = item[-self.horizon :, :].squeeze(
            -1
        )  # squeeze will remove the last dimension if possible.
        return x, y


class AbstractDataset(ABC, ProcessingDatasetProtocol):
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: List = []
        self.process_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]

    @abstractmethod
    def process_data(self) -> None:
        raise NotImplementedError


class FacesDataset(AbstractDataset):
    def __init__(self, paths: List[Path]) -> None:
        super().__init__(paths)

    def process_data(self) -> None:
        for path in self.paths:
            img = self.load_image(path)
            self.dataset.append((img, path.name))

    def load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        return img


class ImgDataset(AbstractDataset):
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

    def __repr__(self) -> str:
        return f"ImgDataset (imgsize {self.img_size}, #classes {len(self.class_names)})"


class TSDataset(AbstractDataset):
    """This assume a txt file with numeric data
    Dropping the first columns is hardcoded
    y label is name-1, because the names start with 1

    """

    def process_data(self) -> None:
        for file in tqdm(self.paths, colour="#1e4706"):
            x_ = np.genfromtxt(file)[:, 3:]
            x = torch.tensor(x_).type(torch.float32)
            y = torch.tensor(int(file.parent.name) - 1)
            self.dataset.append((x, y))

    def __repr__(self) -> str:
        return f"TSDataset (size {len(self)})"


class TensorDataset(DatasetProtocol):
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


class TextDataset(AbstractDataset):
    """This assumes textual data, one line per file"""

    def process_data(self) -> None:
        for file in tqdm(self.paths, colour="#1e4706"):
            with open(file) as f:
                x = f.readline()
            y = file.parent.name
            self.dataset.append((x, y))

    def __repr__(self) -> str:
        return f"TextDataset (len {len(self)})"


class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: DatasetProtocol,
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

    def __repr__(self) -> str:
        return f"BasetDatastreamer: {self.dataset} (streamerlen {len(self)})"

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


def get_breast_cancer_dataset(
    train_perc: float,
) -> Tuple[TensorDataset, TensorDataset, Any]:
    npdata = sk_datasets.load_breast_cancer()
    featurenames = npdata.feature_names
    tensordata = torch.tensor(npdata.data, dtype=torch.float32)
    tensortarget = torch.tensor(npdata.target, dtype=torch.uint8)
    trainidx = int(len(tensordata) * train_perc)
    traindataset = TensorDataset(tensordata[:trainidx], tensortarget[:trainidx])
    testdataset = TensorDataset(tensordata[trainidx:], tensortarget[trainidx:])
    return traindataset, testdataset, featurenames


class BaseDataIterator:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: AbstractDataset, batchsize: int) -> None:
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

    def __init__(self, dataset: AbstractDataset, batchsize: int) -> None:
        super().__init__(dataset, batchsize)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(Y)
        else:
            raise StopIteration
