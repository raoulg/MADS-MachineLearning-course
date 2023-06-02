from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import gin
from loguru import logger
from pydantic import BaseModel, HttpUrl, root_validator
from ray import tune

from src.models import tokenizer

if TYPE_CHECKING:
    from src.models.metrics import Metric

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float


class FileTypes(Enum):
    JPG = ".jpg"
    PNG = ".png"
    TXT = ".txt"
    ZIP = ".zip"
    TGZ = ".tgz"
    TAR = ".tar.gz"
    GZ = ".gz"


class ReportTypes(Enum):
    GIN = 1
    TENSORBOARD = 2
    MLFLOW = 3
    RAY = 4


@gin.configurable
class TrainerSettings(BaseModel):
    epochs: int
    metrics: List[Callable]
    logdir: Path
    train_steps: int
    valid_steps: int
    reporttypes: List[ReportTypes]
    optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-5}
    scheduler_kwargs: Optional[Dict[str, Any]] = {"factor": 0.1, "patience": 10}
    earlystop_kwargs: Optional[Dict[str, Any]] = {
        "save": False,
        "verbose": True,
        "patience": 10,
    }

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("logdir").resolve()
        if not datadir.exists():
            logger.info(f"logdir did not exist. Creating at {datadir}.")
            datadir.mkdir(parents=True)
        return values


class BaseSearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


# this is what ray will use to create configs
class SearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 5)


class BaseSettings(BaseModel):
    data_dir: Path

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


cwd = Path(__file__).parent
cwd = (cwd / "../").resolve()


class GeneralSettings(BaseSettings):
    data_dir: Path = cwd / "data/raw"


class DatasetSettings(GeneralSettings):
    dataset_url: HttpUrl
    filename: Path
    name: str
    formats: List[FileTypes]


class ImgDatasetSettings(DatasetSettings):
    trainfrac: float
    img_size: Tuple[int, int]


class TextDatasetSettings(DatasetSettings):
    maxvocab: int
    maxtokens: int
    clean_fn: Callable


class WindowedDatasetSettings(DatasetSettings):
    horizon: int
    window_size: int


sunspotsettings = WindowedDatasetSettings(
    dataset_url=cast(HttpUrl, "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"),
    filename=Path("sunspots.txt"),
    name="sunspots",
    formats=[],
    horizon=3,
    window_size=26,
)

fashionmnistsettings = DatasetSettings(
    dataset_url=cast(HttpUrl, "https://github.com/zalandoresearch/fashion-mnist"),
    filename=Path(""),
    name="fashion",
    formats=[],
)

gesturesdatasetsettings = DatasetSettings(
    dataset_url=cast(
        HttpUrl, "https://github.com/raoulg/gestures/raw/main/gestures-dataset.zip"
    ),
    filename=Path("gestures.zip"),
    name="gestures",
    formats=[FileTypes.TXT],
)

flowersdatasetsettings = ImgDatasetSettings(
    dataset_url=cast(
        HttpUrl,
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    ),
    filename=Path("flowers.tgz"),
    name="flowers",
    formats=[FileTypes.JPG],
    trainfrac=0.8,
    img_size=(224, 224),
)

imdbdatasetsettings = TextDatasetSettings(
    dataset_url=cast(
        HttpUrl, "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    ),
    filename=Path("aclImdb_v1.tar.gz"),
    name="imdb",
    formats=[FileTypes.TXT],
    maxvocab=10000,
    maxtokens=100,
    clean_fn=tokenizer.clean,
)


class VAESettings(GeneralSettings):
    h1: int = 250
    h2: int = 100
    insize: int = 784
    latent: int = 10
    batchsize: int = 32
    epochs: int = 100
    modelname: Path = Path("vaemodel.pt")
    imgpath: Path = Path("img")
    samplesize: int = 512


class SiameseSettings(GeneralSettings):
    url: HttpUrl = "https://github.com/maticvl/dataHacker/raw/master/DATA/at%26t.zip"  # type: ignore
    filename: Path = Path("faces.zip")
    training: Path = Path("data/faces/training/")


class EurosatSettings(BaseSettings):
    data_dir = Path("../data/raw")
    valid_paths = Path("ds")


class StyleSettings(BaseSettings):
    data_dir = Path("../../data/external")
    trainpath = Path("../../data/external/sentences/train.feather")
    testpath = Path("../../data/external/sentences/test.feather")


class ImageExplainerSettings(GeneralSettings):
    datadir: Path = Path("../../../data/raw/")
    epochs: int = 10
    modelname: Path = Path("trained_model.pt")
    imgpath: Path = Path("img")
    num_classes: int = 10
