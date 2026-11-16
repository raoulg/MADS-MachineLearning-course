"""What lesson 1 gives the study of `study.py`: the caches it trains on, the trainer, the
linear baseline and the params a run logs. The study itself (the budget, the search, the
winner rerun, the comparison in sigmas) is `hpo.exercise.run_study`, the same in every
lesson.

It is the machinery of `05_exercises.ipynb`, written for any design: a design comes in
as `make_model(config, data)`, which builds the model a drawn config chooses.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from notebooktester import param
from torch import nn
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.data import Collate, OneHot

from hpo.exercise import Baseline, Data, Fit, OnEpoch, train
from hpo.optunalog import CHOICES

# The same as in 05_exercises.ipynb, so Part 1's trials say what one trial costs here.
EPOCHS = param(15, test=1)
REPEATS = param(3, test=2)

# Every published cache, by dataset and encoder.
ENCODERS = ["dinov2-small", "resnet-18", "mobilenet_v2_1.0_224"]
CACHES = {
    "dtd": {
        enc: f"pttrn-io/dtd-{enc}"
        for enc in [
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "resnet-18",
            "mobilenet_v2_1.0_224",
        ]
    },
    "eurosat": {enc: f"pttrn-io/eurosat-{enc}" for enc in ENCODERS},
    "oxford-flowers": {enc: f"pttrn-io/oxford-flowers-{enc}" for enc in ENCODERS},
    "fashion-mnist": {
        enc: f"pttrn-io/fashion-mnist-{enc}"
        for enc in ["dinov2-small", "mobilenet_v2_1.0_224"]
    },
}

# The values a string param can take, for the runs that are given their params (see
# hpo.optunalog.log_run).
CHOICES.update(
    dataset=list(CACHES),
    encoder=sorted({enc for encoders in CACHES.values() for enc in encoders}),
    activation=["relu", "gelu", "tanh", "sigmoid", "none"],
    norm=["none", "batch", "layer"],
)

# The linear baseline's choices, under the names the class view puts on its axes.
LINEAR = {
    "n_hidden_layers": 0,
    "hidden_dim": 0,
    "activation": "none",
    "norm": "none",
    "dropout": 0.0,
    "sigma": 0.0,
    "skip": False,
    "gate": False,
    "lr": 1e-3,
}
BASELINE_ENCODER = "dinov2-small"


def load(dataset: str, encoder: str, split: str) -> VectorCache:
    if encoder not in CACHES.get(dataset, {}):
        raise KeyError(
            f"no published cache for {dataset!r} with {encoder!r}; "
            f"CACHES has { ({d: list(e) for d, e in CACHES.items()}) }"
        )
    return VectorCache.from_hub(CACHES[dataset][encoder], split=split)


@cache
def classes_of(dataset: str, encoder: str) -> list[str]:
    """The names of the classes of a cache, in the order of their labels."""
    return list(load(dataset, encoder, "train").dataset.features["label"].names)


@cache
def shape_of(dataset: str, encoder: str) -> tuple[int, int]:
    """(vector width, number of classes) of a cache."""
    train = load(dataset, encoder, "train")
    return train.dim, train.num_classes


def loaders(
    dataset: str, encoder: str, batch_size: int = 64, n_train: int | None = None
) -> list[DataLoader]:
    """The train and test loaders of a cache. `n_train` takes a shuffled subset of the
    training set: the DTD cache is stored grouped by label, so its first rows are a few
    classes, not a smaller sample of all of them."""
    out = []
    for split in ("train", "test"):
        vectors = load(dataset, encoder, split)
        ds = vectors.dataset
        if n_train is not None and split == "train":
            ds = ds.shuffle(seed=0).select(range(n_train))
        oh = OneHot(
            num_classes=vectors.num_classes, label_col="label", target_col="onehot"
        )
        col = Collate(embedding_col="embed", target_col="onehot", padder=torch.stack)
        out.append(
            DataLoader(
                ds.map(oh),
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=col,
            )
        )
    return out


@dataclass
class Lesson1:
    """Lesson 1's harness for `hpo.exercise.run_study`: a head over cached vectors."""

    lesson: int = 1
    epochs: int = EPOCHS
    repeats: int = REPEATS
    # the keys of a search space the harness reads; every other key is the design's
    trainer: frozenset[str] = frozenset(
        {"dataset", "encoder", "epochs", "lr", "n_train"}
    )

    def data(self, config: Mapping[str, Any]) -> Data:
        dataset, encoder = config["dataset"], config["encoder"]
        tr, te = loaders(dataset, encoder, n_train=config.get("n_train"))
        dim, _ = shape_of(dataset, encoder)
        return Data(
            train=tr,
            test=te,
            classes=classes_of(dataset, encoder),
            example=torch.zeros(2, dim),
            dims={"embed_dim": dim},
        )

    def fit(
        self,
        model: nn.Module,
        data: Data,
        config: Mapping[str, Any],
        on_epoch: OnEpoch | None = None,
    ) -> Fit:
        return train(
            model,
            data.train,
            data.test,
            data.classes,
            lr=config.get("lr", 1e-3),
            epochs=config.get("epochs", EPOCHS),
            on_epoch=on_epoch,
        )

    def baseline(self, dataset: str) -> Baseline:
        """The linear head on `BASELINE_ENCODER`."""
        return Baseline(
            config={
                **LINEAR,
                "dataset": dataset,
                "encoder": BASELINE_ENCODER,
                "epochs": EPOCHS,
            },
            make=lambda data: nn.Linear(data.dims["embed_dim"], len(data.classes)),
        )

    def params(self, config: Mapping[str, Any], data: Data) -> dict[str, Any]:
        """The params of a run: the vectors it starts from, its choices, the budget."""
        return {"epochs": EPOCHS, **config, **data.dims}


HARNESS = Lesson1()
