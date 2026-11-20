"""What lesson 2 gives the study of `study.py`: the heartbeats it trains on, the trainer,
the linear baseline and the params a run logs. The study itself (the budget, the search,
the winner rerun, the comparison in sigmas) is `hpo.exercise.Exercise`, the same in every
lesson.

The data is MIT-BIH, split by patient (`pttrn-io/heartbeats-mitbih-raw`): three classes,
N, S and V. A study trains on train without the patients of `VAL_RECORDS`, and is scored
on those six patients, whom the model never saw. Test is left alone: it is for the end
of the course, not for choosing a design.

Imbalance is an axis: `class_weight` weighs every class in the loss by
`(n / (k * n_class)) ** class_weight`, so 0 is the plain loss and 1 weighs every class as
much as the others, however rare.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import torch
from notebooktester import param
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from vectormesh import VectorCache

from hpo.exercise import Baseline, Data
from hpo.measure import Fit, OnEpoch, Training
from hpo.record import Store
from hpo.search import Config, Param

EPOCHS = param(8, test=1)
REPEATS = param(3, test=2)
# train beats under notebooktester; the patients scored on are always all six
TRAIN_ROWS = param(None, test=2000)

REPO = "pttrn-io/heartbeats-mitbih-raw"
REVISION = "b03962404ac3081ec1e3679171ea3a108d5e3601"
CLASSES = ["N", "S", "V"]
# six patients of train with N, S and V among their beats
VAL_RECORDS = (118, 124, 201, 207, 215, 223)
STEPS = 187

# The values a string param can take, for the runs that are given their params (see
# hpo.record.StudyLog).
CHOICES: dict[str, list[str]] = {
    "dataset": ["mitbih"],
    "downsample": ["pool", "stride"],
    "norm": ["none", "batch", "layer"],
}

# The linear baseline's choices, under the names the class view puts on its axes.
LINEAR: Config = {
    "kernel_size": 0,
    "width": 0,
    "depth": 0,
    "downsample": "pool",
    "norm": "none",
    "sigma": 0.0,
    "dropout": 0.0,
    "parallel": False,
    "fold": False,
    "skip": False,
    "excite": False,
    "state": False,
    "gate": False,
    "class_weight": 0.0,
    "lr": 1e-3,
}


@dataclass(frozen=True)
class Split:
    """One split as tensors: beats `(n, 187, 1)`, labels `(n,)`."""

    beats: Tensor
    labels: Tensor

    def counts(self) -> np.ndarray:
        return np.bincount(self.labels.numpy(), minlength=len(CLASSES))

    def loader(self, shuffle: bool, batch_size: int = 128) -> DataLoader:
        return DataLoader(
            TensorDataset(self.beats, self.labels),
            batch_size=batch_size,
            shuffle=shuffle,
        )


@cache
def splits(max_rows: int | None) -> tuple[Split, Split]:
    """The train split and the validation split, both from the hub's split "train":
    train without `VAL_RECORDS`, and those six patients. The hub's split "test" is left
    for the class problem. `max_rows` keeps that many train beats, drawn with a fixed
    seed."""
    source = VectorCache.from_hub(REPO, split="train", revision=REVISION)
    columns = source.dataset.with_format("numpy")[:]
    beats, labels, records = columns["beat"], columns["label"], columns["record"]
    in_valid = np.isin(records, VAL_RECORDS)
    train_rows = np.flatnonzero(~in_valid)
    if max_rows is not None and max_rows < len(train_rows):
        train_rows = np.sort(
            np.random.default_rng(0).choice(train_rows, max_rows, False)
        )

    def split(rows: np.ndarray) -> Split:
        return Split(
            torch.from_numpy(beats[rows]), torch.from_numpy(labels[rows]).long()
        )

    return split(train_rows), split(np.flatnonzero(in_valid))


def weights(counts: np.ndarray, power: float) -> Tensor:
    """The weight of every class in the loss: `(n / (k * n_class)) ** power`."""
    balanced = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(balanced**power, dtype=torch.float32)


@dataclass
class Lesson2:
    """Lesson 2's harness for `hpo.exercise.Exercise`: a model of one heartbeat."""

    lesson: int = 2
    epochs: int = EPOCHS
    repeats: int = REPEATS
    # the keys of a search space the harness reads; every other key is the design's
    trainer: frozenset[str] = frozenset(
        {"dataset", "epochs", "lr", "class_weight", "n_train"}
    )
    choices: Mapping[str, Sequence[str]] = field(default_factory=lambda: CHOICES)
    store: Store = field(default_factory=Store.default)

    def data(self, config: Config) -> Data:
        if config.get("dataset", "mitbih") != "mitbih":
            raise KeyError(
                f"lesson 2 has one dataset, 'mitbih', not {config['dataset']!r}"
            )
        n_train = config.get("n_train", TRAIN_ROWS)
        train, valid = splits(int(n_train) if n_train is not None else None)
        return Data(
            train=train.loader(shuffle=True),
            test=valid.loader(shuffle=False),
            classes=CLASSES,
            example=torch.zeros(2, STEPS, 1),
            dims={"steps": STEPS, "channels": 1},
        )

    def fit(
        self,
        model: nn.Module,
        data: Data,
        config: Config,
        on_epoch: OnEpoch | None = None,
    ) -> Fit:
        labels = data.train.dataset.tensors[1]  # ty: ignore[missing-attribute]  (a TensorDataset)
        counts = np.bincount(labels.numpy(), minlength=len(data.classes))
        loss_fn = nn.CrossEntropyLoss(
            weight=weights(counts, float(config.get("class_weight", 0.0)))
        )
        training = Training(
            lr=float(config.get("lr", 1e-3)),
            epochs=int(config.get("epochs", EPOCHS)),
            logdir=self.store.logdir(),
        )
        return training.fit(
            model, data.train, data.test, data.classes, on_epoch, loss_fn=loss_fn
        )

    def baseline(self, dataset: str) -> Baseline:
        """A linear model on the 187 steps of the beat, with the plain loss."""
        return Baseline(
            config={**LINEAR, "dataset": dataset, "epochs": EPOCHS},
            make=lambda data: nn.Sequential(
                nn.Flatten(), nn.Linear(data.dims["steps"], len(data.classes))
            ),
        )

    def params(self, config: Config, data: Data) -> dict[str, Param]:
        """The params of a run: its choices, the budget, the shape of a beat."""
        return {"epochs": EPOCHS, **config, **data.dims}


HARNESS = Lesson2()
