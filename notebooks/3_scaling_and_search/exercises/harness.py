"""What lesson 3 gives the study of `study.py`: the seven-domain image mix, the trainer,
the linear baseline and the params a run logs. The study itself (the budget, the search,
the winner rerun, the difference with the baseline in points) is
`hpo.exercise.Exercise`, the same in every lesson.

The data is `pttrn-io/mix7-dinov2-small` (or `-base`, the axis `encoder`): every image of
seven datasets as two DINOv2 vectors, the CLS token and the mean of the patch tokens,
with one label out of 321 classes. A run trains on a stratified `share` of the train
rows of every class and is scored on a validation split cut from train: 10% of every
class, the same rows for every run. Test is left alone: it is for the end of the course,
not for choosing a design.

The trainer axes, besides `lr`:

- `encoder`: `dinov2-small` (384 numbers per vector) or `dinov2-base` (768);
- `share`: the share of every class's train rows a run trains on, between 0 and 1; a
  smaller share is a subset of a larger one;
- `drop_domain`: one domain left out of train and validation, or `"none"`;
- `form`: the two vectors as `(batch, 2, dim)` (`"flat"`), or each folded into a grid
  (`"grid"`), `(batch, 2, 16, 24)` for small and `(batch, 2, 24, 32)` for base.

An epoch is a fixed number of optimizer steps, one pass over every train row, whatever
the share: on a share of 10% an epoch passes over its rows ten times. So the epochs of
two runs are the same compute. A run trains until its validation accuracy has not
improved for `PATIENCE` epochs, or for `EPOCHS` epochs at most, and keeps the weights of
its best epoch.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import torch
from notebooktester import param
from torch import Tensor, nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset
from vectormesh import VectorCache

from hpo.exercise import Baseline, Data
from hpo.measure import Confusion, Fit, OnEpoch, Training
from hpo.record import Store
from hpo.search import Config, Param

EPOCHS = param(150, test=1)  # the most epochs a run trains
# the epochs in a row without a better validation accuracy after which a run stops
PATIENCE = param(10, test=1)
REPEATS = 1  # a run takes minutes: one seed, as an expensive run is run once
BATCH = 256

CACHES = {
    "dinov2-small": ("pttrn-io/mix7-dinov2-small", "31320bc67d31dd9c8d12e91a0a0c85cdac337278"),
    "dinov2-base": ("pttrn-io/mix7-dinov2-base", "2a0c3239f2e9fc6b2ea0027ecc3d54c595d740db"),
}  # fmt: skip
# every vector folded row by row into a grid: 384 = 16 x 24, 768 = 24 x 32
GRIDS = {384: (16, 24), 768: (24, 32)}
DOMAINS = [
    "dtd",
    "eurosat",
    "oxford-flowers",
    "food101",
    "gtsrb",
    "galaxy10-decals",
    "bloodmnist",
]
VALID_SHARE = 0.1  # of every class, cut from train once, with a fixed seed

# The values a string param can take, for the runs that are given their params (see
# hpo.record.StudyLog).
CHOICES: dict[str, list[str]] = {
    "dataset": ["mix7"],
    "encoder": list(CACHES),
    "form": ["flat", "grid"],
    "drop_domain": ["none", *DOMAINS],
    "input": ["cls", "mean", "both"],
    "norm": ["none", "batch", "layer"],
    "pool": ["max", "mean"],
}

# The linear baseline's choices, under the names the class view puts on its axes.
LINEAR: Config = {
    "dataset": "mix7",
    "encoder": "dinov2-small",
    "share": 1.0,
    "drop_domain": "none",
    "form": "flat",
    "input": "cls",
    "width": 0,
    "depth": 0,
    "norm": "none",
    "dropout": 0.0,
    "kernel": False,
    "skip": False,
    "gate": False,
    "moe": False,
    "lr": 1e-3,
}


@dataclass(frozen=True)
class Rows:
    """One split of the mix as arrays: `embed` `(n, 2, dim)` (CLS, patch mean), the
    label and the domain of every row."""

    embed: np.ndarray
    label: np.ndarray
    domain: np.ndarray


@cache
def load(encoder: str, split: str = "train") -> Rows:
    """Every row of one split of the mix, in the cache's order (domain by domain)."""
    if encoder not in CACHES:
        raise KeyError(
            f"no mix cache for the encoder {encoder!r}; CACHES has {list(CACHES)}"
        )
    repo, revision = CACHES[encoder]
    columns = VectorCache.from_hub(repo, split=split, revision=revision).dataset
    columns = columns.with_format("numpy")[:]
    return Rows(
        embed=np.asarray(columns["embed"], dtype=np.float32),
        label=np.asarray(columns["label"]),
        domain=np.asarray(columns["domain"]),
    )


@cache
def classes() -> list[str]:
    """The 321 class names, `<domain>/<class>`, in the order of their labels."""
    repo, revision = CACHES["dinov2-small"]
    cache_ = VectorCache.from_hub(repo, split="train", revision=revision)
    return list(cache_.dataset.features["label"].names)


@cache
def order(encoder: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """The validation rows, and per class its train rows in the order a share takes
    them: shuffled once with a fixed seed, so a share of 0.1 keeps the first tenth of
    every class and is a subset of a share of 0.3. Both encoders hold the same images in
    the same order, so they cut the same rows."""
    label = load(encoder).label
    rng = np.random.default_rng(0)
    valid, train = [], []
    for c in range(label.max() + 1):
        rows = rng.permutation(np.flatnonzero(label == c))
        n_valid = max(1, round(VALID_SHARE * len(rows)))
        valid.append(rows[:n_valid])
        train.append(rows[n_valid:])
    return np.sort(np.concatenate(valid)), train


def cut(
    encoder: str, share: float, drop_domain: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    """The train rows of a run and the validation rows it is scored on: a stratified
    `share` of every class, without the domain `drop_domain`."""
    if not 0 < share <= 1:
        raise ValueError(
            f"share is a fraction of the train rows, in (0, 1], not {share}"
        )
    if drop_domain != "none" and drop_domain not in DOMAINS:
        raise KeyError(
            f"drop_domain {drop_domain!r} is no domain of the mix: {DOMAINS}"
        )
    valid, per_class = order(encoder)
    train = np.sort(
        np.concatenate([rows[: max(1, round(share * len(rows)))] for rows in per_class])
    )
    if drop_domain != "none":
        domain = load(encoder).domain
        dropped = DOMAINS.index(drop_domain)
        train, valid = train[domain[train] != dropped], valid[domain[valid] != dropped]
    return train, valid


def fold(embed: Tensor) -> Tensor:
    """`(n, 2, dim)` to `(n, 2, height, width)`: every vector folded row by row into
    the grid of `GRIDS`, as lesson 2 folded a beat."""
    return embed.reshape(len(embed), 2, *GRIDS[embed.shape[-1]])


def tensors(
    encoder: str, share: float, drop_domain: str = "none", form: str = "flat"
) -> tuple[TensorDataset, TensorDataset]:
    """The train and validation rows of a run as tensors, every number standardised with
    the mean and the spread of the train rows."""
    if form not in CHOICES["form"]:
        raise KeyError(f"form {form!r}: the loader gives {CHOICES['form']}")
    train, valid = cut(encoder, share, drop_domain)
    rows = load(encoder)
    mean = rows.embed[train].mean(0)
    std = rows.embed[train].std(0) + 1e-6

    def split(at: np.ndarray) -> TensorDataset:
        embed = torch.from_numpy((rows.embed[at] - mean) / std)
        embed = fold(embed) if form == "grid" else embed
        return TensorDataset(embed, torch.from_numpy(rows.label[at]).long())

    return split(train), split(valid)


def loader(rows: TensorDataset, steps: int | None = None) -> DataLoader:
    """Batches of `BATCH` rows: `steps` batches drawn in random order (passing over the
    rows as often as that takes), or every row once, in order. A batch is cut from the
    tensors in one index, not row by row."""
    if steps is None:
        batches = BatchSampler(range(len(rows)), BATCH, drop_last=False)
    else:
        draw = RandomSampler(rows, num_samples=steps * BATCH)
        batches = BatchSampler(draw, BATCH, drop_last=False)
    return DataLoader(rows, sampler=batches, batch_size=None)


@dataclass
class Patience:
    """Stops a run once its validation accuracy has not improved for `patience` epochs,
    and keeps the weights and the scores of its best epoch."""

    model: nn.Module
    patience: int
    best: float = -1.0
    epoch: int = 0  # the kept epoch, counted from 1
    waited: int = 0
    state: dict[str, Tensor] = field(default_factory=dict)
    confusion: Confusion | None = None
    fit: Fit = field(default_factory=Fit)

    def see(self, epoch: int, fit: Fit) -> bool:
        """Record the fit after `epoch` (counted from 0); True when the run should stop."""
        accuracy = fit.scores().accuracy
        self.fit = fit
        if accuracy > self.best:
            self.best, self.epoch, self.waited = accuracy, epoch + 1, 0
            self.state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            self.confusion = fit.confusion
        else:
            self.waited += 1
        return self.waited >= self.patience

    def kept(self) -> Fit:
        """The fit with the weights and the confusion of the kept epoch."""
        if self.state:
            self.model.load_state_dict(self.state)
        return self.fit.model_copy(update={"confusion": self.confusion})


class Settled(Exception):
    """Raised after an epoch to end a run whose validation accuracy stopped improving."""


class Linear(nn.Module):
    """The baseline: one linear layer on the CLS vector, from either form of the loader."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(2)[:, 0])


@dataclass
class Lesson3:
    """Lesson 3's harness for `hpo.exercise.Exercise`: a head over the mix's vectors."""

    lesson: int = 3
    epochs: int = EPOCHS
    repeats: int = REPEATS
    patience: int = PATIENCE
    # the keys of a search space the harness reads; every other key is the design's
    trainer: frozenset[str] = frozenset(
        {"dataset", "encoder", "share", "drop_domain", "form", "epochs", "lr"}
    )
    choices: Mapping[str, Sequence[str]] = field(default_factory=lambda: CHOICES)
    store: Store = field(default_factory=Store.default)

    def data(self, config: Config) -> Data:
        if config.get("dataset", "mix7") != "mix7":
            raise KeyError(
                f"lesson 3 has one dataset, 'mix7', not {config['dataset']!r}"
            )
        encoder = str(config.get("encoder", "dinov2-small"))
        form = str(config.get("form", "flat"))
        train, valid = tensors(
            encoder,
            float(config.get("share", 1.0)),
            str(config.get("drop_domain", "none")),
            form,
        )
        dim = int(load(encoder).embed.shape[-1])
        n_rows = sum(len(rows) for rows in order(encoder)[1])
        steps = -(-n_rows // BATCH)  # one pass over every train row
        height, width = GRIDS[dim]
        return Data(
            train=loader(train, steps),
            test=loader(valid),
            classes=classes(),
            example=torch.zeros(2, *train.tensors[0].shape[1:]),
            dims={
                "dim": dim,
                "height": height,
                "breadth": width,
                "n_train": len(train),
                "steps_per_epoch": steps,
            },
        )

    def fit(
        self,
        model: nn.Module,
        data: Data,
        config: Config,
        on_epoch: OnEpoch | None = None,
    ) -> Fit:
        """Train `model` until its validation accuracy settles, and keep its best epoch."""
        patience = Patience(model, self.patience)

        def watch(epoch: int, fit: Fit) -> None:
            if on_epoch is not None:
                on_epoch(epoch, fit)
            if patience.see(epoch, fit):
                raise Settled

        training = Training(
            lr=float(config.get("lr", 1e-3)),
            epochs=int(config.get("epochs", EPOCHS)),
            logdir=self.store.logdir(),
            device="cpu",
        )
        try:
            training.fit(model, data.train, data.test, data.classes, watch)
        except Settled:
            pass
        return patience.kept()

    def baseline(self, dataset: str) -> Baseline:
        """A linear layer on the CLS vector of `dinov2-small`, every train row."""
        return Baseline(
            config={**LINEAR, "dataset": dataset, "epochs": EPOCHS},
            make=lambda data: Linear(data.dims["dim"], len(data.classes)),
        )

    def params(self, config: Config, data: Data) -> dict[str, Param]:
        """The params of a run: its choices, the budget, the size of its data."""
        return {"epochs": EPOCHS, **config, **data.dims}


def kept_epoch(fit: Fit) -> int:
    """The epoch whose weights a run kept, counted from 1: its best validation accuracy."""
    return int(np.argmax(fit.curves.accuracy)) + 1


HARNESS = Lesson3()
