"""The machinery `study.py` runs on: the caches, the trainer that counts its seconds, the
spread between runs, and the objective an Optuna study calls once per trial.

It is the machinery of `05_exercises.ipynb`, written for any design: a design comes in
as `make_model(config, dim, n_classes)`, which builds the model a drawn config chooses.
"""

import sys
import time
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from notebooktester import param
from torch import nn
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.components.metrics import Accuracy
from vectormesh.data import Collate, OneHot
from vectormesh.training import Trainer, TrainerSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optunalog import (
    CHOICES,
    draw,
    record,
)  # the lesson folder is put on sys.path above

MakeModel = Callable[[Mapping[str, Any], int, int], nn.Module]

# The same as in 05_exercises.ipynb, so a baseline of Part 1 can stand in for this one,
# and Part 1's trials say what one trial costs here.
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
# optunalog.log_run).
CHOICES.update(
    dataset=list(CACHES),
    encoder=sorted({enc for encoders in CACHES.values() for enc in encoders}),
    activation=["relu", "gelu", "tanh", "sigmoid", "none"],
    norm=["none", "batch", "layer"],
)

# The keys of a search space the trainer uses; every other key is the design's.
TRAINER = {"dataset", "encoder", "epochs", "lr", "n_train"}

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


def check_space(
    space: Mapping[str, Any], options: Iterable[str], design: Iterable[str]
) -> None:
    """A `ValueError` unless every option of the design is an axis of `space`, and every
    key of `space` is used by the design (`design`, the names of its choices) or by the
    trainer (`TRAINER`)."""
    for axis in options:
        if axis not in space:
            raise ValueError(
                f"SPACE has no axis {axis!r}, which designs.py marks @option({axis!r}): "
                f"add it to SPACE, as {axis!r}: Choice([False, True]) to search it or "
                f"{axis!r}: False to leave the stage out of every trial"
            )
    used = set(design) | TRAINER
    for key in space:
        if key not in used:
            raise ValueError(
                f"SPACE has the axis {key!r}, which neither the design nor the trainer "
                f"uses: give Config in designs.py a field {key!r} and a stage that reads "
                "it, or take it out of SPACE"
            )


def load(dataset: str, encoder: str, split: str) -> VectorCache:
    if encoder not in CACHES.get(dataset, {}):
        raise KeyError(
            f"no published cache for {dataset!r} with {encoder!r}; "
            f"CACHES has { ({d: list(e) for d, e in CACHES.items()}) }"
        )
    return VectorCache.from_hub(CACHES[dataset][encoder], split=split)


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


class TimedTrainer(Trainer):
    """A Trainer that adds up the seconds spent in training steps. Evaluation after every
    epoch is not counted: `train_seconds` is the cost of fitting, not of scoring."""

    train_seconds = 0.0

    def trainbatches(self) -> float:
        start = time.perf_counter()
        loss = super().trainbatches()
        self.train_seconds += time.perf_counter() - start
        return loss


def trainer(
    model: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    lr: float,
    epochs: int,
    reporters: list[Callable[..., None]] | None = None,
) -> TimedTrainer:
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[Accuracy()],
        logdir=Path("modellogs").resolve(),  # never created unless earlystop saves
        train_steps=len(train_dl),
        valid_steps=len(test_dl),
        optimizer_kwargs={"lr": lr},
        earlystop_kwargs=None,
    )
    return TimedTrainer(
        model=model,
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train_dl,
        validdataloader=test_dl,
        progress=False,
        reporters=reporters or [],
    )


def run(
    model: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[float, float]:
    """Train `model` and return (test accuracy, seconds spent in training steps)."""
    fit = trainer(model, train_dl, test_dl, lr, epochs)
    return fit.loop().metric_dict["Accuracy"], fit.train_seconds


def run_spread(
    make: Callable[[], nn.Module],
    train_dl: DataLoader,
    test_dl: DataLoader,
    repeats: int = REPEATS,
    **kwargs: Any,
) -> tuple[np.ndarray, float]:
    """Accuracy of `repeats` runs, each from other random weights (nothing is seeded), and
    the mean seconds spent in training steps."""
    scores, seconds = [], []
    for _ in range(repeats):
        accuracy, spent = run(make(), train_dl, test_dl, **kwargs)
        scores.append(accuracy)
        seconds.append(spent)
    return np.array(scores), float(np.mean(seconds))


def compare(name: str, scores: np.ndarray, versus: np.ndarray | None = None) -> float:
    """Print the mean and spread of `scores`; against `versus`, also the difference in
    points and in sigmas, which it returns (0.0 without `versus`)."""
    line = f"{name:<32} {scores.mean():.4f} +/- {scores.std():.4f}   (n={len(scores)})"
    sigmas = 0.0
    if versus is not None:
        diff = scores.mean() - versus.mean()
        pooled = np.sqrt(
            scores.var(ddof=1) / len(scores) + versus.var(ddof=1) / len(versus)
        )
        sigmas = float(diff / pooled) if pooled > 0 else 0.0
        line += f"   {100 * diff:+.2f} pt, {sigmas:+.1f} sigma"
    print(line)
    return sigmas


def run_params(
    dataset: str, encoder: str, config: Mapping[str, Any], **extra: Any
) -> dict[str, Any]:
    """The params of a run: the vectors it starts from, its choices, the budget."""
    return {
        "epochs": EPOCHS,
        **config,
        "dataset": dataset,
        "encoder": encoder,
        "embed_dim": shape_of(dataset, encoder)[0],
        **extra,
    }


def objective(
    space: Mapping[str, Any], make_model: MakeModel, design: Path
) -> Callable[[optuna.Trial], float]:
    """What an Optuna study calls once per trial: draw a config from `space`, build it
    with `make_model`, train it, report the accuracy after every epoch, and return the
    last one. Every trial carries `design`, the design script, as its artifact."""

    def run_trial(trial: optuna.Trial) -> float:
        config = draw(trial, space)
        tr, te = loaders(
            config["dataset"], config["encoder"], n_train=config.get("n_train")
        )
        dim, n_classes = shape_of(config["dataset"], config["encoder"])
        model = make_model(config, dim, n_classes)
        curves: dict[str, list[float]] = {"train_loss": [], "test_loss": []}

        # every epoch reaches the store, and the study's pruner can stop a trial that is
        # behind the others
        def to_optuna(epoch, train_loss, test_loss, metric_dict) -> None:
            curves["train_loss"].append(train_loss)
            curves["test_loss"].append(test_loss)
            trial.report(metric_dict["Accuracy"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit = trainer(model, tr, te, config["lr"], config["epochs"], [to_optuna])
        try:
            return fit.loop().metric_dict["Accuracy"]
        finally:
            # a pruned trial is recorded as well, with the epochs it ran
            record(
                trial,
                "trial",
                dataset=config["dataset"],
                model=model,
                example=torch.zeros(2, dim),
                train_seconds=fit.train_seconds,
                curves=curves,
                attrs={"embed_dim": dim},
                design=design,
            )

    return run_trial
