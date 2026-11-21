"""What lesson 3 gives the study of `study.py`: the heartbeats it trains on, the trainer,
the linear baseline, the params a run logs, and the readouts of a capacity study (the
curves with their control, and the control at the winner). The study itself (the budget,
the search, the winner rerun, the comparison in sigmas) is `hpo.exercise.Exercise`, the
same in every lesson.

The data is lesson 2's: MIT-BIH, split by patient (`pttrn-io/heartbeats-mitbih-raw`),
three classes, N, S and V. A study trains on train without the patients of
`VAL_RECORDS`, and is scored on those six patients, whom the model never saw. Test is
left alone: it is for the end of the course, not for choosing a design.

The three capacity axes of a run:

- parameters: `width` and `depth` of the design, logged as `n_params`;
- data: `train_fraction`, the share of the training beats a run trains on. The beats of
  every class are shuffled once, with a fixed seed, and a fraction takes the first part
  of every class, so a run on 0.2 trains on the same mix of N, S and V as one on 1.0, and
  on a subset of the beats of a run on 0.5;
- compute: `epochs`, the passes over those beats.

Imbalance is an axis as in lesson 2: `class_weight` weighs every class in the loss by
`(n / (k * n_class)) ** class_weight`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import cast

import numpy as np
import optuna
import pandas as pd
import torch
from notebooktester import param
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from vectormesh import VectorCache

from hpo.exercise import Baseline, Data, Exercise, Outcome
from hpo.measure import Architecture, Comparison, Fit, OnEpoch, Training
from hpo.record import Role, Store, StudyLog
from hpo.search import Config, Param

EPOCHS = param(8, test=1)
REPEATS = param(3, test=2)
# train beats under notebooktester, before a train_fraction is taken; the patients
# scored on are always all six
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
    "routing": ["dense", "sparse"],
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
    "n_experts": 0,
    "routing": "dense",
    "parallel": False,
    "fold": False,
    "skip": False,
    "excite": False,
    "state": False,
    "route": False,
    "matched": False,
    "gate": False,
    "class_weight": 0.0,
    "train_fraction": 1.0,
    "lr": 1e-3,
}

# the axes a capacity curve is read over: parameters, data, compute
CAPACITY = ("n_params", "train_fraction", "epochs")


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
    for the end of the course. `max_rows` keeps that many train beats, drawn with a fixed
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


@cache
def fraction(max_rows: int | None, train_fraction: float) -> Split:
    """`train_fraction` of the train beats, stratified: the beats of every class are
    shuffled once with a fixed seed, and the first `train_fraction` of every class is
    kept (at least one beat of each). A smaller fraction holds a subset of the beats of a
    larger one."""
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}: it is the share "
            "of the training beats a run trains on"
        )
    train, _ = splits(max_rows)
    if train_fraction == 1.0:
        return train
    labels = train.labels.numpy()
    rng = np.random.default_rng(0)
    kept = []
    for c in range(len(CLASSES)):
        rows = rng.permutation(np.flatnonzero(labels == c))
        kept.append(rows[: max(1, round(train_fraction * len(rows)))])
    rows = np.sort(np.concatenate(kept))
    return Split(train.beats[rows], train.labels[rows])


def weights(counts: np.ndarray, power: float) -> Tensor:
    """The weight of every class in the loss: `(n / (k * n_class)) ** power`."""
    balanced = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(balanced**power, dtype=torch.float32)


@dataclass
class Lesson3:
    """Lesson 3's harness for `hpo.exercise.Exercise`: a model of one heartbeat, trained
    on a share of the beats for a number of epochs."""

    lesson: int = 3
    epochs: int = EPOCHS
    repeats: int = REPEATS
    # the keys of a search space the harness reads; every other key is the design's
    trainer: frozenset[str] = frozenset(
        {"dataset", "epochs", "lr", "class_weight", "train_fraction"}
    )
    choices: Mapping[str, Sequence[str]] = field(default_factory=lambda: CHOICES)
    store: Store = field(default_factory=Store.default)

    def data(self, config: Config) -> Data:
        if config.get("dataset", "mitbih") != "mitbih":
            raise KeyError(
                f"lesson 3 has one dataset, 'mitbih', not {config['dataset']!r}"
            )
        train = fraction(TRAIN_ROWS, float(config.get("train_fraction", 1.0)))
        _, valid = splits(TRAIN_ROWS)
        return Data(
            train=train.loader(shuffle=True),
            test=valid.loader(shuffle=False),
            classes=CLASSES,
            example=torch.zeros(2, STEPS, 1),
            dims={"steps": STEPS, "channels": 1, "n_train": len(train.labels)},
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
        """A linear model on the 187 steps of the beat, with the plain loss, on every
        training beat."""
        return Baseline(
            config={**LINEAR, "dataset": dataset, "epochs": EPOCHS},
            make=lambda data: nn.Sequential(
                nn.Flatten(), nn.Linear(data.dims["steps"], len(data.classes))
            ),
        )

    def params(self, config: Config, data: Data) -> dict[str, Param]:
        """The params of a run: its choices, the budget, the shape of a beat and the
        number of beats it trained on."""
        return {"epochs": EPOCHS, **config, **data.dims}


HARNESS = Lesson3()


def arm(config: Mapping[str, object]) -> str:
    """Which arm of the capacity study a config is: `route` (the route stage), `matched`
    (its control of the same parameter count) or `plain` (neither)."""
    if not config.get("route"):
        return "plain"
    return "matched" if config.get("matched") else "route"


def trial_rows(log: StudyLog) -> pd.DataFrame:
    """The finished trials of a search, one row each: the objective, the arm, `n_params`,
    `n_train` and every param the trial drew or was given."""
    rows = []
    for t in log.trials(optuna.trial.TrialState.COMPLETE):
        config = t.user_attrs.get("config", {})
        rows.append(
            {
                "value": t.value,
                "arm": arm(config),
                "n_params": t.user_attrs.get("n_params"),
                "n_train": t.user_attrs.get("n_train"),
                **config,
            }
        )
    return pd.DataFrame(rows)


def curve(table: pd.DataFrame, axis: str, bins: int = 3) -> pd.DataFrame:
    """The objective along `axis`, per arm: the trials put in `bins` bins of about as
    many trials each (or one per value when the axis has few), and per bin and arm the
    mean, its standard error and the number of trials. `route - matched` is the
    difference of the two arms in standard errors, where both have two trials or more."""
    values = table[axis].astype(float)
    if values.nunique() <= bins:
        where = values
    else:
        where = pd.qcut(values, bins, duplicates="drop")
    out = {}
    for name, group in table.groupby([where, table["arm"]], observed=True):
        out[name] = {
            "mean": group["value"].mean(),
            "se": group["value"].std(ddof=1) / np.sqrt(len(group)),
            "n": len(group),
        }
    frame = cast(pd.DataFrame, pd.DataFrame(out).T.unstack(level=1)).swaplevel(axis=1)
    arms = [a for a in ("route", "matched", "plain") if a in frame.columns]
    frame = frame[[(a, stat) for a in arms for stat in ("mean", "se", "n")]]
    frame.index.name = axis
    if {"route", "matched"} <= set(arms):
        frame["route - matched", "sigma"] = [
            sigmas(table[(where == bin_) & (table["arm"] == "route")]["value"],
                   table[(where == bin_) & (table["arm"] == "matched")]["value"])
            for bin_ in frame.index
        ]  # fmt: skip
    return frame


def sigmas(a: pd.Series, b: pd.Series) -> float:
    """The difference of the means of `a` and `b` in standard errors, or NaN when either
    has fewer than two values."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(Comparison.of("a", a.to_numpy(), versus=b.to_numpy()).sigmas or 0.0)


def control(exercise: Exercise, outcome: Outcome) -> Comparison:
    """The winner's route stage against its capacity-matched control: the winner's config
    with `route` on, once with the router and once with `matched`, each run as often as
    the baseline and logged to the study's `-runs` (the route arm as `hypothesis`, the
    control as `ablation`). An arm that is the winner reuses the winner's runs."""
    best = outcome.best
    runs = StudyLog.runs(
        f"{exercise.name}-runs", exercise.questions.idea, exercise.objective,
        choices=CHOICES, store=exercise.store,
    )  # fmt: skip
    measured: dict[str, tuple[np.ndarray, int]] = {}
    arms: list[tuple[str, Role, dict]] = [
        (
            "route at the winner",
            "hypothesis",
            {**best, "route": True, "matched": False},
        ),
        ("matched control", "ablation", {**best, "route": True, "matched": True}),
    ]
    for name, role, config in arms:
        data = HARNESS.data(config)
        model = exercise.make_model(config, data)
        n_params = sum(p.numel() for p in model.parameters())
        if config == best:
            measured[name] = (np.asarray(outcome.winner.values), n_params)
            continue
        results = exercise.repeats(
            lambda config=config, data=data: exercise.make_model(config, data),
            data,
            config,
        )
        runs.log(
            name,
            role,
            results,
            params=HARNESS.params(config, data),
            dataset=str(config.get("dataset")),
            architecture=Architecture.of(model, data.example),
            design=exercise.script,
        )
        measured[name] = (results.values(exercise.objective), n_params)
    (route_name, (route, route_n)), (ctrl_name, (ctrl, ctrl_n)) = measured.items()
    against = Comparison.of(f"{route_name} ({route_n:,} params)", route, versus=ctrl)
    print()
    print(Comparison.of(f"{ctrl_name} ({ctrl_n:,} params)", ctrl))
    print(against)
    print(f"the route stage against its control: {against.sigmas:+.1f} sigma")
    return against


def read(exercise: Exercise, outcome: Outcome) -> None:
    """Print the capacity curves of the search, per arm, along every capacity axis it
    searched, and then the winner's route stage against its control."""
    table = trial_rows(StudyLog.load(exercise.name, exercise.store))
    searched = set(exercise.space.searched())
    # n_params moves with width and depth; the other two are axes of their own
    moved = {"n_params": {"width", "depth"} & searched}
    for axis in CAPACITY:
        if not (moved.get(axis) or axis in searched) or table.empty:
            continue
        print(f"\ncapacity curve over {axis}, per arm ({exercise.objective}):")
        print(curve(table, axis).round(3).to_string())
    control(exercise, outcome)
