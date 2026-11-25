"""What lesson 3 gives the study of `study.py`: the heartbeats it trains on, the trainer,
the linear baseline, the params a run logs, the pairing of the route stage with its
control, and the readouts of a capacity study (the paired curves, the use of the experts,
and the control at the winner). The study itself (the budget, the search, the winner
rerun, the comparison in sigmas) is `hpo.exercise.Exercise`, the same in every lesson;
`RoutedExercise` pairs its trials and adds the use of the experts to every trial.

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
- compute, counted here as the training examples a run sees: `epochs * n_train` (times
  the tiles below), logged as `train_examples`, and the optimizer steps as
  `train_steps`. Compute in floating-point operations also grows with the parameters: a
  run of width 64 costs many times one of width 8 at the same `train_examples`. At a
  fixed `epochs`, a smaller `train_fraction` is also fewer examples seen: the data axis
  and the compute axis move together. `equal_steps` (a trainer key, False by default)
  holds them apart: an epoch passes over the beats of the fraction about
  `1 / train_fraction` times (`tiles`), so every fraction gets about the steps of a run
  on every beat, and the validation patients are scored as often.

Imbalance is an axis as in lesson 2: `class_weight` weighs every class in the loss by
`(n / (k * n_class)) ** class_weight`.

Load balancing is an axis of the design (`balance` in `designs.Config`): the trainer adds
`balance * load_balancing_loss(model)` to the loss of every training step, never to the
validation loss. `load_balancing_loss` sums the
auxiliary loss of every `SparseMoE`, so it acts with `routing="sparse"` only; with a
dense `MoE` the trainer warns that the weight changes nothing.

The route stage and its control are compared in pairs (`paired`): a trial that draws the
route stage is followed by its twin, the same draw with `matched` flipped, from the same
seed, so the two differ in the stage and in the noise of training. The curves read the
difference within each pair (`pairs`, `paired_curve`), with the standard error of that
difference: the standard deviation of the differences over the square root of the
number of pairs.

The use of the experts (`expert_use`, `Use`) is read after training, on the validation
patients: per expert, the share of the positions whose router gives that expert its
largest weight, as notebook 3.1 counts it, and the router's weight for that expert
averaged over the positions. Even use is `1 / n_experts` per expert. Both are also taken
per position of the beat and per class, and an expert is idle only when it is below
`IDLE` (5%) at every position and in every class, so an expert that takes one position
of every beat, or the beats of one class, is in use. A router with one expert in use is
collapsed, one with idle experts and more than one in use is partly collapsed.

Every difference in standard errors is read with lesson 2's words (`verdict`): within 2
no result, 2 to 3 marginal, 3 or more a result. The spread is that of the seeds on the
six validation patients, never that of other patients.
"""

import math
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from notebooktester import param
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from vectormesh import VectorCache
from vectormesh.components import MoE, SparseMoE, load_balancing_loss

from hpo.exercise import Baseline, Data, Exercise, Outcome
from hpo.measure import Architecture, Comparison, Fit, OnEpoch, Training
from hpo.record import Role, Store, StudyLog
from hpo.search import Budget, Config, Param

EPOCHS = param(30, test=1)
REPEATS = param(3, test=2)
# train beats under notebooktester, before a train_fraction is taken; the patients
# scored on are always all six
TRAIN_ROWS = param(None, test=2000)
BATCH = 128

REPO = "pttrn-io/heartbeats-mitbih-raw"
REVISION = "b03962404ac3081ec1e3679171ea3a108d5e3601"
CLASSES = ["N", "S", "V"]
# six patients of train with N, S and V among their beats
VAL_RECORDS = (118, 124, 201, 207, 215, 223)
STEPS = 187
# an expert whose share (sparse routing) or weight (dense routing) is below this at every
# position and in every class is idle
IDLE = 0.05
# the class store `make dashboard LESSON=3` builds from the class's handed-in runs
CLASS_DB: Path = Store.default().root / "class.db"

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
    "balance": 0.0,
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
CAPACITY = ("n_params", "train_fraction", "train_examples")
# the params of a space that move each capacity axis
MOVED_BY = {
    "n_params": {"width", "depth", "n_experts"},
    "train_fraction": {"train_fraction"},
    "train_examples": {"epochs", "train_fraction"},
}


@dataclass(frozen=True)
class Split:
    """One split as tensors: beats `(n, 187, 1)`, labels `(n,)`."""

    beats: Tensor
    labels: Tensor

    def counts(self) -> np.ndarray:
        return np.bincount(self.labels.numpy(), minlength=len(CLASSES))

    def loader(self, shuffle: bool, batch_size: int = BATCH) -> DataLoader:
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


def tiles(train_fraction: float, equal_steps: bool) -> int:
    """How often an epoch passes over the beats of `train_fraction`: once, or with
    `equal_steps` about `1 / train_fraction` times, so that an epoch on a share of the
    beats has about the optimizer steps of an epoch on every beat."""
    return max(1, round(1 / train_fraction)) if equal_steps else 1


def weights(counts: np.ndarray, power: float) -> Tensor:
    """The weight of every class in the loss: `(n / (k * n_class)) ** power`."""
    balanced = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(balanced**power, dtype=torch.float32)


class Balanced(nn.Module):
    """A loss plus `balance` times the load-balancing loss of `model`, while `model` is
    training; in evaluation it is the loss alone. The model is held outside the module's
    children, so moving the loss to a device leaves the model where it is."""

    def __init__(self, loss: nn.Module, model: nn.Module, balance: float) -> None:
        super().__init__()
        self.loss = loss
        self.balance = balance
        self.held = [model]

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        loss = self.loss(scores, labels)
        model = self.held[0]
        if model.training:
            loss = loss + self.balance * load_balancing_loss(model).to(loss.device)
        return loss


def routers(model: nn.Module) -> dict[str, MoE | SparseMoE]:
    """Every mixture of experts in `model`, by its name in the model."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, MoE | SparseMoE)
    }


@dataclass(frozen=True)
class Use:
    """How one router used its experts on the positions of the validation patients.

    - `share`: per expert, the share of the positions whose router gives that expert its
      largest weight: the expert a top-1 router (`SparseMoE`) runs on them.
    - `weight`: per expert, its router weight (the softmax over the experts) averaged
      over the positions: what a dense router (`MoE`) mixes of that expert.
    - `sparse`: whether the router is top-1 (`SparseMoE`).
    - `peak_share`, `peak_weight`: per expert, the largest share or weight it gets at one
      position of the beat (over the beats) or among the beats of one class (over their
      positions). An expert that takes the same one of 23 positions in every beat has a
      share of 1/23 overall and a peak share of 1; one that takes the beats of S, 3% of
      the validation beats, has a peak share of 1 too.

    Even use is `1 / n_experts` in all of them. A dense router can give one expert the
    largest weight everywhere while its weights stay close to even, so for `MoE` the
    weight says what the router mixes in and the share only which expert leads; for
    `SparseMoE` the share is what each expert gets. `in_use` counts the experts whose
    peak, on the measure that fits the router, is at `IDLE` or above."""

    share: list[float]
    weight: list[float]
    sparse: bool
    peak_share: list[float]
    peak_weight: list[float]

    @property
    def fitting(self) -> list[float]:
        """The measure that says what the experts get: share for sparse, weight for
        dense routing."""
        return self.share if self.sparse else self.weight

    @property
    def peak(self) -> list[float]:
        """The peak of the measure that fits the router (see `fitting`)."""
        return self.peak_share if self.sparse else self.peak_weight

    @property
    def in_use(self) -> int:
        return sum(v >= IDLE for v in self.peak)

    @property
    def state(self) -> str:
        """`collapsed` (one expert in use), `partly collapsed` (idle experts, more than
        one in use) or `spread` (no idle expert)."""
        if self.in_use <= 1:
            return "collapsed"
        return "partly collapsed" if self.in_use < len(self.share) else "spread"


@torch.no_grad()
def expert_use(model: nn.Module) -> dict[str, Use]:
    """The use of the experts of every mixture of experts in `model`, by its name in the
    model, on the validation patients. Empty for a model without a router."""
    found = routers(model)
    if not found:
        return {}
    n_classes = len(CLASSES)
    # per router and measure: the sum over the beats per position (positions, experts)
    # and the sum over the positions per class (classes, experts)
    by_position: dict[tuple[str, str], Tensor] = {}
    by_class: dict[tuple[str, str], Tensor] = {}
    labels_now: list[Tensor] = []

    def counter(name: str) -> Callable[[nn.Module, tuple, Tensor], None]:
        def count(module: nn.Module, inputs: tuple, logits: Tensor) -> None:
            labels = labels_now[-1]
            # (beats, positions, experts): a dense router keeps the positions, a sparse
            # one routes them flat, beat by beat
            per_beat = logits.reshape(len(labels), -1, logits.shape[-1]).cpu()
            one_class = F.one_hot(labels, n_classes).float()
            measures = {
                "share": F.one_hot(per_beat.argmax(-1), per_beat.shape[-1]).float(),
                "weight": per_beat.softmax(-1),
            }
            for kind, values in measures.items():
                key = (name, kind)
                position = values.sum(0)
                in_class = torch.einsum("bc,bpe->ce", one_class, values)
                if key in by_position:
                    position = position + by_position[key]
                    in_class = in_class + by_class[key]
                by_position[key], by_class[key] = position, in_class

        return count

    hooks = [m.router.register_forward_hook(counter(n)) for n, m in found.items()]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    _, valid = splits(TRAIN_ROWS)
    try:
        for beats, labels in valid.loader(shuffle=False, batch_size=1024):
            labels_now.append(labels)
            model(beats.to(device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    n_beats = len(valid.labels)
    class_beats = torch.from_numpy(valid.counts()).float().clamp(min=1).unsqueeze(1)

    def measured(name: str, kind: str) -> tuple[list[float], list[float]]:
        position = by_position[name, kind] / n_beats  # (positions, experts)
        positions = position.shape[0]
        in_class = by_class[name, kind] / (
            class_beats * positions
        )  # (classes, experts)
        peak = torch.maximum(position.max(0).values, in_class.max(0).values)
        return position.mean(0).tolist(), peak.tolist()

    out = {}
    for name in found:
        share, peak_share = measured(name, "share")
        weight, peak_weight = measured(name, "weight")
        out[name] = Use(
            share=share,
            weight=weight,
            sparse=isinstance(found[name], SparseMoE),
            peak_share=peak_share,
            peak_weight=peak_weight,
        )
    return out


def use_attrs(uses: Sequence[Mapping[str, Use]]) -> dict[str, Param]:
    """The use of the experts of one or more runs as user attrs (see `Use`):

    - `expert_share` and `expert_weight`: per run the value of every expert, in the
      order of the experts, runs apart by ` | `;
    - `expert_peak`: per run the peak of every expert (`Use.peak`: the largest share for
      sparse, weight for dense routing, at one position or in one class);
    - `top_expert_share` and `top_expert_weight`: the value of the most used expert,
      averaged over the runs; `1 / n_experts` is even use, 1 a collapsed router;
    - `experts_in_use`: per run the experts whose peak is at `IDLE` or above, a number
      for one run, runs apart by ` | ` for several.

    A model with more than one router gets them per router, with `:<name>` after the
    attr. Empty without a router."""
    names = sorted({name for use in uses for name in use})
    out: dict[str, Param] = {}
    for name in names:
        suffix = "" if len(names) == 1 else f":{name}"
        runs = [use[name] for use in uses if name in use]
        for kind in ("share", "weight"):
            values = [getattr(run, kind) for run in runs]
            out[f"expert_{kind}{suffix}"] = " | ".join(
                " ".join(f"{v:.3f}" for v in run) for run in values
            )
            out[f"top_expert_{kind}{suffix}"] = round(
                float(np.mean([max(run) for run in values])), 4
            )
        out[f"expert_peak{suffix}"] = " | ".join(
            " ".join(f"{v:.3f}" for v in run.peak) for run in runs
        )
        in_use = [run.in_use for run in runs]
        out[f"experts_in_use{suffix}"] = (
            in_use[0] if len(in_use) == 1 else " | ".join(map(str, in_use))
        )
    return out


@dataclass
class Lesson3:
    """Lesson 3's harness for `hpo.exercise.Exercise`: a model of one heartbeat, trained
    on a share of the beats for a number of epochs."""

    lesson: int = 3
    epochs: int = EPOCHS
    repeats: int = REPEATS
    # the keys of a search space the harness reads; every other key is the design's
    trainer: frozenset[str] = frozenset(
        {"dataset", "epochs", "lr", "class_weight", "train_fraction", "equal_steps"}
    )
    choices: Mapping[str, Sequence[str]] = field(default_factory=lambda: CHOICES)
    store: Store = field(default_factory=Store.default)

    def data(self, config: Config) -> Data:
        if config.get("dataset", "mitbih") != "mitbih":
            raise KeyError(
                f"lesson 3 has one dataset, 'mitbih', not {config['dataset']!r}"
            )
        share = float(config.get("train_fraction", 1.0))
        train = fraction(TRAIN_ROWS, share)
        _, valid = splits(TRAIN_ROWS)
        n_train = len(train.labels)
        k = tiles(share, bool(config.get("equal_steps", False)))
        if k > 1:
            train = Split(train.beats.repeat(k, 1, 1), train.labels.repeat(k))
        epochs = int(config.get("epochs", EPOCHS))
        return Data(
            train=train.loader(shuffle=True),
            test=valid.loader(shuffle=False),
            classes=CLASSES,
            example=torch.zeros(2, STEPS, 1),
            dims={
                "steps": STEPS,
                "channels": 1,
                "n_train": n_train,
                # compute: the training examples and the optimizer steps of the run
                "train_examples": epochs * k * n_train,
                "train_steps": epochs * math.ceil(k * n_train / BATCH),
            },
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
        loss_fn: nn.Module = nn.CrossEntropyLoss(
            weight=weights(counts, float(config.get("class_weight", 0.0)))
        )
        balance = float(config.get("balance", 0.0))
        if balance:
            if not any(isinstance(m, SparseMoE) for m in model.modules()):
                warnings.warn(
                    f"balance={balance} changes nothing here: only a SparseMoE "
                    '(route on, routing "sparse", matched off) has a load-balancing '
                    "loss",
                    stacklevel=2,
                )
            loss_fn = Balanced(loss_fn, model, balance)
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
        number of beats and of training examples it trained on."""
        return {"epochs": EPOCHS, **config, **data.dims}


HARNESS = Lesson3()


def verdict(sigma: float) -> str:
    """Lesson 2's word for a difference in standard errors: `no result` within 2,
    `marginal` from 2 to 3, `result` at 3 or more; `-` where there is no sigma."""
    if not np.isfinite(sigma):
        return "-"
    size = abs(sigma)
    return "no result" if size < 2 else "marginal" if size < 3 else "result"


# z of 80% power: a true difference of (sigmas + POWER) standard errors passes `sigmas`
# standard errors four times in five
POWER = 0.84


def pairs_needed(spread: float, effect: float, sigmas: float = 2.0) -> int:
    """The pairs a group needs to see a true difference of `effect` pass `sigmas`
    standard errors four times in five (80% power), when the differences within the
    pairs have the standard deviation `spread`: the standard error is
    `spread / sqrt(pairs)`, so `pairs = ((sigmas + POWER) * spread / effect) ** 2`."""
    return math.ceil(((sigmas + POWER) * spread / effect) ** 2)


def detectable(se: float, sigmas: float = 2.0) -> float:
    """The smallest true difference that passes `sigmas` standard errors four times in
    five, at standard error `se`: `(sigmas + POWER) * se`."""
    return (sigmas + POWER) * se


def mean_draw(axis: object, default: float) -> float:
    """The mean value a trial draws from one axis of a space: the middle of an `Int` or
    `Float`, `(high - low) / ln(high / low)` when it is drawn on a log scale, the mean of
    a `Choice` (the share of True for a choice of booleans), the value of a fixed axis,
    and `default` for an axis the space leaves out."""
    if axis is None:
        return default
    if isinstance(axis, IntDistribution | FloatDistribution):
        low, high = float(axis.low), float(axis.high)
        if axis.log and high > low:
            return (high - low) / math.log(high / low)
        return (low + high) / 2
    if isinstance(axis, CategoricalDistribution):
        return float(np.mean([float(v) for v in axis.choices]))  # ty: ignore[invalid-argument-type]  (numbers and booleans)
    return float(axis)  # ty: ignore[invalid-argument-type]  (a fixed number)


def seconds_per_trial(log: StudyLog, space: Mapping[str, object]) -> float:
    """What one trial of `space` costs on this machine, from the finished trials of
    `log` (the search of Part 1): their seconds are fit as `a * epochs + b *
    train_examples`, the cost of scoring the validation patients after every epoch and
    the cost of every training example seen, and that fit is read at the mean `epochs`
    and the mean training examples of a trial of `space` (`mean_draw`). A space on
    every beat costs more per trial than the search of Part 1, whose shares are mostly
    below 1.0; a space on a small share costs less."""
    took, ran = [], []
    for t in log.trials(optuna.trial.TrialState.COMPLETE):
        if t.duration is not None and t.user_attrs.get("train_examples"):
            took.append(t.duration.total_seconds())
            epochs = float(t.user_attrs["config"]["epochs"])
            ran.append([epochs, float(t.user_attrs["train_examples"])])
    if not took:
        raise ValueError(f"the study {log.name!r} has no finished trial yet")
    cost, *_ = np.linalg.lstsq(np.array(ran), np.array(took), rcond=None)
    per_epoch, per_example = np.clip(cost, 0.0, None)
    epochs = mean_draw(space.get("epochs"), EPOCHS)
    share = mean_draw(space.get("train_fraction"), 1.0)
    equal = mean_draw(space.get("equal_steps"), 0.0)
    n_beats = len(fraction(TRAIN_ROWS, 1.0).labels)
    examples = epochs * n_beats * (equal + (1 - equal) * share)
    seconds = per_epoch * epochs + per_example * examples
    return float(seconds) if seconds > 0 else float(np.median(took))


def paired(
    run: Callable[[optuna.Trial], float],
    study: optuna.Study,
    seeded: Callable[[int | None], None],
    left: Callable[[], int] = lambda: 1,
    seeds: np.random.Generator | None = None,
) -> Callable[[optuna.Trial], float]:
    """`run` with every trial of the route stage paired with its control.

    A trial that starts fresh gets a `pair` id and a random `seed` as user attrs. If it
    drew the route stage (`route` on, `matched` searched), its twin is put in the queue:
    the same params with `matched` flipped, the same `pair` and `seed`, and Optuna runs
    it next. `seeded(seed)` is called before the trial runs and `seeded(None)` after, so
    the model builder can seed both arms alike. `left()` says how many trials are still
    to run after this one; with none left, no twin is queued. `seeds` draws the seed of
    every pair; a generator made from a fixed seed gives a search the same seeds on every
    run, and none gives fresh ones."""
    seeds = seeds or np.random.default_rng()

    def run_trial(trial: optuna.Trial) -> float:
        fresh = "pair" not in trial.user_attrs
        if fresh:
            trial.set_user_attr("pair", uuid.uuid4().hex[:8])
            trial.set_user_attr("seed", int(seeds.integers(2**31)))
        seeded(int(trial.user_attrs["seed"]))
        try:
            return run(trial)
        finally:
            seeded(None)
            config = trial.user_attrs.get("config", {})
            if fresh and config.get("route") and "matched" in trial.params and left():
                study.enqueue_trial(
                    {**trial.params, "matched": not trial.params["matched"]},
                    user_attrs={
                        "pair": trial.user_attrs["pair"],
                        "seed": trial.user_attrs["seed"],
                    },
                )

    return run_trial


class RoutedExercise(Exercise):
    """`hpo.exercise.Exercise` with paired trials and the use of the experts.

    - Every trial of the route stage is followed by its twin with `matched` flipped, on
      the same draw and seed (`paired`): the model of both is built from the seed and
      trains from it, so the two see the beats in the same order.
    - Once a trial's model has trained, `use_attrs` of it become user attrs of the
      trial, so the dashboard shows the use of the experts next to its objective. A
      trial without a router gets none of them."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        make = self.make_model
        self.built: nn.Module | None = None
        self.seed: int | None = None
        self.left = 0

        def keep(config: Config, data: Data) -> nn.Module:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            self.built = make(config, data)
            if self.seed is not None:
                torch.manual_seed(
                    self.seed
                )  # the same order of the beats for both arms
            return self.built

        self.make_model = keep

    def budget(self) -> Budget:
        """The budget at what a trial of this space costs (`seconds_per_trial`), not at
        the median trial of Part 1's search."""
        super().budget()  # says to run Part 1 first when its search is not there
        like = StudyLog.load(self.like, self.store)
        budget = Budget.at(
            self.questions.minutes,
            seconds_per_trial(like, self.space),
            like.name,
            space=self.space,
            epochs=self.harness.epochs,
            repeats=self.harness.repeats,
        )
        self.left = param(budget.n_trials, test=2)  # the trials `run` will ask for
        return budget

    def trial(self, search: StudyLog) -> Callable[[optuna.Trial], float]:
        run = super().trial(search)

        def seeded(seed: int | None) -> None:
            self.seed = seed

        def with_use(trial: optuna.Trial) -> float:
            self.built = None
            self.left -= 1
            try:
                return run(trial)
            finally:
                if self.built is not None:
                    for key, value in use_attrs([expert_use(self.built)]).items():
                        trial.set_user_attr(key, value)

        return paired(with_use, search.study, seeded, lambda: self.left)


def arm(config: Mapping[str, object]) -> str:
    """Which arm of the capacity study a config is: `route` (the route stage), `matched`
    (its control of the same parameter count) or `plain` (neither)."""
    if not config.get("route"):
        return "plain"
    return "matched" if config.get("matched") else "route"


def trial_rows(log: StudyLog | optuna.Study) -> pd.DataFrame:
    """The finished trials of a search, one row each: the objective, the arm, its `pair`,
    `n_params`, `n_train`, `train_examples`, the use of its experts (NaN without a
    router) and every param the trial drew or was given."""
    done = optuna.trial.TrialState.COMPLETE
    found = (
        log.trials(done)
        if isinstance(log, StudyLog)
        else log.get_trials(deepcopy=False, states=(done,))
    )
    rows = []
    for t in found:
        a = t.user_attrs
        config = a.get("config", {})
        rows.append(
            {
                "value": t.value,
                "arm": arm(config),
                "pair": a.get("pair"),
                "n_params": a.get("n_params"),
                "n_train": a.get("n_train"),
                "train_examples": a.get("train_examples"),
                "top_expert_share": a.get("top_expert_share", np.nan),
                "top_expert_weight": a.get("top_expert_weight", np.nan),
                "experts_in_use": a.get("experts_in_use", np.nan),
                **config,
            }
        )
    return pd.DataFrame(rows)


def pairs(table: pd.DataFrame) -> pd.DataFrame:
    """One row per pair whose two arms both finished: the params of the draw (from the
    route arm), the objective of `route` and of `matched`, their difference, and the use
    of the route arm's experts."""
    rows = []
    for _, group in table.dropna(subset=["pair"]).groupby("pair"):
        by_arm = {row["arm"]: row for _, row in group.iterrows()}
        if set(by_arm) != {"route", "matched"}:
            continue
        route, matched = by_arm["route"], by_arm["matched"]
        rows.append(
            {
                **route.drop(["value", "arm"]).to_dict(),
                "route": route["value"],
                "matched": matched["value"],
                "route - matched": route["value"] - matched["value"],
            }
        )
    return pd.DataFrame(rows)


def number(value: float) -> str:
    """A value of an axis as a label: whole numbers with thousands separators, the
    others with three significant digits."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:.3g}"


def groups(table: pd.DataFrame, axis: str, bins: int = 3) -> pd.Series:
    """The group of every pair along `axis`, as `paired_curve` reads them: `bins` groups
    of about as many pairs each (one group per value when the axis has that few values),
    each labelled by the smallest and largest value in it, in the order of the axis."""
    values = table[axis].astype(float)
    if values.nunique() <= bins:
        index = values.rank(method="dense").astype(int)
    else:
        index = pd.qcut(values.rank(method="first"), bins, labels=False)
    labels = {}
    for i in sorted(index.unique()):
        members = table.loc[index == i, axis]
        lo, hi = members.min(), members.max()
        labels[i] = number(lo) if lo == hi else f"{number(lo)} to {number(hi)}"
    order = [labels[i] for i in sorted(labels)]
    return pd.Series(
        pd.Categorical(index.map(labels), categories=order, ordered=True),
        index=table.index,
        name=axis,
    )


def paired_curve(table: pd.DataFrame, axis: str, bins: int = 3) -> pd.DataFrame:
    """The pairs (`pairs`) along `axis`: put in `bins` groups of about as many pairs each
    (one group per value when the axis has that few values), each group labelled by the
    smallest and largest value in it. Per group:

    - `pairs`, and the mean objective of each arm;
    - `route - matched`: the mean difference within the pairs;
    - `se` and `sigma`: the paired standard error, the standard deviation (ddof=1) of the
      differences within the pairs over the square root of the number of pairs, and the
      mean difference in those standard errors, with its `verdict`;
    - `se unpaired` and `sigma unpaired`: lesson 2's standard error of two independent
      arms, `sqrt(std_route² / n + std_matched² / n)`, on the same trials. Its spread
      holds every difference between the draws of a group (width, depth, share,
      epochs), which the pairs cancel, so it is larger when the draws differ.

    No sigma for a group of fewer than two pairs, nor for one whose values do not vary."""
    group = groups(table, axis, bins)
    rows = {}
    for label, members in table.groupby(group, observed=True):
        diff = members["route - matched"]
        n = len(diff)
        nan = float("nan")
        se = diff.std(ddof=1) / math.sqrt(n) if n > 1 else nan
        unpaired = (
            math.sqrt(
                (members["route"].var(ddof=1) + members["matched"].var(ddof=1)) / n
            )
            if n > 1
            else nan
        )
        sigma = diff.mean() / se if n > 1 and se > 0 else nan
        rows[label] = {
            "pairs": n,
            "route": members["route"].mean(),
            "matched": members["matched"].mean(),
            "route - matched": diff.mean(),
            "se": se,
            "sigma": sigma,
            "reading": verdict(sigma),
            "se unpaired": unpaired,
            "sigma unpaired": diff.mean() / unpaired if unpaired > 0 else nan,
        }
    frame = pd.DataFrame(rows).T
    numbers = frame.columns.drop("reading")
    frame[numbers] = frame[numbers].astype(float)
    frame["pairs"] = frame["pairs"].astype(int)
    frame.index.name = axis
    return frame


def class_pairs(*studies: str, db: Path = CLASS_DB) -> pd.DataFrame:
    """The pairs (`pairs`) of studies in the class store that `make dashboard LESSON=3`
    builds from the class's handed-in runs. By default the handed-out search,
    `lesson3-notebook`, which the class store pools: one study with the trials of every
    student. Studies of students go by their name in that store, `<study>@<student>`;
    read them together only when they searched the same space. Empty when the store or
    a study is not there."""
    names = studies or ("lesson3-notebook",)
    if not db.exists():
        return pd.DataFrame()
    url = f"sqlite:///{db}"
    there = set(optuna.get_all_study_names(url))
    tables = [
        trial_rows(optuna.load_study(study_name=name, storage=url))
        for name in names
        if name in there
    ]
    tables = [t for t in tables if not t.empty]
    return pairs(pd.concat(tables, ignore_index=True)) if tables else pd.DataFrame()


def use_table(table: pd.DataFrame) -> pd.DataFrame:
    """The trials of the route arm, the most concentrated router first: the objective,
    the experts in use, the share and the weight of the most used expert (see `Use`),
    and the params that set the router's work (`n_experts`, `routing`, `balance`) and
    its capacity."""
    routed = table[table["arm"] == "route"]
    columns = [
        c
        for c in ("value", "experts_in_use", "top_expert_share", "top_expert_weight",
                  "n_experts", "routing", "balance", "width", "depth",
                  "train_fraction", "epochs")
        if c in routed.columns
    ]  # fmt: skip
    return routed[columns].sort_values("top_expert_weight", ascending=False)


def control(exercise: Exercise, outcome: Outcome) -> Comparison:
    """The winner's route stage against its capacity-matched control: the winner's config
    with `route` on, once with the router and once with `matched`, each run as often as
    the baseline and logged to the study's `-runs` (the route arm as `hypothesis`, the
    control as `ablation`), the route arm with the use of its experts. An arm that is the
    winner reuses the winner's runs."""
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
        trained: list[nn.Module] = []

        def make(
            config: Config = config,
            data: Data = data,
            trained: list[nn.Module] = trained,
        ) -> nn.Module:
            trained.append(exercise.make_model(config, data))
            return trained[-1]

        results = exercise.repeats(make, data, config)
        use = use_attrs([expert_use(m) for m in trained])
        runs.log(
            name,
            role,
            results,
            params=HARNESS.params(config, data),
            dataset=str(config.get("dataset")),
            architecture=Architecture.of(model, data.example),
            design=exercise.script,
            attrs=use,
        )
        if use:
            print(f"{name}: share of the positions per expert  {use['expert_share']}")
            print(f"{name}: router weight per expert          {use['expert_weight']}")
            print(f"{name}: experts in use per run            {use['experts_in_use']}")
        measured[name] = (results.values(exercise.objective), n_params)
    (route_name, (route, route_n)), (ctrl_name, (ctrl, ctrl_n)) = measured.items()
    against = Comparison.of(f"{route_name} ({route_n:,} params)", route, versus=ctrl)
    print()
    print(Comparison.of(f"{ctrl_name} ({ctrl_n:,} params)", ctrl))
    print(against)
    print(
        "(the winner was chosen on these six patients, so this comparison leans towards "
        "the winner's arm; the spread is that of the seeds, on the same patients)"
    )
    return against


def read(exercise: Exercise, outcome: Outcome) -> None:
    """Print the paired curves of the search along every capacity axis it moved, the use
    of the experts in the trials of the route arm, and then the winner's route stage
    against its control."""
    table = trial_rows(StudyLog.load(exercise.name, exercise.store))
    searched = set(exercise.space.searched())
    both = pairs(table) if not table.empty else pd.DataFrame()
    print(f"\n{len(both)} pairs of route and matched on the same draw and seed")
    for axis in CAPACITY:
        if both.empty or not MOVED_BY[axis] & searched:
            continue
        print(
            f"\nroute against matched within the pairs, over {axis} ({exercise.objective}):"
        )
        print(paired_curve(both, axis).round(3).to_string())
    if len(both) > 1:
        spread = float(both["route - matched"].std(ddof=1))
        print(
            f"\nthe differences within the pairs spread by {spread:.4f} (std); to see a "
            "true difference pass two standard errors four times in five, a group "
            f"needs {pairs_needed(spread, 0.01)} pairs for 0.01 and "
            f"{pairs_needed(spread, 0.02)} for 0.02. Pairs of classmates who searched "
            "the same space can be read together (harness.class_pairs)"
        )
    if not table.empty and (table["arm"] == "route").any():
        print(
            "\nthe use of the experts in the route arm: experts_in_use counts the experts "
            f"at {IDLE:.0%} or more at some position or in some class (share for "
            "sparse, weight for dense routing); top_expert_share and top_expert_weight "
            "are those of the most used expert over all positions (1 / n_experts is "
            "even use, 1 a collapsed router). Every expert per trial: expert_share, "
            "expert_weight and expert_peak in the dashboard"
        )
        print(use_table(table).round(3).to_string(index=False))
    control(exercise, outcome)
