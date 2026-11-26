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
- `drop_domain`: one domain left out of train and validation, or `"none"`; a run
  without a domain is scored on the other six, so its accuracy is read against another
  run's on those six domains, from the recall per class every run logs;
- `form`: the two vectors as `(batch, 2, dim)` (`"flat"`), or each folded into a grid
  (`"grid"`), `(batch, 2, 16, 24)` for small and `(batch, 2, 24, 32)` for base; the
  design's kernel reads the grid.

An epoch is a fixed number of optimizer steps, one pass over every train row, whatever
the share: on a share of 10% an epoch passes over its rows ten times. So the epochs of
two runs are the same steps. A run trains until its validation accuracy has not
improved for `PATIENCE` epochs, or for `EPOCHS` epochs at most, and keeps the weights of
its best epoch.

The notebook's helpers are at the end: `run` (one run, logged and stored), `table`,
`front` and `plot_front` (the Pareto front of accuracy against FLOPs per image),
`rows_of` (the runs of a study in the store, to draw its front), `search` (the
search of Part 1) and `from_study` (a setting of `study.py`).
"""

import ast
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from huggingface_hub.utils import disable_progress_bars as hub_quiet
from loguru import logger
from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from notebooktester import param
from pandas.io.formats.style import Styler
from torch import Tensor, nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset
from vectormesh import VectorCache

from hpo.exercise import Baseline, Data
from hpo.measure import Architecture, Confusion, Fit, OnEpoch, Training
from hpo.record import Idea, RunRecord, Store, StudyLog
from hpo.record.runs import Role
from hpo.search import Choice, Config, Float, Int, Param, Space

# the notebook and study.py print what a run found, not the progress of every download
logger.remove()
logger.add(sys.stderr, level="WARNING")
datasets.disable_progress_bars()
hub_quiet()
optuna.logging.set_verbosity(optuna.logging.WARNING)

EPOCHS = param(150, test=1)  # the most epochs a run trains
# the epochs in a row without a better validation accuracy after which a run stops
PATIENCE = param(10, test=1)
# one seed per run, the course's default; the notebook says beside E2 what another
# seed moves
REPEATS = 1
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


# The notebook's helpers. Part 1 is the same for everyone, so its idea is the course's:
# "lesson 3 as handed out".
SEED = 0
OBJECTIVE = "accuracy"
NOTEBOOK = "lesson3-notebook"
# Okabe-Ito, one hue per group, told apart without colour vision
HUES = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
Make = Callable[[Config, Data], nn.Module]


@cache
def notebook_runs() -> StudyLog:
    """The study `lesson3-notebook-runs`: the runs of Part 1 that are not trials."""
    return StudyLog.runs(
        name=f"{NOTEBOOK}-runs",
        idea=Idea.handed_out(lesson=3),
        objective=OBJECTIVE,
        choices=CHOICES,
    )


def results() -> Path:
    """The file every run of Part 1 is stored in, by name."""
    name = "part1-test.json" if HARNESS.store.testing else "part1.json"
    return HARNESS.store.root.parent / "lesson3" / name


def stored() -> dict:
    path = results()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def store(rows: Sequence[dict]) -> None:
    kept = stored() | {row["name"]: row for row in rows}
    results().parent.mkdir(parents=True, exist_ok=True)
    results().write_text(json.dumps(kept), encoding="utf-8")


def run(name: str, config: Config, role: Role, experiment: str, make: Make) -> dict:
    """One run of `config` from `SEED`: trained until it settles, its accuracy, FLOPs and
    latency measured, logged to `lesson3-notebook-runs` and stored under `name`."""
    data = HARNESS.data(config)
    torch.manual_seed(SEED)
    model = make(config, data)
    start = time.time()
    fit = HARNESS.fit(model, data, config)
    seconds = time.time() - start
    architecture = Architecture.of(model, data.example)
    notebook_runs().log(
        name=name,
        role=role,
        results=fit.as_repeats(),
        dataset="mix7",
        params=HARNESS.params(config, data),
        architecture=architecture,
    )
    scores = fit.scores()
    row = {
        "name": name,
        "experiment": experiment,
        "accuracy": scores.accuracy,
        "flops": architecture.flops_per_image,
        "latency_us": architecture.latency_us,
        "n_params": architecture.n_params,
        "kept_epoch": kept_epoch(fit),
        "epochs_run": len(fit.curves.accuracy),
        "minutes": seconds / 60,
        "curve": fit.curves.accuracy,
        "share": config["share"],
        "width": config.get("width", 0),
    }
    store([row])
    return row


def table(rows: Sequence[dict]) -> Styler:
    """The runs side by side: accuracy, what one image costs, and how the run settled."""
    columns = [
        "accuracy",
        "flops",
        "latency_us",
        "n_params",
        "kept_epoch",
        "epochs_run",
        "minutes",
    ]
    return (
        pd.DataFrame(rows)
        .set_index("name")[columns]
        .style.format(
            {
                "accuracy": "{:.3f}",
                "flops": "{:,.0f}",
                "latency_us": "{:.2f}",
                "n_params": "{:,.0f}",
                "minutes": "{:.1f}",
            }
        )
    )


def metric_of(rows: Sequence[dict]) -> str:
    """The metric the rows are scored on: their study's objective, or the notebook's."""
    return rows[0].get("objective", OBJECTIVE) if rows else OBJECTIVE


def front(rows: Sequence[dict]) -> list[dict]:
    """The runs on the Pareto front, from cheap to dear. A run is on the front when no
    other run is at least as accurate at no more FLOPs per image and better on one of
    the two."""
    metric = metric_of(rows)

    def beats(other: dict, row: dict) -> bool:
        as_good = other[metric] >= row[metric] and other["flops"] <= row["flops"]
        return as_good and (
            other[metric] > row[metric] or other["flops"] < row["flops"]
        )

    on = [row for row in rows if not any(beats(other, row) for other in rows)]
    return sorted(on, key=lambda r: r["flops"])


def plain_ticks(axis: Axis) -> None:
    """A log axis named at 1, 2, 3 and 5 times each power of ten, in plain numbers:
    20k, 0.5."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    axis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    )
    axis.set_minor_formatter(NullFormatter())


def plot_front(rows: Sequence[dict], title: str) -> list[dict]:
    """The score against FLOPs per image, one colour per `experiment` of the rows, the
    front as steps with its runs named. Returns the runs on the front."""
    on = front(rows)
    metric = metric_of(rows)
    _, ax = plt.subplots(figsize=(9, 4.8))
    groups = list(dict.fromkeys(r["experiment"] for r in rows))
    for group, hue in zip(groups, HUES, strict=False):
        mine = [r for r in rows if r["experiment"] == group]
        ax.scatter(
            [r["flops"] for r in mine],
            [r[metric] for r in mine],
            color=hue,
            s=40,
            label=group,
            zorder=3,
        )
    ax.step(
        [r["flops"] for r in on],
        [r[metric] for r in on],
        where="post",
        color="gray",
        lw=1,
        zorder=2,
        label="Pareto front",
    )
    # a ring on every run of the front, its name above and to its left: no run lies
    # there, or it would be more accurate and cheaper
    ax.scatter(
        [r["flops"] for r in on],
        [r[metric] for r in on],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    for r in on:
        ax.annotate(
            r["name"],
            (r["flops"], r[metric]),
            xytext=(-7, 5),
            ha="right",
            va="bottom",
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    plain_ticks(ax.xaxis)
    ax.set_xlabel("FLOPs per image (log scale); better is up and to the left")
    ax.set_ylabel(f"validation {metric.replace('_', ' ')}")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()
    return on


def rows_of(study: str, by: str | None = None) -> list[dict]:
    """The finished runs of a study in the store as rows for `plot_front`, grouped by
    their role (baseline, trial, final), or by the value of the axis `by`. Each row
    holds its score under the study's objective."""
    log = StudyLog.load(study, HARNESS.store)
    rows = []
    for t in log.trials(optuna.trial.TrialState.COMPLETE):
        config = t.user_attrs.get("config") or t.params
        role = t.user_attrs.get("role", "trial")
        rows.append(
            {
                "name": t.user_attrs.get("name", f"{role} {t.number}"),
                "experiment": role if by is None else f"{by}={config.get(by)}",
                "objective": log.objective,
                log.objective: t.value,
                "flops": t.user_attrs["flops_per_image"],
                "latency_us": t.user_attrs["latency_us"],
                **{k: v for k, v in config.items() if k in t.params},
            }
        )
    return rows


def search(space: Space, n_trials: int, make: Make, experiment: str) -> list[dict]:
    """The search of Part 1 in the study `lesson3-notebook`: `n_trials` trials drawn at
    random from `space`, each trained until it settles and logged with its FLOPs and
    latency. Returns them as rows."""
    log = StudyLog.search(
        name=NOTEBOOK,
        space=space,
        idea=Idea.handed_out(lesson=3),
        objective=OBJECTIVE,
        attrs={"origin": "notebook"},
    )

    mine: list[int] = []  # the numbers of this search's trials

    def run_trial(trial: optuna.Trial) -> float:
        mine.append(trial.number)
        config = space.draw(trial)
        data = HARNESS.data(config)
        model = make(config, data)
        seen = Fit()

        def to_optuna(epoch: int, fit: Fit) -> None:
            trial.report(fit.scores().get(OBJECTIVE), epoch)

        try:
            seen = HARNESS.fit(model, data, config, to_optuna)
            return seen.scores().get(OBJECTIVE)
        finally:
            log.record(
                trial,
                RunRecord(
                    role="trial",
                    dataset="mix7",
                    config=config,
                    fixed=space.fixed(),
                    train_seconds=seen.seconds,
                    curves=seen.curves,
                    scores=seen.confusion.scores() if seen.confusion else None,
                    confusion=seen.confusion,
                    extra=dict(data.dims),
                ),
                Architecture.of(model, data.example),
            )

    # no progress bar: a saved notebook keeps it at its first frame
    log.study.optimize(run_trial, n_trials=n_trials)
    done = [t for t in log.trials(optuna.trial.TrialState.COMPLETE) if t.number in mine]
    rows = [
        {
            "name": f"trial {t.number}",
            "experiment": experiment,
            "accuracy": t.value,
            "flops": t.user_attrs["flops_per_image"],
            "latency_us": t.user_attrs["latency_us"],
            **t.params,
        }
        for t in done
    ]
    store(rows)
    return rows


def from_study(name: str) -> Param | Space:
    """The value `study.py` gives `name` (`"SPACE"`, `"MINUTES"`), as it stands. Only that
    assignment is read: the rest of `study.py` asks for your own words first."""
    source = (Path(__file__).resolve().parent / "study.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name:
            names = {"Space": Space, "Choice": Choice, "Float": Float, "Int": Int}
            return eval(ast.unparse(node.value), {**names, "EPOCHS": EPOCHS})
    raise ValueError(f"study.py gives no value to {name}")
