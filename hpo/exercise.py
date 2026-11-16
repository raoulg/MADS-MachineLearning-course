"""The study of an exercise, the same in every lesson: a baseline, a search over a design
of the student's own, the winner rerun, and the difference in sigmas.

A lesson says what is its own in a `Harness`: how a config becomes data (`data`), how a
model is trained on it (`fit`), which linear model is its baseline (`baseline`), and
which params a run logs (`params`). `run_study` does the rest, and every lesson's
`exercises/study.py` is then its questions to the student (STUDY, IDEA, EXPECTS, SPACE,
MINUTES) and one call:

    run_study(HARNESS, study=STUDY, idea=IDEA, expects=EXPECTS, space=SPACE,
              minutes=MINUTES, design=Head, config=Config, make_model=make_model,
              design_file=Path(designs.__file__), objective="macro_recall")

`train` is the trainer every harness can use: vectormesh's `Trainer`, timed, scored with
every metric of `hpo.metrics` on the whole test set after every epoch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
import torch
from notebooktester import param
from torch import nn
from torch.utils.data import DataLoader
from vectormesh.training import Trainer, TrainerSettings

from hpo.designs import check_space, choices, options
from hpo.metrics import SCALARS, Scorer, value
from hpo.optunalog import (
    TESTING,
    Choice,
    budget,
    draw,
    log_run,
    record,
    search_study,
    space_line,
    use_study,
)

EXPECTS_PLACEHOLDER = "say what you expect, and why"


@dataclass
class Fit:
    """What training a model gave: every metric after the last epoch (as
    `hpo.metrics.scores` gives them), the seconds spent in training steps, and per
    epoch the losses and the scalar metrics."""

    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    curves: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "test_loss": []}
    )


# Called after every epoch with the epoch's number and the fit so far; it may raise
# optuna.TrialPruned to stop the training.
OnEpoch = Callable[[int, Fit], None]


@dataclass
class Data:
    """What a config trains and is scored on: the loaders, the names of the classes, one
    batch input of two rows (for the layer table and the shapes), and the sizes a design
    is built with (`dims`, for example `{"dim": 384}`)."""

    train: DataLoader
    test: DataLoader
    classes: list[str]
    example: Any
    dims: dict[str, int] = field(default_factory=dict)


@dataclass
class Baseline:
    """The baseline of a dataset: the config it is trained with (dataset, lr, epochs, ...)
    and the model it builds on the data of that config."""

    config: dict[str, Any]
    make: Callable[[Data], nn.Module]


class Harness(Protocol):
    """What a lesson's `exercises/harness.py` gives `run_study`."""

    lesson: int
    epochs: int
    repeats: int
    trainer: Collection[str]  # the keys of a space the harness reads, not the design

    def data(self, config: Mapping[str, Any]) -> Data: ...

    def fit(
        self,
        model: nn.Module,
        data: Data,
        config: Mapping[str, Any],
        on_epoch: OnEpoch | None = None,
    ) -> Fit: ...

    def baseline(self, dataset: str) -> Baseline: ...

    def params(self, config: Mapping[str, Any], data: Data) -> dict[str, Any]: ...


class TimedTrainer(Trainer):
    """A Trainer that adds up the seconds spent in training steps. Evaluation after every
    epoch is not counted: `train_seconds` is the cost of fitting, not of scoring."""

    train_seconds = 0.0

    def trainbatches(self) -> float:
        start = time.perf_counter()
        loss = super().trainbatches()
        self.train_seconds += time.perf_counter() - start
        return loss


def train(
    model: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    classes: Sequence[str],
    *,
    lr: float,
    epochs: int,
    loss_fn: nn.Module | None = None,
    optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
    on_epoch: OnEpoch | None = None,
) -> Fit:
    """Train `model` for `epochs` and score it with every metric on `test_dl` after every
    epoch. `on_epoch` sees the fit after each epoch."""
    scorer = Scorer(test_dl, classes)
    fit = Fit(curves={"train_loss": [], "test_loss": [], **{m: [] for m in SCALARS}})

    def report(
        epoch: int,
        train_loss: float,
        test_loss: float,
        metric_dict: dict[str, float],  # empty: the trainer computes no metric
    ) -> None:
        fit.metrics = scorer(model)
        fit.seconds = trainer.train_seconds
        fit.curves["train_loss"].append(float(train_loss))
        fit.curves["test_loss"].append(float(test_loss))
        for m in SCALARS:
            fit.curves[m].append(fit.metrics[m])
        if on_epoch is not None:
            on_epoch(epoch, fit)

    settings = TrainerSettings(
        epochs=epochs,
        metrics=[],  # every metric comes from the scorer, over the whole test set
        logdir=Path("modellogs").resolve(),  # never created unless earlystop saves
        train_steps=len(train_dl),
        valid_steps=len(test_dl),
        optimizer_kwargs={"lr": lr},
        earlystop_kwargs=None,
    )
    trainer = TimedTrainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn or nn.CrossEntropyLoss(),
        optimizer=optimizer,
        traindataloader=train_dl,
        validdataloader=test_dl,
        progress=False,
        reporters=[report],
    )
    trainer.loop()
    fit.seconds = trainer.train_seconds
    return fit


def values(runs: Sequence[Mapping[str, Any]], metric: str = "accuracy") -> np.ndarray:
    """One metric of every repeat, as an array: `values(runs, "recall/F")`."""
    return np.array([value(r, metric) for r in runs])


def spread(
    make: Callable[[], nn.Module],
    fit: Callable[[nn.Module], Fit],
    repeats: int,
) -> tuple[list[dict[str, Any]], float]:
    """Every metric of `repeats` runs, each from other random weights (nothing is
    seeded), and the mean seconds spent in training steps. `fit(model)` trains one."""
    runs, seconds = [], []
    for _ in range(repeats):
        done = fit(make())
        runs.append(done.metrics)
        seconds.append(done.seconds)
    return runs, float(np.mean(seconds))


def sigmas(scores: np.ndarray, versus: np.ndarray) -> float:
    """The difference of the means in units of its standard error."""
    pooled = np.sqrt(
        scores.var(ddof=1) / len(scores) + versus.var(ddof=1) / len(versus)
    )
    return float((scores.mean() - versus.mean()) / pooled) if pooled > 0 else 0.0


def compare(name: str, scores: np.ndarray, versus: np.ndarray | None = None) -> float:
    """Print the mean and spread of `scores`; against `versus`, also the difference in
    points and in sigmas, which it returns (0.0 without `versus`)."""
    line = f"{name:<32} {scores.mean():.4f} +/- {scores.std():.4f}   (n={len(scores)})"
    found = 0.0
    if versus is not None:
        found = sigmas(scores, versus)
        line += (
            f"   {100 * (scores.mean() - versus.mean()):+.2f} pt, {found:+.1f} sigma"
        )
    print(line)
    return found


def check_expects(expects: str) -> str:
    """`expects` stripped; the placeholder raises a `ValueError` outside notebooktester."""
    said = expects.strip() if isinstance(expects, str) else ""
    if not said or (said == EXPECTS_PLACEHOLDER and not TESTING):
        raise ValueError(
            "EXPECTS: say in one line what you expect to happen and why, before you "
            f"run, instead of {EXPECTS_PLACEHOLDER!r}"
        )
    return said


def objective(
    harness: Harness,
    space: Mapping[str, Any],
    make_model: Callable[[Mapping[str, Any], Data], nn.Module],
    metric: str,
    design_file: Path | None = None,
) -> Callable[[optuna.Trial], float]:
    """What an Optuna study calls once per trial: draw a config from `space`, build it
    with `make_model`, train it with the harness, report `metric` after every epoch, and
    return it after the last. Every trial logs every metric, and carries `design_file`,
    the design script, as its artifact."""

    def run_trial(trial: optuna.Trial) -> float:
        config = draw(trial, space)
        data = harness.data(config)
        model = make_model(config, data)
        seen = Fit()

        # every epoch reaches the store, and the study's pruner can stop a trial that is
        # behind the others
        def to_optuna(epoch: int, fit: Fit) -> None:
            nonlocal seen
            seen = fit
            trial.report(value(fit.metrics, metric), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            seen = harness.fit(model, data, config, to_optuna)
            return value(seen.metrics, metric)
        finally:
            # a pruned trial is recorded as well, with the epochs it ran
            record(
                trial,
                "trial",
                dataset=config.get("dataset"),
                model=model,
                example=data.example,
                train_seconds=seen.seconds,
                curves=seen.curves,
                metrics=seen.metrics or None,
                attrs=dict(data.dims),
                design=design_file,
            )

    return run_trial


def run_study(
    harness: Harness,
    *,
    study: str,
    idea: Mapping[str, str],
    expects: str,
    space: Mapping[str, Any],
    minutes: float,
    design: type,
    config: type,
    make_model: Callable[[Mapping[str, Any], Data], nn.Module],
    design_file: Path,
    objective_metric: str = "accuracy",
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    like: str | None = None,
) -> dict[str, Any]:
    """The study `lesson<N>-study-<study>` over the design `design` (its options) with the
    config dataclass `config` (its choices): check the space, turn `minutes` into trials,
    measure the baseline of every dataset the space can draw, search, rerun the winner,
    and log baseline and winner in `lesson<N>-study-<study>-runs`, every run with
    `design_file`. Returns what it printed, as data.

    The cost of a trial is measured on the search study `like`, `lesson<N>-notebook` by
    default: the handed-out search of the lesson's exercise notebook."""
    expects = check_expects(expects)
    check_space(space, options(design), choices(config), harness.trainer)
    lesson = harness.lesson
    like = like or f"lesson{lesson}-notebook"
    # Under notebooktester the placeholder names the notebook's own study as well, which
    # holds another space; the tester's run of a script keeps a study of its own.
    name = f"lesson{lesson}-study-{study}{'-script' if TESTING else ''}"

    try:
        spend = budget(
            minutes,
            like=like,
            epochs=harness.epochs,
            repeats=harness.repeats,
            space=space,
        )
    except ValueError as error:
        raise ValueError(
            f"budget: run Part 1 of lesson {lesson}'s exercise notebook on this machine "
            f"first; it measures what one trial costs here ({error})"
        ) from None
    n_trials = param(spend["n_trials"], test=2)
    print(
        f"budget   : {spend['n_trials']} trials in {minutes} minutes at "
        f"{spend['seconds_per_trial']}s each, {spend['trials_per_axis']} per searched axis"
    )
    if n_trials == 0:
        raise ValueError(
            f"budget: {minutes} minutes hold no trial on this machine once the baseline "
            "and the winner are set aside; give the study more minutes or fewer epochs"
        )

    def fit_with(data: Data, cfg: Mapping[str, Any]) -> Callable[[nn.Module], Fit]:
        return lambda model: harness.fit(model, data, cfg)

    # the baseline of every dataset the space can draw, before the search
    drawn = space.get("dataset")
    datasets = drawn.choices if isinstance(drawn, Choice) else [drawn]
    baselines = {}
    for dataset in datasets:
        base = harness.baseline(str(dataset))
        data = harness.data(base.config)
        runs, seconds = spread(
            lambda b=base, d=data: b.make(d),
            fit_with(data, base.config),
            harness.repeats,
        )
        baselines[str(dataset)] = (base, data, runs, seconds)
        print(f"baseline : measured on {dataset}")

    searched = search_study(
        name,
        space,
        sampler=sampler,
        pruner=pruner,
        attrs={
            "origin": "study",
            "study": study,
            "budget": {**spend, "n_trials": n_trials},
            # a study over a design of your own never searches the handed-out space
            "space_is_default": False,
            "expects": expects,
        },
        idea=idea,
        objective=objective_metric,
    )
    start = time.time()
    searched.optimize(
        objective(harness, space, make_model, objective_metric, design_file),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    done = [t for t in searched.trials if t.state == optuna.trial.TrialState.COMPLETE]
    table = pd.DataFrame([{objective_metric: t.value, **t.params} for t in done])
    print(
        f"{n_trials} trials in {time.time() - start:.0f}s; {len(done)} finished in all"
    )
    print(
        table.sort_values(objective_metric, ascending=False)
        .head(6)
        .to_string(index=False)
    )

    best = searched.best_trial.user_attrs["config"]
    runs_study = use_study(f"{name}-runs", idea=idea, objective=objective_metric)

    base, base_data, base_runs, base_seconds = baselines[str(best.get("dataset"))]
    base_model = base.make(base_data)
    log_run(
        "baseline",
        "baseline",
        dataset=base.config.get("dataset"),
        metrics=base_runs,
        params=harness.params(base.config, base_data),
        train_seconds=base_seconds,
        study=runs_study,
        model=base_model,
        example=base_data.example,
        design=design_file,
    )

    data = harness.data(best)
    win_runs, win_seconds = spread(
        lambda: make_model(best, data), fit_with(data, best), harness.repeats
    )
    log_run(
        "winner",
        "final",
        dataset=best.get("dataset"),
        metrics=win_runs,
        params=harness.params(best, data),
        train_seconds=win_seconds,
        study=runs_study,
        model=make_model(best, data),
        example=data.example,
        attrs={"search_space": space_line(space)},
        design=design_file,
    )

    baseline_scores = values(base_runs, objective_metric)
    winner_scores = values(win_runs, objective_metric)
    print()
    print("winner   :", {k: v for k, v in best.items() if k != "epochs"})
    print(f"objective: {objective_metric}")
    compare(f"baseline on {best.get('dataset')}", baseline_scores)
    found = compare("your search winner", winner_scores, versus=baseline_scores)
    print(f"\nthe winner against the baseline: {found:+.1f} sigma")
    print(f"idea     : {idea['idea_name']}  ({idea['idea']})")
    print(f"studies  : {name}, {name}-runs  (make optuna-dashboard)")
    return {
        "study": name,
        "objective": objective_metric,
        "baseline": baseline_scores.tolist(),
        "winner": winner_scores.tolist(),
        "sigmas": found,
        "best": best,
    }
