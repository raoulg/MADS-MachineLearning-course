"""Keep the runs of the exercise in Optuna, and read them in optuna-dashboard.

Optuna stores studies; a study holds trials; a trial holds the params it drew, one value
per epoch, its final value and free-form user attrs. This module writes the exercise's
runs in that shape, to one sqlite file, `artefacts/optuna/studies.db` at the root of the
course repository:

- a search is a study (`search_study`): every trial draws its params from the search
  space (`draw`) and reports its accuracy per epoch, so a pruner can stop it early;
- a run that is not a draw from a space (a baseline, an ablation, a winner rerun) is a
  trial in a second study (`use_study` + `log_run`), with the params it was given and
  one value per repeat.

Every trial carries, as user attrs, what the class run protocol asks of a run: `role`,
`dataset`, `session`, `n_params`, the one-line `architecture`, `train_seconds` and the
loss curves (`record`). The layer table goes along twice: as the trial's note, which the
dashboard renders, and as the artifact `architecture.json`.

`make optuna-dashboard` opens the dashboard on the store. Trials made while
notebooktester runs go to `artefacts/optuna/test.db`, so a test never mixes with a search.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna_dashboard import save_note
from runlog import ROLES, SESSION, _num, describe

# The axes of a search space, under the names a space reads best with:
# `Int(0, 3)`, `Int(32, 1024, log=True)`, `Float(1e-4, 1e-2, log=True)`, `Choice([...])`.
# Both ends of a range are included.
Choice = CategoricalDistribution
Float = FloatDistribution
Int = IntDistribution

# set_metric_names and the artifact store are marked experimental in Optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

TESTING = bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))
ROOT = Path(__file__).resolve().parents[2] / "artefacts" / "optuna"
STORE = ROOT / ("test.db" if TESTING else "studies.db")
ARTIFACTS = ROOT / ("test-artifacts" if TESTING else "artifacts")

# The values a string param can take, per param name: `CHOICES["encoder"] = [...]`.
# `log_run` needs them, because Optuna holds a categorical param to one list of choices
# for the whole study. A string param without an entry is written as a user attr.
CHOICES: dict[str, list[Any]] = {}

_current: optuna.Study | None = None


def storage() -> str:
    """The sqlite URL of the store; its folder is created on first use."""
    ROOT.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STORE}"


def _study(name: str, **kwargs: Any) -> optuna.Study:
    study = optuna.create_study(
        study_name=name,
        storage=storage(),
        direction="maximize",
        load_if_exists=True,
        **kwargs,
    )
    study.set_metric_names(["accuracy"])
    return study


def _axis_line(name: str, axis: Any) -> str:
    if isinstance(axis, CategoricalDistribution):
        if len(axis.choices) == 1:
            return f"{name}={_num(axis.choices[0])}"
        return f"{name} in {{{','.join(_num(v) for v in axis.choices)}}}"
    if isinstance(axis, (FloatDistribution, IntDistribution)):
        kind = "int" if isinstance(axis, IntDistribution) else "U"
        kind = f"log{kind}" if axis.log else kind
        return f"{name} ~ {kind}({_num(axis.low)}, {_num(axis.high)})"
    return f"{name}={_num(axis)}"


def space_line(space: Mapping[str, Any]) -> str:
    """A search space in one line, for example
    `encoder=dinov2-small | hidden_dim ~ logint(32, 1024) | lr ~ logU(1e-4, 1e-2)`."""
    return " | ".join(_axis_line(name, axis) for name, axis in space.items())


def searched_axes(space: Mapping[str, Any]) -> list[str]:
    """The names of the axes a search draws from: every axis that is not one fixed value."""
    return sorted(
        name
        for name, axis in space.items()
        if isinstance(axis, BaseDistribution) and not axis.single()
    )


def search_study(
    name: str,
    space: Mapping[str, Any],
    *,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> optuna.Study:
    """The study `name` in the store, made if it does not exist yet, that maximises
    accuracy over `space`.

    `sampler=None` draws every trial at random and independently; `pruner=None` lets
    every trial run to its last epoch. The study carries its space as user attrs
    (`search_space`, `searched_axes`) next to whatever `attrs` holds. A study that
    exists already gets the new trials added to it; `session` tells the executions apart.
    """
    study = _study(
        name,
        sampler=sampler or optuna.samplers.RandomSampler(),
        pruner=pruner or optuna.pruners.NopPruner(),
    )
    study.set_user_attr("search_space", space_line(space))
    study.set_user_attr("searched_axes", ",".join(searched_axes(space)))
    for key, value in (attrs or {}).items():
        study.set_user_attr(key, value)
    return study


def draw(trial: optuna.Trial, space: Mapping[str, Any]) -> dict[str, Any]:
    """One config from `space`: every axis drawn through the trial, every fixed value as
    it is. The fixed values are written on the trial as user attrs, and the whole config
    as the user attr `config`, so a trial can be rebuilt from the store alone."""
    config = {}
    for name, axis in space.items():
        if isinstance(axis, CategoricalDistribution):
            config[name] = trial.suggest_categorical(name, axis.choices)
        elif isinstance(axis, IntDistribution):
            config[name] = trial.suggest_int(
                name, axis.low, axis.high, step=axis.step, log=axis.log
            )
        elif isinstance(axis, FloatDistribution):
            config[name] = trial.suggest_float(
                name, axis.low, axis.high, step=axis.step, log=axis.log
            )
        else:
            config[name] = axis
            trial.set_user_attr(name, axis)
    trial.set_user_attr("config", config)
    return config


def record(
    trial: optuna.Trial,
    role: str,
    *,
    dataset: str | None = None,
    model: torch.nn.Module | None = None,
    example: Any = None,
    train_seconds: float | None = None,
    curves: Mapping[str, Sequence[float]] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write on `trial` what the class run protocol asks of a run.

    User attrs: `role`, `session`, `dataset`, `train_seconds`, one list per entry of
    `curves` (per epoch; the accuracy per epoch is the trial's own intermediate value),
    everything in `attrs`, and with `model` + `example` the model's `n_params` and
    one-line `architecture`. The layer table of the model becomes the trial's note and
    the artifact `architecture.json`.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    trial.set_user_attr("role", role)
    trial.set_user_attr("session", SESSION)
    if dataset is not None:
        trial.set_user_attr("dataset", dataset)
    if train_seconds is not None:
        trial.set_user_attr("train_seconds", round(float(train_seconds), 3))
    for name, values in (curves or {}).items():
        trial.set_user_attr(name, [round(float(v), 5) for v in values])
    for key, value in (attrs or {}).items():
        trial.set_user_attr(key, value)

    if model is None:
        return
    if example is None:
        raise ValueError("describe(model) needs an example batch input")
    arch = describe(model, example)
    trial.set_user_attr("n_params", arch["n_params"])
    trial.set_user_attr("architecture", arch["signature"])
    save_note(trial, f"`{arch['signature']}`\n\n```\n{arch['summary']}\n```")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "architecture.json"
        path.write_text(json.dumps(arch, indent=1))
        upload_artifact(
            artifact_store=FileSystemArtifactStore(str(ARTIFACTS)),
            file_path=str(path),
            study_or_trial=trial,
        )


def use_study(name: str) -> optuna.Study:
    """Select the study `log_run` writes to, made if it does not exist yet."""
    global _current
    _current = _study(name)
    return _current


def _fixed(name: str, value: Any) -> BaseDistribution | None:
    """The distribution a given param is stored under, or None for a value Optuna cannot
    hold as a param (a string without an entry in `CHOICES`)."""
    if isinstance(value, (bool, np.bool_)):
        return CategoricalDistribution([False, True])
    if isinstance(value, (int, np.integer)):
        return IntDistribution(int(value), int(value))
    if isinstance(value, (float, np.floating)):
        return FloatDistribution(float(value), float(value))
    if name in CHOICES and value in CHOICES[name]:
        return CategoricalDistribution(CHOICES[name])
    return None


def log_run(
    name: str,
    role: str,
    *,
    dataset: str | None = None,
    params: Mapping[str, Any] | None = None,
    scores: Mapping[str, Sequence[float]],
    model: torch.nn.Module | None = None,
    example: Any = None,
    train_seconds: float | None = None,
    attrs: Mapping[str, Any] | None = None,
    study: optuna.Study | None = None,
) -> int:
    """Write one run that is not a draw from a search space as a trial, and return its
    number. It goes to `study`, or to the study selected with `use_study`.

    - `params`: the choices the run was given, each under a stable name. They are the
      trial's params, so the dashboard can put them on an axis.
    - `scores`: `{"accuracy": [...]}`, one value per repeat. The mean is the trial's
      value, every repeat an intermediate value (step = repeat), and the spread goes
      into the user attrs `accuracy_std` and `n_seeds`.
    - everything else as in `record`; `name` is the user attr `name`.
    """
    target = study or _current
    if target is None:
        raise RuntimeError("select a study first: use_study('lesson1-notebook-runs')")
    values = np.asarray(scores["accuracy"], dtype=float)

    given, distributions, loose = {}, {}, {}
    for key, value in (params or {}).items():
        dist = _fixed(key, value)
        if dist is None:
            loose[key] = value
        else:
            given[key] = value.item() if isinstance(value, np.generic) else value
            distributions[key] = dist

    target.enqueue_trial(given)
    trial = target.ask(distributions)
    for repeat, value in enumerate(values):
        trial.report(float(value), repeat)
    record(
        trial,
        role,
        dataset=dataset,
        model=model,
        example=example,
        train_seconds=train_seconds,
        attrs={
            "name": name,
            "accuracy_std": round(float(values.std()), 5),
            "n_seeds": len(values),
            **loose,
            **(attrs or {}),
        },
    )
    target.tell(trial, float(values.mean()))
    return trial.number
