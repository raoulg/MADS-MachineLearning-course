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
import re
import tempfile
import time
import uuid
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
from torchinfo import summary

# The axes of a search space, under the names a space reads best with:
# `Int(0, 3)`, `Int(32, 1024, log=True)`, `Float(1e-4, 1e-2, log=True)`, `Choice([...])`.
# Both ends of a range are included.
Choice = CategoricalDistribution
Float = FloatDistribution
Int = IntDistribution

# set_metric_names and the artifact store are marked experimental in Optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# What a run is for. The class view compares baselines with finals per student and reads
# trials as the search space everyone explored.
ROLES = ("baseline", "trial", "final", "ablation", "hypothesis")

STUDY_PLACEHOLDER = "name-your-study"
# <first name>-<what it studies>: who it is from, and what it is about
STUDY_NAME = re.compile(r"[a-z]+-[a-z0-9][a-z0-9-]{1,38}")

# One timestamp per execution of a notebook (this module is imported once, at the top):
# every trial carries it as the user attr `session`, so a notebook run twice can be told
# apart and the class view keeps only a student's latest session of the handed-out runs.
SESSION = time.strftime("%Y-%m-%d %H:%M:%S")

# The largest design script a submission carries; matches MAX_ARTIFACT_BYTES in
# scripts/run_submission.py, which measures the text as JSON.
MAX_DESIGN_BYTES = 256_000

TESTING = bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))
ROOT = Path(__file__).resolve().parents[2] / "artefacts" / "optuna"
STORE = ROOT / ("test.db" if TESTING else "studies.db")
ARTIFACTS = ROOT / ("test-artifacts" if TESTING else "artifacts")

# The values a string param can take, per param name: `CHOICES["encoder"] = [...]`.
# `log_run` needs them, because Optuna holds a categorical param to one list of choices
# for the whole study. A string param without an entry is written as a user attr.
CHOICES: dict[str, list[Any]] = {}

_current: optuna.Study | None = None

# A classmate's run as the class store names it in its tag `submission_ref`: the class
# runs repository's path of the run's file, without `.json`.
SUBMISSION_REF = re.compile(
    r"lesson\d+/[A-Za-z0-9][A-Za-z0-9-]{0,38}/[0-9a-f]{32}", re.ASCII
)


def _last_dim(shape: Any) -> int | None:
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[-1], int):
        return shape[-1]
    return None


def describe(model: torch.nn.Module, example: Any) -> dict[str, Any]:
    """The architecture of `model` as data: every layer with its input and output shape.

    `example` is one batch input, exactly what you would pass as `model(example)`: a
    tensor, or a tuple of tensors for a model with two streams. Two rows are enough.
    """
    stats = summary(
        model,
        input_data=[example],
        verbose=0,
        depth=8,
        mode="eval",
        col_names=["input_size", "output_size", "num_params"],
    )
    layers = []
    for info in stats.summary_list:
        if info.class_name == "ModuleList":
            continue
        layers.append(
            {
                "depth": info.depth,
                "type": info.class_name,
                "name": info.var_name,
                "input_shape": info.input_size,
                "output_shape": info.output_size,
                "params": info.num_params if info.is_leaf_layer else None,
                "leaf": info.is_leaf_layer,
            }
        )

    steps: list[str] = []
    for layer in layers:
        if not layer["leaf"]:
            continue
        d_in, d_out = _last_dim(layer["input_shape"]), _last_dim(layer["output_shape"])
        if layer["params"] and d_in is not None and d_out is not None:
            steps.append(f"{layer['type']}({d_in}->{d_out})")
        else:
            steps.append(str(layer["type"]))

    return {
        "signature": " > ".join(steps),
        "n_params": int(stats.total_params),
        "n_trainable": int(stats.trainable_params),
        "input_shape": layers[0]["input_shape"] if layers else None,
        "output_shape": layers[0]["output_shape"] if layers else None,
        "layers": layers,
        "summary": str(stats),
    }


def _num(x: Any) -> str:
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float) and 0 < abs(x) < 0.1:
        mantissa, exponent = f"{x:e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def check_study_name(name: Any) -> str:
    """`name`, stripped and in lowercase, if it can name a study; otherwise a `ValueError`
    that says what to do. The placeholder `name-your-study` passes only under notebooktester."""
    testing = bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))
    name = name.strip().lower() if isinstance(name, str) else ""
    if not name or (name == STUDY_PLACEHOLDER and not testing):
        raise ValueError(
            f"name your study: set STUDY to your first name and what it studies "
            f"instead of {STUDY_PLACEHOLDER!r}, for example 'sam-depth-on-eurosat'"
        )
    if not STUDY_NAME.fullmatch(name):
        raise ValueError(
            f"study name {name!r}: use your first name, a dash, and what the study "
            "studies, in lowercase letters, digits and dashes, for example "
            "'sam-depth-on-eurosat'"
        )
    return name


def _builds_on(builds_on: str | Sequence[str] | None) -> list[str]:
    """The submission_refs in `builds_on` (one string, comma-separated, or a list of
    them), stripped and without empty entries; a `ValueError` names one that is not a
    submission_ref."""
    if builds_on is None:
        return []
    parts = [builds_on] if isinstance(builds_on, str) else list(builds_on)
    refs = [ref.strip() for part in parts for ref in part.split(",") if ref.strip()]
    for ref in refs:
        if not SUBMISSION_REF.fullmatch(ref):
            raise ValueError(
                f"builds_on: {ref!r} is not a submission_ref lesson<N>/<login>/<run_id>; "
                "copy the user attr submission_ref from the classmate's run in the class view"
            )
    return refs


def _design_text(design: str | Path | None) -> str | None:
    text = Path(design).read_text() if design is not None else None
    if text is not None and len(json.dumps(text)) > MAX_DESIGN_BYTES:
        raise ValueError(
            f"design: {design} is {len(json.dumps(text))} bytes as JSON, more than a "
            f"submission carries ({MAX_DESIGN_BYTES}); keep only the architectures in it"
        )
    return text


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
    builds_on: str | Sequence[str] | None = None,
    design: str | Path | None = None,
) -> None:
    """Write on `trial` what the class run protocol asks of a run.

    User attrs: `run_id` (32 hex characters, the id a classmate's `builds_on` and the
    class view's `submission_ref` name; made once), `role`, `session`, `dataset`,
    `train_seconds`, one list per entry of `curves` (per epoch; the accuracy per epoch is
    the trial's own intermediate value), everything in `attrs`, and with `model` +
    `example` the model's `n_params` and one-line `architecture`. The layer table of the
    model becomes the trial's note and the artifact `architecture.json`.

    - `builds_on`: the classmate's run (or runs) this one builds on, each named by its
      `submission_ref` (`lesson1/<login>/<run_id>`); written as the user attr `builds_on`,
      comma-separated. An entry that is not a submission_ref raises a `ValueError`
      before anything is written.
    - `design`: the path of your design script, the file that defines the architectures;
      its text becomes the artifact `design.py`. A script larger than a submission
      carries (`MAX_DESIGN_BYTES`) raises a `ValueError` before anything is written.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    refs = _builds_on(builds_on)
    design_text = _design_text(design)
    if "run_id" not in trial.user_attrs:
        trial.set_user_attr("run_id", uuid.uuid4().hex)
    if refs:
        trial.set_user_attr("builds_on", ",".join(refs))
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

    artifacts: dict[str, str] = {}
    if model is not None:
        if example is None:
            raise ValueError("describe(model) needs an example batch input")
        arch = describe(model, example)
        trial.set_user_attr("n_params", arch["n_params"])
        trial.set_user_attr("architecture", arch["signature"])
        save_note(trial, f"`{arch['signature']}`\n\n```\n{arch['summary']}\n```")
        artifacts["architecture.json"] = json.dumps(arch, indent=1)
    if design_text is not None:
        artifacts["design.py"] = design_text
    if not artifacts:
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in artifacts.items():
            path = Path(tmp) / name
            path.write_text(text)
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
    builds_on: str | Sequence[str] | None = None,
    design: str | Path | None = None,
) -> int:
    """Write one run that is not a draw from a search space as a trial, and return its
    number. It goes to `study`, or to the study selected with `use_study`.

    - `params`: the choices the run was given, each under a stable name. They are the
      trial's params, so the dashboard can put them on an axis.
    - `scores`: `{"accuracy": [...]}`, one value per repeat. The mean is the trial's
      value, every repeat an intermediate value (step = repeat), and the spread between
      the repeats goes into the user attrs `accuracy_std` and `n_repeats`.
    - everything else as in `record`, `builds_on` and `design` included; `name` is the
      user attr `name`.
    """
    target = study or _current
    if target is None:
        raise RuntimeError("select a study first: use_study('lesson1-notebook-runs')")
    _builds_on(builds_on)  # a bad argument raises before a trial exists
    _design_text(design)
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
        builds_on=builds_on,
        design=design,
        attrs={
            "name": name,
            "accuracy_std": round(float(values.std()), 5),
            "n_repeats": len(values),
            **loose,
            **(attrs or {}),
        },
    )
    target.tell(trial, float(values.mean()))
    return trial.number
