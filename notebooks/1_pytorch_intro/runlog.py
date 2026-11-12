"""Log runs in the shape the class run protocol expects.

Every run this lesson asks you to submit is logged through `log_run`, which writes the
same things for every student: a `role` tag, the dataset, the parameters, the score (with
its seed spread when there is one), per-epoch curves when there are some, and the
architecture with the tensor shape at every layer. `scripts/submit_runs.py` exports those
runs to the class repository; `docs/run-protocol.md` describes the whole protocol.

Two kinds of run reach the class. The notebook as handed out logs through the module's
`log_run` (tag `origin=notebook`) and is the same for everyone. Your own study logs
through a `Study`, which carries its name and its search space on every run (tag
`origin=study`), so the runs themselves say what you searched. The search space is also
written one tag per axis (`space_tags`), so mlflow can filter runs by what they searched.

Runs go to `artefacts/mlflow/mlruns.db` at the root of the course repository, unless
`MLFLOW_TRACKING_URI` is set, in which case mlflow's own convention wins.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torchinfo import summary

TRACKING_DB = Path(__file__).resolve().parents[2] / "artefacts" / "mlflow" / "mlruns.db"

# What a run is for. The class analysis compares baselines with finals per student and
# reads trials as the search space everyone explored.
ROLES = ("baseline", "trial", "final", "ablation", "hypothesis")

STUDY_PLACEHOLDER = "name-your-study"
STUDY_NAME = re.compile(r"[a-z0-9][a-z0-9-]{2,39}")


def _tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{TRACKING_DB}"
    mlflow.set_tracking_uri(uri)
    return uri


def _experiment_id(name: str) -> str:
    """The id of experiment `name`, created if it does not exist yet."""
    uri = _tracking_uri()
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    # A new experiment keeps its artifacts next to a sqlite store, not in whatever
    # directory the notebook happens to run from.
    location = None
    if uri.startswith("sqlite:///"):
        root = Path(uri.removeprefix("sqlite:///")).parent / "artifacts" / name
        location = root.as_uri()
    return mlflow.create_experiment(name, artifact_location=location)


def use_experiment(name: str) -> str:
    """Point mlflow at the course tracking store and select experiment `name`."""
    _experiment_id(name)
    mlflow.set_experiment(name)
    return mlflow.get_tracking_uri()


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


def log_run(
    name: str,
    role: str,
    *,
    dataset: str | None = None,
    params: Mapping[str, Any] | None = None,
    scores: Mapping[str, Sequence[float]] | None = None,
    curves: Mapping[str, Sequence[float]] | None = None,
    model: torch.nn.Module | None = None,
    example: Any = None,
    tags: Mapping[str, str] | None = None,
    train_seconds: float | None = None,
) -> str:
    """Log one run in the protocol's shape, tagged `origin=notebook`, and return its id.

    The run goes to the experiment selected with `use_experiment`.

    - `dataset`: what the score was measured on, written as the tag `dataset`. The class
      analysis never ranks runs of different datasets against each other.
    - `params`: every choice that distinguishes this run from another, each under a stable
      name with a plain value (a number, a short string, or a bool, written `true`/`false`):
      in a parallel-coordinates plot only params and metrics can be axes.
    - `train_seconds`: the time spent in training steps (not vectorizing, not evaluating),
      the mean over seeds when there are several; logged as the metric `train_seconds`.
    - `scores`: final values per seed, e.g. `{"accuracy": [0.71, 0.72, 0.70]}`. Logs the
      mean as `accuracy`, the spread as `accuracy_std`, every seed as `accuracy_per_seed`
      (step = seed) and `n_seeds`. A single value is a list of one.
    - `curves`: per-epoch values, e.g. `{"accuracy": [...], "train_loss": [...]}`, logged
      with step = epoch.
    - `model` + `example`: logs `n_params`, the one-line `architecture` tag and the full
      layer table as the artifact `architecture.json`.
    """
    return _log_run(
        name,
        role,
        dataset=dataset,
        params=params,
        scores=scores,
        curves=curves,
        model=model,
        example=example,
        tags={**(tags or {}), "origin": "notebook"},
        train_seconds=train_seconds,
    )


def _param(value: Any) -> Any:
    return str(value).lower() if isinstance(value, (bool, np.bool_)) else value


def _log_run(
    name: str,
    role: str,
    *,
    dataset: str | None,
    params: Mapping[str, Any] | None,
    scores: Mapping[str, Sequence[float]] | None,
    curves: Mapping[str, Sequence[float]] | None,
    model: torch.nn.Module | None,
    example: Any,
    tags: Mapping[str, str],
    train_seconds: float | None,
    experiment_id: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    both = set(scores or {}) & set(curves or {})
    if both:
        raise ValueError(f"{sorted(both)} given as both a score and a curve; pick one")

    with mlflow.start_run(run_name=name, experiment_id=experiment_id) as run:
        mlflow.set_tag("role", role)
        if os.environ.get("NOTEBOOKTESTER_RUNNING"):
            mlflow.set_tag("test_run", "true")
        if dataset is not None:
            mlflow.set_tag("dataset", dataset)
        mlflow.set_tags(dict(tags))
        if params:
            mlflow.log_params({k: _param(v) for k, v in params.items()})
        if train_seconds is not None:
            mlflow.log_metric("train_seconds", float(train_seconds))

        for metric, values in (curves or {}).items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric, float(value), step=epoch)

        for metric, values in (scores or {}).items():
            arr = np.asarray(values, dtype=float)
            for seed, value in enumerate(arr):
                mlflow.log_metric(f"{metric}_per_seed", float(value), step=seed)
            mlflow.log_metric(metric, float(arr.mean()))
            mlflow.log_metric(f"{metric}_std", float(arr.std()))
            mlflow.log_metric("n_seeds", len(arr))

        if model is not None:
            if example is None:
                raise ValueError("describe(model) needs an example batch input")
            arch = describe(model, example)
            mlflow.set_tag("architecture", arch["signature"][:5000])
            mlflow.log_metric("n_params", arch["n_params"])
            mlflow.log_dict(arch, "architecture.json")

        for path, content in (artifacts or {}).items():
            mlflow.log_dict(content, path)

        return run.info.run_id


def _jsonable(value: Any) -> Any:
    """`value` as plain JSON data; anything JSON has no type for becomes its repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _axis(value: Any) -> dict[str, Any]:
    from ray.tune.search import sample

    if isinstance(value, sample.Categorical):
        return {"kind": "choice", "values": _jsonable(list(value.categories))}
    if isinstance(value, sample.Integer):
        if isinstance(value.get_sampler(), sample.Integer._Uniform):
            return {"kind": "randint", "low": value.lower, "high": value.upper}
    elif isinstance(value, sample.Float):
        sampler = value.get_sampler()
        if isinstance(sampler, sample.Float._LogUniform):
            return {"kind": "loguniform", "low": value.lower, "high": value.upper}
        if isinstance(sampler, sample.Float._Uniform):
            return {"kind": "uniform", "low": value.lower, "high": value.upper}
    if isinstance(value, sample.Domain):
        bounds = [getattr(value, a) for a in ("lower", "upper") if hasattr(value, a)]
        args = ", ".join(
            [*map(_num, bounds), f"sampler={_sampler(value.get_sampler())}"]
        )
        return {"kind": "other", "repr": f"{type(value).__name__}({args})"}
    if isinstance(value, Mapping):
        return {"kind": "other", "repr": repr(_jsonable(value))}
    return {"kind": "fixed", "value": _jsonable(value)}


def _sampler(sampler: Any) -> str:
    """A ray sampler by what it does, without the memory address of its default repr."""
    if hasattr(sampler, "sampler") and hasattr(sampler, "q"):
        return f"Quantized({_sampler(sampler.sampler)}, q={_num(sampler.q)})"
    name = type(sampler).__name__.lstrip("_")
    shape = [
        f"{a}={_num(getattr(sampler, a))}"
        for a in ("mean", "sd", "base")
        if hasattr(sampler, a)
    ]
    return f"{name}({', '.join(shape)})" if shape else name


def describe_space(space: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """A ray tune search space as data, one entry per axis in the order of `space`.

    `tune.choice` gives `{"kind": "choice", "values": [...]}`; `tune.loguniform` gives
    `{"kind": "loguniform", "low": .., "high": ..}`, and likewise `uniform` and `randint`;
    a constant gives `{"kind": "fixed", "value": ..}`. Anything else (`quniform`, `randn`,
    `grid_search`, ...) gives `{"kind": "other", "repr": "..."}`.
    """
    return {str(name): _axis(value) for name, value in space.items()}


def _num(x: Any) -> str:
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float) and 0 < abs(x) < 0.1:
        mantissa, exponent = f"{x:e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def space_line(space: Mapping[str, Any]) -> str:
    """A search space in one line, for example
    `encoder=dinov2-small | hidden_dim in {128,256,512} | lr ~ logU(1e-4, 1e-2)`."""
    parts = []
    for name, axis in describe_space(space).items():
        kind = axis["kind"]
        if kind == "choice" and len(axis["values"]) == 1:
            parts.append(f"{name}={_num(axis['values'][0])}")
        elif kind == "choice":
            parts.append(f"{name} in {{{','.join(_num(v) for v in axis['values'])}}}")
        elif kind == "loguniform":
            parts.append(f"{name} ~ logU({_num(axis['low'])}, {_num(axis['high'])})")
        elif kind == "uniform":
            parts.append(f"{name} ~ U({_num(axis['low'])}, {_num(axis['high'])})")
        elif kind == "randint":
            parts.append(f"{name} ~ randint({axis['low']}, {axis['high']})")
        elif kind == "fixed":
            parts.append(f"{name}={_num(axis['value'])}")
        else:
            parts.append(f"{name} ~ {axis['repr']}")
    return " | ".join(parts)


def _plain(x: Any) -> str:
    """A number as a short plain decimal (0.0003, 0.5, 64), a bool as true/false, a string
    as it is."""
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return np.format_float_positional(x, trim="-")
    return str(x)


def space_tags(space: Mapping[str, Any]) -> dict[str, str]:
    """A search space as mlflow tags that filter per axis.

    One tag per axis, `space_<axis>`, with a short canonical value: `choice[64,256]`,
    `loguniform[0.0003,0.003]`, `uniform[0,0.5]`, `randint[1,4]`, `fixed[dinov2-small]`
    (a choice of one value is fixed; `other[...]` for an axis `describe_space` does not
    know); `searched_axes`, the sorted, comma-separated names of the axes that are not
    fixed; and `search_space`, the line `space_line` gives. In mlflow:
    `tags.space_encoder LIKE 'fixed%'`, or `tags.searched_axes LIKE '%n_hidden_layers%'`.
    """
    tags = {}
    searched = []
    for name, axis in describe_space(space).items():
        kind = axis["kind"]
        if kind == "choice" and len(axis["values"]) == 1:
            value = f"fixed[{_plain(axis['values'][0])}]"
        elif kind == "choice":
            value = f"choice[{','.join(_plain(v) for v in axis['values'])}]"
        elif kind in ("loguniform", "uniform", "randint"):
            value = f"{kind}[{_plain(axis['low'])},{_plain(axis['high'])}]"
        elif kind == "fixed":
            value = f"fixed[{_plain(axis['value'])}]"
        else:
            value = f"other[{axis['repr']}]"
        tags[f"space_{name}"] = value[:5000]
        if not value.startswith("fixed["):
            searched.append(name)
    tags["searched_axes"] = ",".join(sorted(searched))
    tags["search_space"] = space_line(space)[:5000]
    return tags


@dataclass
class Study:
    """Your own study: a name, a search space and a budget.

    Every run logged through `study.log_run` goes to the experiment `lesson<N>-study` and
    carries the study with it: the tags `origin=study`, `study`, `budget` (JSON), the
    search space as `space_tags` writes it (`search_space`, `searched_axes` and one
    `space_<axis>` per axis) and, when `default_space` is given, `space_is_default`; and
    the artifact `search_space.json` (what `describe_space` gives).

        study = Study("depth-on-eurosat", space=param_space,
                      budget={"n_trials": 16, "epochs": 15, "seeds": 3})
        study.log_run("winner", "final", dataset="eurosat", scores={"accuracy": per_seed})
    """

    name: str
    space: Mapping[str, Any]
    budget: Mapping[str, Any]
    default_space: Mapping[str, Any] | None = None
    lesson: int = 1
    tags: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        testing = bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))
        self.name = self.name.strip()
        if not self.name or (self.name == STUDY_PLACEHOLDER and not testing):
            raise ValueError(
                f"name your study: set STUDY to a name of your own instead of "
                f"{STUDY_PLACEHOLDER!r}, for example 'depth-on-eurosat'"
            )
        if not STUDY_NAME.fullmatch(self.name):
            raise ValueError(
                f"study name {self.name!r}: use 3 to 40 lowercase letters, digits and "
                "dashes, starting with a letter or digit, for example 'depth-on-eurosat'"
            )
        self.tags = {
            "origin": "study",
            "study": self.name,
            "budget": json.dumps(_jsonable(self.budget), sort_keys=True),
            **space_tags(self.space),
        }
        if self.default_space is not None:
            same = describe_space(self.space) == describe_space(self.default_space)
            self.tags["space_is_default"] = str(same).lower()

    @property
    def experiment(self) -> str:
        return f"lesson{self.lesson}-study"

    def log_run(
        self,
        name: str,
        role: str,
        *,
        dataset: str | None = None,
        params: Mapping[str, Any] | None = None,
        scores: Mapping[str, Sequence[float]] | None = None,
        curves: Mapping[str, Sequence[float]] | None = None,
        model: torch.nn.Module | None = None,
        example: Any = None,
        tags: Mapping[str, str] | None = None,
        train_seconds: float | None = None,
    ) -> str:
        """`log_run` for this study: same arguments, into `lesson<N>-study`, with the
        study's tags and its `search_space.json`. The selected experiment stays as it is."""
        return _log_run(
            name,
            role,
            dataset=dataset,
            params=params,
            scores=scores,
            curves=curves,
            model=model,
            example=example,
            tags={**(tags or {}), **self.tags},
            train_seconds=train_seconds,
            experiment_id=_experiment_id(self.experiment),
            artifacts={"search_space.json": describe_space(self.space)},
        )
