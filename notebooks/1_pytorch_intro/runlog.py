"""Log runs in the shape the class run protocol expects.

Every run this lesson asks you to submit is logged through `log_run`, which writes the
same things for every student: a `role` tag, the parameters, the score (with its seed
spread when there is one), per-epoch curves when there are some, and the architecture
with the tensor shape at every layer. `scripts/submit_runs.py` exports those runs to the
class repository; `docs/run-protocol.md` describes the whole protocol.

Runs go to `artefacts/mlflow/mlruns.db` at the root of the course repository, unless
`MLFLOW_TRACKING_URI` is set, in which case mlflow's own convention wins.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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


def use_experiment(name: str) -> str:
    """Point mlflow at the course tracking store and select experiment `name`."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{TRACKING_DB}"
    mlflow.set_tracking_uri(uri)
    # A new experiment keeps its artifacts next to a sqlite store, not in whatever
    # directory the notebook happens to run from.
    if uri.startswith("sqlite:///") and mlflow.get_experiment_by_name(name) is None:
        root = Path(uri.removeprefix("sqlite:///")).parent / "artifacts" / name
        mlflow.create_experiment(name, artifact_location=root.as_uri())
    mlflow.set_experiment(name)
    return uri


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
    params: Mapping[str, Any] | None = None,
    scores: Mapping[str, Sequence[float]] | None = None,
    curves: Mapping[str, Sequence[float]] | None = None,
    model: torch.nn.Module | None = None,
    example: Any = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log one run in the protocol's shape and return its run id.

    - `scores`: final values per seed, e.g. `{"accuracy": [0.71, 0.72, 0.70]}`. Logs the
      mean as `accuracy`, the spread as `accuracy_std`, every seed as `accuracy_per_seed`
      (step = seed) and `n_seeds`. A single value is a list of one.
    - `curves`: per-epoch values, e.g. `{"accuracy": [...], "train_loss": [...]}`, logged
      with step = epoch.
    - `model` + `example`: logs `n_params`, the one-line `architecture` tag and the full
      layer table as the artifact `architecture.json`.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    both = set(scores or {}) & set(curves or {})
    if both:
        raise ValueError(f"{sorted(both)} given as both a score and a curve; pick one")

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("role", role)
        if os.environ.get("NOTEBOOKTESTER_RUNNING"):
            mlflow.set_tag("test_run", "true")
        if tags:
            mlflow.set_tags(dict(tags))
        if params:
            mlflow.log_params(dict(params))

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

        return run.info.run_id
