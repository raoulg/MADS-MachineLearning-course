"""The metrics every run of a classification problem logs, from one confusion matrix.

A run logs all of them, whatever its study searches: a question a student asks later
(does the gate help the rare class?) is then answered by runs that are already shared.

    accuracy       the fraction of test rows predicted right
    macro_recall   the mean over the classes of each class's recall: every class counts
                   as much, however rare it is
    recall         per class: of the rows of that class, the fraction predicted as it
    precision      per class: of the rows predicted as that class, the fraction that is it
                   (0 for a class that is never predicted)
    f1             per class: the harmonic mean of the two

A class without rows in the test set has no recall and no F1: it is left out of `recall`,
`f1` and the mean of `macro_recall`, rather than counted as 0. It keeps its precision,
since a model can still predict it.

`scores` computes them over the whole test set at once. An average of per-batch values is
not the same number for macro recall, precision or F1, because a batch holds few rows of
a rare class, so these are never averaged over batches.

As user attrs of a trial, `metrics` is the dict `scores` gives (the mean over the repeats
for a run that was repeated) and `metrics_std` the same shape with the spread over the
repeats. The scalar metrics are also logged per epoch, as lists under their own name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

# The metrics with one value for the whole test set; one of them is a study's objective.
SCALARS = ("accuracy", "macro_recall")
# The metrics with one value per class.
PER_CLASS = ("recall", "precision", "f1")
OBJECTIVES = SCALARS


def confusion(y_true: Any, y_pred: Any, n_classes: int) -> np.ndarray:
    """The confusion matrix: row = the true class, column = the predicted class."""
    t = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    return np.bincount(t * n_classes + p, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, with 0 where den is 0 (a class with no rows, or never predicted)."""
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def from_confusion(matrix: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    """Every metric of `matrix`, per class under the names in `classes`."""
    matrix = np.asarray(matrix)
    hit = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = _ratio(hit, support)
    precision = _ratio(hit, predicted)
    f1 = _ratio(2 * precision * recall, precision + recall)
    present = support > 0
    names = [str(c) for c in classes]
    if len(names) != len(hit):
        raise ValueError(f"{len(names)} class names for {len(hit)} classes")

    def per_class(values: np.ndarray, only_present: bool) -> dict[str, float]:
        return {
            name: round(float(v), 5)
            for name, v, keep in zip(names, values, present, strict=True)
            if keep or not only_present
        }

    return {
        "accuracy": float(hit.sum() / max(1, matrix.sum())),
        # a class without test rows has no recall to average
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "recall": per_class(recall, only_present=True),
        "precision": per_class(precision, only_present=False),
        "f1": per_class(f1, only_present=True),
    }


def scores(y_true: Any, y_pred: Any, classes: Sequence[str]) -> dict[str, Any]:
    """Every metric of the predictions `y_pred` against the labels `y_true` (class
    indices into `classes`)."""
    return from_confusion(confusion(y_true, y_pred, len(classes)), classes)


def _labels(target: torch.Tensor) -> torch.Tensor:
    """Class indices from a target batch: one-hot rows or indices already."""
    return target.argmax(dim=-1) if target.ndim > 1 else target


def predict(
    model: nn.Module, batches: Iterable[tuple[Any, torch.Tensor]]
) -> tuple[np.ndarray, np.ndarray]:
    """(labels, predictions) of `model` over every batch, in eval mode and without
    gradients; the model's mode is restored afterwards."""
    training = model.training
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    truth, guess = [], []
    try:
        model.eval()
        with torch.no_grad():
            for x, y in batches:
                if isinstance(x, (tuple, list)):
                    x = type(x)(part.to(device) for part in x)
                else:
                    x = x.to(device)
                guess.append(model(x).argmax(dim=-1).cpu())
                truth.append(_labels(y).cpu())
    finally:
        model.train(training)
    return torch.cat(truth).numpy(), torch.cat(guess).numpy()


class Scorer:
    """Every metric of a model on one test loader: `Scorer(test_dl, classes)(model)`."""

    def __init__(
        self, batches: Iterable[tuple[Any, torch.Tensor]], classes: Sequence[str]
    ) -> None:
        self.batches = batches
        self.classes = [str(c) for c in classes]

    def __call__(self, model: nn.Module) -> dict[str, Any]:
        truth, guess = predict(model, self.batches)
        return scores(truth, guess, self.classes)


def _leaves(m: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in m.items():
        if isinstance(value, Mapping):
            out.update(_leaves(value, f"{prefix}{key}/"))
        else:
            out[f"{prefix}{key}"] = float(value)
    return out


def _nest(flat: Mapping[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        head, _, tail = key.partition("/")
        if tail:
            out.setdefault(head, {})[tail] = value
        else:
            out[head] = value
    return out


def summarise(
    repeats: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(mean, spread) over the repeats of a run, each in the shape `scores` gives."""
    if not repeats:
        raise ValueError("summarise: no repeats")
    flat = [_leaves(r) for r in repeats]
    keys = flat[0].keys()
    if any(f.keys() != keys for f in flat):
        raise ValueError("summarise: the repeats do not hold the same metrics")
    values = {k: np.array([f[k] for f in flat]) for k in keys}
    mean = {k: round(float(v.mean()), 5) for k, v in values.items()}
    std = {k: round(float(v.std()), 5) for k, v in values.items()}
    return _nest(mean), _nest(std)


def value(metrics: Mapping[str, Any], name: str) -> float:
    """One metric out of `metrics`: a scalar (`macro_recall`) or one class of a metric
    per class (`recall/F`)."""
    head, _, tail = name.partition("/")
    found = metrics.get(head)
    if tail:
        found = found.get(tail) if isinstance(found, Mapping) else None
    if not isinstance(found, (int, float)):
        raise KeyError(f"no metric {name!r} in {sorted(_leaves(metrics))}")
    return float(found)
