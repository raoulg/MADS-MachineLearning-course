"""Every metric of a classification run, from one confusion matrix over the whole test set.

A run logs all of them, whatever its study searches: a question a student asks later
(does the gate help the rare class?) is then answered by runs that are already shared.
It logs the confusion matrix itself as well, so a metric that counts (balanced accuracy,
Cohen's kappa) can be added later and read from those same runs.

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

The confusion is counted over the whole test set at once. An average of per-batch values
is not the same number for macro recall, precision or F1, because a batch holds few rows
of a rare class, so these are never averaged over batches.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, get_args

import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

from hpo.measure.tensors import Batch, Counts, Labels, PerRepeat, Targets

# The metrics with one value for the whole test set; one of them is a study's objective.
Objective = Literal["accuracy", "macro_recall"]
OBJECTIVES: tuple[Objective, ...] = get_args(Objective)
# The metrics with one value per class.
PerClass = Literal["recall", "precision", "f1"]
PER_CLASS: tuple[PerClass, ...] = get_args(PerClass)
# A scalar (`macro_recall`), or one class of a metric per class (`recall/F`).
MetricName = str
DECIMALS = 5


class Scores(BaseModel):
    """Every metric of one run, or the mean or the spread of them over its repeats."""

    model_config = ConfigDict(frozen=True)

    accuracy: float
    macro_recall: float
    recall: dict[str, float]
    precision: dict[str, float]
    f1: dict[str, float]

    def get(self, name: MetricName) -> float:
        """One metric: `scores.get("macro_recall")`, `scores.get("recall/F")`."""
        head, _, cls = name.partition("/")
        if head in OBJECTIVES and not cls:
            return getattr(self, head)
        if head in PER_CLASS and cls in getattr(self, head):
            return getattr(self, head)[cls]
        raise KeyError(f"no metric {name!r} in {self.names()}")

    def names(self) -> list[MetricName]:
        """Every metric by the name `get` takes."""
        return [*OBJECTIVES, *(f"{m}/{c}" for m in PER_CLASS for c in getattr(self, m))]

    def scalars(self) -> dict[Objective, float]:
        return {m: getattr(self, m) for m in OBJECTIVES}

    def rounded(self) -> Scores:
        """The same scores to `DECIMALS` decimals, as a run logs them."""
        return Scores(
            **{m: round(v, DECIMALS) for m, v in self.scalars().items()},
            **{
                m: {c: round(v, DECIMALS) for c, v in getattr(self, m).items()}
                for m in PER_CLASS
            },
        )


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, with 0 where den is 0 (a class with no rows, or never predicted)."""
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


class Confusion(BaseModel):
    """The confusion matrix of a run, under the names of its classes: row = the true
    class, column = the predicted class."""

    model_config = ConfigDict(frozen=True)

    classes: list[str]
    counts: list[list[int]]

    @model_validator(mode="after")
    def _square(self) -> Confusion:
        n = len(self.classes)
        if len(self.counts) != n or any(len(row) != n for row in self.counts):
            raise ValueError(f"a confusion of {n} classes is {n} x {n} counts")
        return self

    @classmethod
    @jaxtyped(typechecker=beartype)
    def of(cls, labels: Labels, predicted: Labels, classes: Sequence[str]) -> Confusion:
        """The confusion of the predictions `predicted` against the labels `labels`,
        both class indices into `classes`."""
        n = len(classes)
        flat = labels.long().cpu() * n + predicted.long().cpu()
        counts = torch.bincount(flat, minlength=n * n).reshape(n, n)
        return cls(classes=[str(c) for c in classes], counts=counts.tolist())

    def matrix(self) -> Counts:
        return np.asarray(self.counts, dtype=np.int64).reshape(
            len(self.classes), len(self.classes)
        )

    def __add__(self, other: Confusion) -> Confusion:
        """The counts of two runs on the same classes together."""
        if other.classes != self.classes:
            raise ValueError("only confusions of the same classes add up")
        return Confusion(
            classes=self.classes, counts=(self.matrix() + other.matrix()).tolist()
        )

    def scores(self) -> Scores:
        matrix = self.matrix()
        hit = np.diag(matrix).astype(float)
        support = matrix.sum(axis=1)
        recall = _ratio(hit, support)
        precision = _ratio(hit, matrix.sum(axis=0))
        f1 = _ratio(2 * precision * recall, precision + recall)
        present = support > 0

        def per_class(values: np.ndarray, only_present: bool) -> dict[str, float]:
            return {
                name: float(v)
                for name, v, keep in zip(self.classes, values, present, strict=True)
                if keep or not only_present
            }

        return Scores(
            accuracy=float(hit.sum() / max(1, matrix.sum())),
            # a class without test rows has no recall to average
            macro_recall=float(recall[present].mean()) if present.any() else 0.0,
            recall=per_class(recall, only_present=True),
            precision=per_class(precision, only_present=False),
            f1=per_class(f1, only_present=True),
        )


@jaxtyped(typechecker=beartype)
def labels_of(targets: Targets) -> Labels:
    """Class indices from a batch of targets: one-hot rows, or indices already."""
    return targets.argmax(dim=-1) if targets.ndim > 1 else targets.long()


class Scorer:
    """The confusion of a model on one test loader: `Scorer(test_dl, classes)(model)`."""

    def __init__(self, batches: Iterable[Batch], classes: Sequence[str]) -> None:
        self.batches = batches
        self.classes = [str(c) for c in classes]

    @jaxtyped(typechecker=beartype)
    def predict(self, model: nn.Module) -> tuple[Labels, Labels]:
        """(labels, predictions) of `model` over every batch, in eval mode and without
        gradients; the model's mode is restored afterwards."""
        training = model.training
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        truth, guess = [], []
        try:
            model.eval()
            with torch.no_grad():
                for x, y in self.batches:
                    if isinstance(x, tuple | list):
                        x = type(x)(part.to(device) for part in x)
                    else:
                        x = x.to(device)
                    guess.append(model(x).argmax(dim=-1).cpu())
                    truth.append(labels_of(y).cpu())
        finally:
            model.train(training)
        return torch.cat(truth), torch.cat(guess)

    def __call__(self, model: nn.Module) -> Confusion:
        truth, guess = self.predict(model)
        return Confusion.of(truth, guess, self.classes)


class Repeats(BaseModel):
    """The repeats of one run, each from other random weights: the confusion of every
    repeat and the mean seconds a repeat spent in training steps."""

    model_config = ConfigDict(frozen=True)

    runs: list[Confusion]
    seconds: float

    @model_validator(mode="after")
    def _some(self) -> Repeats:
        if not self.runs:
            raise ValueError("a run has at least one repeat")
        if any(r.classes != self.runs[0].classes for r in self.runs):
            raise ValueError("the repeats of a run are scored on the same classes")
        return self

    def __len__(self) -> int:
        return len(self.runs)

    def scores(self) -> list[Scores]:
        return [r.scores() for r in self.runs]

    def logged(self) -> list[Objective]:
        """The scalar metrics: a run with its confusion logs them all."""
        return list(OBJECTIVES)

    def values(self, name: MetricName = "accuracy") -> PerRepeat:
        """One metric of every repeat: `repeats.values("recall/F")`."""
        return np.array([s.get(name) for s in self.scores()], dtype=float)

    def mean(self) -> Scores:
        return self._over(spread=False)

    def std(self) -> Scores:
        """The spread over the repeats, in the shape of the scores."""
        return self._over(spread=True)

    def confusion(self) -> Confusion:
        """The counts of every repeat together."""
        total = self.runs[0]
        for run in self.runs[1:]:
            total = total + run
        return total

    def _over(self, spread: bool) -> Scores:
        scores = self.scores()
        first = scores[0]

        def of(name: MetricName) -> float:
            values = np.array([s.get(name) for s in scores], dtype=float)
            return float(values.std() if spread else values.mean())

        return Scores(
            **{m: of(m) for m in OBJECTIVES},
            **{m: {c: of(f"{m}/{c}") for c in getattr(first, m)} for m in PER_CLASS},
        )


class Scalars(BaseModel):
    """The repeats of a run that has the scalar metrics only, such as a nearest
    neighbour floor or a run scored by a trainer's own accuracy."""

    model_config = ConfigDict(frozen=True)

    accuracy: list[float] | None = None
    macro_recall: list[float] | None = None
    seconds: float = 0.0

    @model_validator(mode="after")
    def _some(self) -> Scalars:
        given = [v for v in (self.accuracy, self.macro_recall) if v is not None]
        if not given or not all(given):
            raise ValueError("Scalars: give accuracy, macro_recall or both, per repeat")
        if len({len(v) for v in given}) > 1:
            raise ValueError("Scalars: every metric has one value per repeat")
        return self

    def __len__(self) -> int:
        return len(next(v for v in (self.accuracy, self.macro_recall) if v))

    def logged(self) -> list[Objective]:
        return [m for m in OBJECTIVES if getattr(self, m) is not None]

    def values(self, name: MetricName = "accuracy") -> PerRepeat:
        if name not in self.logged():
            raise KeyError(f"no metric {name!r}: this run logs {self.logged()}")
        return np.array(getattr(self, name), dtype=float)


# What a run logged, per repeat: every metric, or the scalars alone.
Results = Repeats | Scalars
