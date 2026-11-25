"""A set of repeats as its mean and spread, and against another set, the difference of
their means."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from hpo.measure.tensors import PerRepeat


class Comparison(BaseModel):
    """The mean and spread of `name`'s repeats; against another set, also the difference
    of the means in points."""

    model_config = ConfigDict(frozen=True)

    name: str
    values: list[float]
    mean: float
    std: float
    points: float | None = None

    @classmethod
    def of(
        cls, name: str, scores: PerRepeat, versus: PerRepeat | None = None
    ) -> Comparison:
        points = None
        if versus is not None:
            points = 100 * float(scores.mean() - versus.mean())
        return cls(
            name=name,
            values=[float(v) for v in scores],
            mean=float(scores.mean()),
            std=float(scores.std()),
            points=points,
        )

    def __str__(self) -> str:
        line = f"{self.name:<32} {self.mean:.4f} +/- {self.std:.4f}   (n={len(self.values)})"
        if self.points is not None:
            line += f"   {self.points:+.2f} pt"
        return line
