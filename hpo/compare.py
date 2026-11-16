"""Two sets of repeats, and the difference of their means in units of its standard error."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from hpo.tensors import PerRepeat


class Comparison(BaseModel):
    """The mean and spread of `name`'s repeats; against another set, also the difference
    in points and in sigmas."""

    model_config = ConfigDict(frozen=True)

    name: str
    values: list[float]
    mean: float
    std: float
    points: float | None = None
    sigmas: float | None = None

    @classmethod
    def of(
        cls, name: str, scores: PerRepeat, versus: PerRepeat | None = None
    ) -> Comparison:
        points = sigmas = None
        if versus is not None:
            diff = float(scores.mean() - versus.mean())
            pooled = float(
                np.sqrt(
                    scores.var(ddof=1) / len(scores) + versus.var(ddof=1) / len(versus)
                )
            )
            points = 100 * diff
            sigmas = diff / pooled if pooled > 0 else 0.0
        return cls(
            name=name,
            values=[float(v) for v in scores],
            mean=float(scores.mean()),
            std=float(scores.std()),
            points=points,
            sigmas=sigmas,
        )

    def __str__(self) -> str:
        line = f"{self.name:<32} {self.mean:.4f} +/- {self.std:.4f}   (n={len(self.values)})"
        if self.points is not None and self.sigmas is not None:
            line += f"   {self.points:+.2f} pt, {self.sigmas:+.1f} sigma"
        return line
