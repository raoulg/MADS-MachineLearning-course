"""What the minutes a student gives a study buy on their machine, as the study attr
`budget`."""

from __future__ import annotations

import platform

from pydantic import BaseModel, ConfigDict

from hpo.runs import StudyLog
from hpo.space import Space


class Budget(BaseModel):
    """`n_trials` is what is left of `minutes` once `reruns` runs of `repeats` repeats
    each are set aside (the baseline and the winner); `trials_per_axis` says how thin
    the search is spread over the axes it searches."""

    model_config = ConfigDict(frozen=True)

    minutes: float
    seconds_per_trial: float
    measured_on: str
    machine: str
    n_trials: int
    repeats: int
    epochs: int | None = None
    searched_axes: int | None = None
    trials_per_axis: float | None = None

    @classmethod
    def at(
        cls,
        minutes: float,
        seconds_per_trial: float,
        measured_on: str,
        *,
        space: Space | None = None,
        epochs: int | None = None,
        repeats: int = 3,
        reruns: int = 2,
    ) -> Budget:
        """The budget at a cost per trial that is known already."""
        seconds = max(float(seconds_per_trial), 1e-3)
        n_trials = max(0, int(minutes * 60 / seconds) - reruns * repeats)
        axes = space.searched() if space is not None else None
        return cls(
            minutes=minutes,
            seconds_per_trial=round(seconds, 2),
            measured_on=measured_on,
            machine=f"{platform.system()} {platform.machine()}",
            n_trials=n_trials,
            repeats=repeats,
            epochs=epochs,
            searched_axes=None if axes is None else len(axes),
            trials_per_axis=None
            if axes is None
            else round(n_trials / max(1, len(axes)), 1),
        )

    @classmethod
    def measure(
        cls,
        minutes: float,
        like: StudyLog,
        *,
        space: Space | None = None,
        epochs: int | None = None,
        repeats: int = 3,
        reruns: int = 2,
    ) -> Budget:
        """The budget at what the finished trials of the search study `like` took on
        this machine, scaled to `epochs`."""
        return cls.at(
            minutes,
            like.seconds_per_trial(epochs),
            like.name,
            space=space,
            epochs=epochs,
            repeats=repeats,
            reruns=reruns,
        )

    def __str__(self) -> str:
        return (
            f"{self.n_trials} trials in {self.minutes} minutes at "
            f"{self.seconds_per_trial}s each, {self.trials_per_axis} per searched axis"
        )
