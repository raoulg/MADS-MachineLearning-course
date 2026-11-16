"""The axes a study draws from, and the config one draw gives.

A space holds, per name, an axis to draw from or one fixed value:

    Space({
        "dataset": "dtd",                       # fixed: every trial gets it
        "n_hidden_layers": Int(0, 3),           # both ends included
        "hidden_dim": Int(32, 1024, log=True),
        "lr": Float(1e-4, 1e-2, log=True),
        "skip": Choice([False, True]),
    })
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

Choice = CategoricalDistribution
Float = FloatDistribution
Int = IntDistribution
Axis = CategoricalDistribution | FloatDistribution | IntDistribution
# a value a param can take: what a fixed axis holds and what a draw gives
Param = str | int | float | bool
Config = dict[str, Param]


def _num(x: object) -> str:
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float) and 0 < abs(x) < 0.1:
        mantissa, exponent = f"{x:e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def _axis_line(name: str, axis: Axis | Param) -> str:
    if isinstance(axis, CategoricalDistribution):
        if len(axis.choices) == 1:
            return f"{name}={_num(axis.choices[0])}"
        return f"{name} in {{{','.join(_num(v) for v in axis.choices)}}}"
    if isinstance(axis, FloatDistribution | IntDistribution):
        kind = "int" if isinstance(axis, IntDistribution) else "U"
        kind = f"log{kind}" if axis.log else kind
        return f"{name} ~ {kind}({_num(axis.low)}, {_num(axis.high)})"
    return f"{name}={_num(axis)}"


class Space(Mapping[str, Axis | Param]):
    """A search space: per name an axis to draw from, or one fixed value."""

    def __init__(self, axes: Mapping[str, Axis | Param]) -> None:
        self.axes = dict(axes)

    def __getitem__(self, name: str) -> Axis | Param:
        return self.axes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.axes)

    def __len__(self) -> int:
        return len(self.axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Space) and self.line() == other.line()

    def __hash__(self) -> int:
        return hash(self.line())

    def __repr__(self) -> str:
        return f"Space({self.line()})"

    def line(self) -> str:
        """The space in one line, for example
        `encoder=dinov2-small | hidden_dim ~ logint(32, 1024) | lr ~ logU(1e-4, 1e-2)`."""
        return " | ".join(_axis_line(name, axis) for name, axis in self.axes.items())

    def searched(self) -> list[str]:
        """The axes a search draws from: every axis that is not one fixed value."""
        return sorted(
            name
            for name, axis in self.axes.items()
            if isinstance(axis, BaseDistribution) and not axis.single()
        )

    def fixed(self) -> Config:
        """The values that are not drawn, as every trial carries them."""
        return {
            name: axis
            for name, axis in self.axes.items()
            if not isinstance(axis, BaseDistribution)
        }

    def draw(self, trial: optuna.Trial) -> Config:
        """One config: every axis drawn through the trial, every fixed value as it is."""
        config: Config = {}
        for name, axis in self.axes.items():
            if isinstance(axis, CategoricalDistribution):
                choice = trial.suggest_categorical(name, axis.choices)
                if choice is None:
                    raise ValueError(f"the axis {name!r} holds None, which no param is")
                config[name] = choice
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
        return config

    def check(
        self,
        options: Collection[str],
        choices: Collection[str],
        trainer: Collection[str],
    ) -> None:
        """A `ValueError` unless every option of the design is an axis here, and every
        axis is read by the design (`choices`, the fields of its config) or by the
        trainer (`trainer`, the keys the lesson's harness reads)."""
        for axis in options:
            if axis not in self.axes:
                raise ValueError(
                    f"SPACE has no axis {axis!r}, which designs.py marks "
                    f"@option({axis!r}): add it to SPACE, as {axis!r}: "
                    f"Choice([False, True]) to search it or {axis!r}: False to leave "
                    "the stage out of every trial"
                )
        used = set(choices) | set(trainer)
        for key in self.axes:
            if key not in used:
                raise ValueError(
                    f"SPACE has the axis {key!r}, which neither the design nor the "
                    f"trainer uses: give Config in designs.py a field {key!r} and a "
                    "stage that reads it, or take it out of SPACE"
                )
