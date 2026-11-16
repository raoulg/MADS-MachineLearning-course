"""What a lesson gives the study of its exercise: how a config becomes data, how a model
is trained on it, which linear model is the baseline, and which params a run logs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from torch import nn
from torch.utils.data import DataLoader
from vectormesh.types import TensorInput

from hpo.space import Config, Param
from hpo.tensors import Batch
from hpo.training import Fit, OnEpoch


@dataclass(frozen=True)
class Data:
    """What a config trains and is scored on: the loaders, the names of the classes, one
    batch input of two rows (for the layer table and the shapes), and the sizes a design
    is built with (`dims`, for example `{"embed_dim": 384}`)."""

    train: DataLoader[Batch]
    test: DataLoader[Batch]
    classes: list[str]
    example: TensorInput
    dims: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Baseline:
    """The baseline of a dataset: the config it is trained with (dataset, lr, epochs, ...)
    and the model it builds on the data of that config."""

    config: Config
    make: Callable[[Data], nn.Module]


class Harness(Protocol):
    """What a lesson's `exercises/harness.py` gives `hpo.exercise.Exercise`."""

    lesson: int
    epochs: int
    repeats: int
    # the keys of a space the harness reads; every other key is the design's
    trainer: frozenset[str]
    # per string param, the values it can take (see `hpo.runs.StudyLog`)
    choices: Mapping[str, Sequence[str]]

    def data(self, config: Config) -> Data: ...

    def fit(
        self,
        model: nn.Module,
        data: Data,
        config: Config,
        on_epoch: OnEpoch | None = None,
    ) -> Fit: ...

    def baseline(self, dataset: str) -> Baseline: ...

    def params(self, config: Config, data: Data) -> dict[str, Param]: ...
