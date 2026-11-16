"""How a model is trained: vectormesh's Trainer, timed, and scored with every metric of
`hpo.metrics` on the whole test set after every epoch."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader
from vectormesh.training import Trainer, TrainerSettings

from hpo.metrics import OBJECTIVES, Confusion, Repeats, Scorer, Scores
from hpo.tensors import Batch


class Curves(BaseModel):
    """Per epoch: the losses, and the scalar metrics under their own names."""

    train_loss: list[float] = Field(default_factory=list)
    test_loss: list[float] = Field(default_factory=list)
    accuracy: list[float] = Field(default_factory=list)
    macro_recall: list[float] = Field(default_factory=list)

    def add(self, train_loss: float, test_loss: float, scores: Scores | None) -> None:
        self.train_loss.append(float(train_loss))
        self.test_loss.append(float(test_loss))
        if scores is not None:
            for m in OBJECTIVES:
                getattr(self, m).append(scores.get(m))

    def attrs(self) -> dict[str, list[float]]:
        """Every curve that holds an epoch, as a run logs it."""
        return {
            name: [round(v, 5) for v in values]
            for name, values in self.model_dump().items()
            if values
        }


class Fit(BaseModel):
    """What training a model gave: the confusion after the last epoch, the seconds spent
    in training steps, and the curves."""

    confusion: Confusion | None = None
    seconds: float = 0.0
    curves: Curves = Field(default_factory=Curves)

    def scores(self) -> Scores:
        if self.confusion is None:
            raise ValueError("this fit ran no epoch, so it has no scores")
        return self.confusion.scores()

    def as_repeats(self) -> Repeats:
        """This fit as a run of one repeat."""
        if self.confusion is None:
            raise ValueError("this fit ran no epoch, so it has no scores")
        return Repeats(runs=[self.confusion], seconds=self.seconds)


# Called after every epoch with the epoch's number and the fit so far; it may raise
# optuna.TrialPruned to stop the training.
OnEpoch = Callable[[int, Fit], None]


class TimedTrainer(Trainer):
    """A Trainer that adds up the seconds spent in training steps. Evaluation after every
    epoch is not counted: `train_seconds` is the cost of fitting, not of scoring."""

    train_seconds: float = 0.0

    def trainbatches(self) -> float:
        start = time.perf_counter()
        loss = super().trainbatches()
        self.train_seconds += time.perf_counter() - start
        return loss


class Training(BaseModel):
    """How long and how fast a model learns, and where its trainer may write."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lr: float = 1e-3
    epochs: int
    logdir: Path
    optimizer: type[torch.optim.Optimizer] = torch.optim.Adam

    def fit(
        self,
        model: nn.Module,
        train: DataLoader[Batch],
        test: DataLoader[Batch],
        classes: Sequence[str],
        on_epoch: OnEpoch | None = None,
        loss_fn: nn.Module | None = None,
    ) -> Fit:
        """Train `model` and score it on `test` after every epoch; `on_epoch` sees the
        fit after each one."""
        scorer = Scorer(test, classes)
        fit = Fit()

        def report(
            epoch: int,
            train_loss: float,
            test_loss: float,
            metric_dict: dict[str, float],  # empty: the trainer computes no metric
        ) -> None:
            fit.confusion = scorer(model)
            fit.seconds = trainer.train_seconds
            fit.curves.add(train_loss, test_loss, fit.confusion.scores())
            if on_epoch is not None:
                on_epoch(epoch, fit)

        settings = TrainerSettings(
            epochs=self.epochs,
            metrics=[],  # every metric comes from the scorer, over the whole test set
            logdir=self.logdir,
            train_steps=len(train),
            valid_steps=len(test),
            optimizer_kwargs={"lr": self.lr},
            earlystop_kwargs=None,
        )
        trainer = TimedTrainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
            optimizer=self.optimizer,
            traindataloader=train,
            validdataloader=test,
            progress=False,
            reporters=[report],
        )
        trainer.loop()
        fit.seconds = trainer.train_seconds
        return fit


def repeat(
    make: Callable[[], nn.Module], fit: Callable[[nn.Module], Fit], n: int
) -> Repeats:
    """`n` runs of the model `make()` builds, each from other random weights (nothing is
    seeded); `fit(model)` trains one."""
    fits = [fit(make()) for _ in range(n)]
    confusions = [f.confusion for f in fits if f.confusion is not None]
    seconds = sum(f.seconds for f in fits) / max(1, len(fits))
    return Repeats(runs=confusions, seconds=seconds)
