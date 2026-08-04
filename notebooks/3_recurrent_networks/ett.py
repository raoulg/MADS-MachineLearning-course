"""The ETTh1 series, loaded once and imported by every notebook in this lesson.

`ETDataset/ett`, config `h1` — hourly readings from an electricity transformer in a
Chinese county, July 2016 onward. The target is **OT**, the transformer's oil
temperature; alongside it run six power-load covariates (HUFL, HULL, MUFL, MULL,
LUFL, LULL — High/Middle/Low UseFul and UseLess load). CC-BY-4.0. It is the
benchmark Informer, Autoformer, FEDformer, DLinear and PatchTST all report on, so
the numbers this lesson produces sit next to numbers you can look up.

Two things this module does that are worth knowing, because both are the kind of
detail that quietly wrecks a timeseries experiment.

**1. The loader.** `datasets` 5.0 refuses script-based datasets, and `ETDataset/ett`
is one. The way around it is the auto-converted `refs/convert/parquet` branch with
explicit `data_files` — the same escape hatch lesson 3 uses for Monash. It is not a
different dataset, it is the same rows served as parquet.

**2. The splits are prefixes of one another, not disjoint.** This is the GluonTS
rolling-evaluation convention and it surprises everyone the first time. The `train`
split holds **one row**: the first 8640 hours. The `validation` split holds **120
rows**, and row *i* is that same series extended by `24 * (i + 1)` hours. `test`
holds 240 rows on the same pattern. So every validation row *contains the entire
training series inside it*, and the longest test row (14400 hours) contains
everything there is. Hand those splits straight to a model as if they were disjoint
and you have leaked your training data into your test set — while doing exactly what
the split names told you to do.

The honest move is the one this module makes: take the longest row, which *is* the
whole series, and cut it in time yourself. 14400 hours = 600 days = the canonical
12/4/4-month split (8640 / 2880 / 2880) that every ETTh1 paper uses. The boundaries
are read off the dataset's own row lengths rather than typed in, so they stay right
if the dataset ever changes.

Everything after loading — splitting, scaling, windowing — stays in the notebooks.
That is the lesson, not the plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

REPO = "ETDataset/ett"
REVISION = "refs/convert/parquet"
CONFIG = "h1"
FREQ = "h"

#: order fixed by the dataset builder: data[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]]
COVARIATE_NAMES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
TARGET_NAME = "OT"


@dataclass(frozen=True)
class ETT:
    """One transformer's history: the target, its covariates, and when each reading happened.

    values      (T,)      oil temperature, degrees Celsius
    covariates  (T, 6)    the six power-load channels, columns in COVARIATE_NAMES order
    timestamps  (T,)      hourly DatetimeIndex, so calendar features are constructible
    n_train     int       index where the 12-month training span ends
    n_val       int       index where the following 4-month validation span ends
    """

    values: torch.Tensor
    covariates: torch.Tensor
    timestamps: pd.DatetimeIndex
    n_train: int
    n_val: int

    def __len__(self) -> int:
        return len(self.values)

    @property
    def splits(self) -> dict[str, slice]:
        """The three spans as slices, so a notebook can loop over them by name."""
        return {
            "train": slice(0, self.n_train),
            "val": slice(self.n_train, self.n_val),
            "test": slice(self.n_val, len(self)),
        }

    def __repr__(self) -> str:
        a, b = self.timestamps[0], self.timestamps[-1]
        return (
            f"ETT({TARGET_NAME}, {len(self)} hourly steps, {a:%Y-%m-%d} .. {b:%Y-%m-%d}, "
            f"{self.covariates.shape[1]} covariates, "
            f"split {self.n_train}/{self.n_val - self.n_train}/{len(self) - self.n_val})"
        )


def _row(split: str, index: int):
    """One row of one split, from the parquet branch."""
    ds = load_dataset(
        REPO,
        revision=REVISION,
        data_files={split: f"{CONFIG}/{split}/0000.parquet"},
    )[split]
    return ds[index if index >= 0 else len(ds) + index], len(ds)


def load_ett() -> ETT:
    """The full ETTh1 series with the canonical 12/4/4-month boundaries.

    Reads three rows to do it: the last test row (the longest — the whole series),
    the single train row (to learn where 12 months ends) and the last validation row
    (where 16 months ends). Nothing is hardcoded except the dataset id.
    """
    full, _ = _row("test", -1)
    train_row, _ = _row("train", 0)
    val_row, _ = _row("validation", -1)

    values = torch.tensor(np.asarray(full["target"], dtype="float32"))
    covariates = torch.tensor(np.asarray(full["feat_dynamic_real"], dtype="float32")).T
    timestamps = pd.date_range(start=full["start"], periods=len(values), freq=FREQ)

    return ETT(
        values=values,
        covariates=covariates,
        timestamps=timestamps,
        n_train=len(train_row["target"]),
        n_val=len(val_row["target"]),
    )


def calendar_features(timestamps: pd.DatetimeIndex) -> torch.Tensor:
    """(T, 4) — hour, day-of-week, day-of-month and day-of-year, each scaled to [-0.5, 0.5].

    The four time features the transformers-library forecasters expect in
    `past_time_features` / `future_time_features`, and the same four every ETTh1 paper
    uses. Scaling rather than one-hot keeps the tensor narrow; the *cyclical* encoding
    (sin/cos of the same quantities) is a design choice you get to make yourself in
    `ts_features.py`.
    """
    t = pd.Series(timestamps)
    return torch.tensor(
        np.stack(
            [
                t.dt.hour / 23.0 - 0.5,
                t.dt.dayofweek / 6.0 - 0.5,
                (t.dt.day - 1) / 30.0 - 0.5,
                (t.dt.dayofyear - 1) / 365.0 - 0.5,
            ],
            axis=-1,
        ),
        dtype=torch.float32,
    )


def window_index(n: int, size: int, stride: int = 1) -> torch.Tensor:
    """(n_windows, size) index matrix — the broadcast trick derived in 01_windowing.

    `stride` exists because consecutive hourly windows overlap by everything but one
    step: at W=96 two neighbours share 95/96 of their content, so striding by 24 costs
    almost no information and shrinks an epoch 24x.
    """
    starts = torch.arange(0, n - size + 1, stride).reshape(-1, 1)
    return torch.arange(size).reshape(1, -1) + starts
