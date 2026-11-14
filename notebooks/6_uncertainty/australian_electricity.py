"""One Australian state's half-hourly electricity demand, loaded once and imported by
`6_uncertainty/00_forecasting_baselines`, `6_uncertainty/01_probabilistic` and
`3_scaling_and_search/01_gates_highways_moe`.

`Monash-University/monash_tsf`, config `australian_electricity_demand` — five states'
worth of half-hourly demand (MW), 2002 onward, CC-BY-4.0. Three things worth knowing:

**1. Read from the parquet-convert branch.** The dataset ships as a plain dataset; this
module reads it through `refs/convert/parquet`, the revision the other Monash loaders in
this course use.

**2. There are five series (`T1`-`T5`); this module loads `T1`.** The choice is arbitrary
among five near-identical demand series. To use another, change `ITEM_ID` below; nothing
else changes shape.

**3. The splits are cut in calendar time here, not taken from the dataset.** Monash's
train/validation/test rows follow the GluonTS rolling-evaluation convention: they are
prefixes of one another and differ by 60 half-hourly steps (30 hours), far too little to
window. This module takes the longest row (`test`, the whole series) and cuts it itself:
the last full year is test, the year before that validation, everything earlier train.

The series has a clean daily and weekly pattern, which makes it a good first series to
read calibration on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

REPO = "Monash-University/monash_tsf"
REVISION = "refs/convert/parquet"
CONFIG = "australian_electricity_demand"
FREQ = "30min"
ITEM_ID = "T1"

TARGET_NAME = "demand"


@dataclass(frozen=True)
class AusElec:
    """One state's demand history: the target and when each reading happened.

    values      (T,)      half-hourly demand, MW
    timestamps  (T,)      half-hourly DatetimeIndex
    n_train     int       index where the training span ends
    n_val       int       index where the following validation span ends
    """

    values: torch.Tensor
    timestamps: pd.DatetimeIndex
    n_train: int
    n_val: int

    def __len__(self) -> int:
        return len(self.values)

    @property
    def splits(self) -> dict[str, slice]:
        return {
            "train": slice(0, self.n_train),
            "val": slice(self.n_train, self.n_val),
            "test": slice(self.n_val, len(self)),
        }

    def __repr__(self) -> str:
        a, b = self.timestamps[0], self.timestamps[-1]
        return (
            f"AusElec({ITEM_ID}, {len(self)} half-hourly steps, {a:%Y-%m-%d} .. {b:%Y-%m-%d}, "
            f"split {self.n_train}/{self.n_val - self.n_train}/{len(self) - self.n_val})"
        )


STEPS_PER_DAY = 48
TEST_DAYS = 365
VAL_DAYS = 365


def _row(split: str):
    """The `ITEM_ID` row of one split, from the parquet-convert branch."""
    ds = load_dataset(
        REPO, revision=REVISION, data_files={split: f"{CONFIG}/{split}/0000.parquet"}
    )[split]
    return ds[ds["item_id"].index(ITEM_ID)]


def load_ausalia() -> AusElec:
    """`ITEM_ID`'s full demand series, split by calendar time (see module docstring
    point 3): last full year test, the year before that validation, the rest train.
    """
    full = _row("test")
    values = torch.tensor(np.asarray(full["target"], dtype="float32"))
    timestamps = pd.date_range(start=full["start"], periods=len(values), freq=FREQ)

    n_test = TEST_DAYS * STEPS_PER_DAY
    n_val = VAL_DAYS * STEPS_PER_DAY
    n_train = len(values) - n_test - n_val

    return AusElec(
        values=values,
        timestamps=timestamps,
        n_train=n_train,
        n_val=n_train + n_val,
    )
