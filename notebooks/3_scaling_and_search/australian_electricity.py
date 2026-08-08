"""One Australian state's half-hourly electricity demand, loaded once and imported by
`04_probabilistic`.

`Monash-University/monash_tsf`, config `australian_electricity_demand` — five states'
worth of half-hourly demand (MW), 2002 onward, CC-BY-4.0. This module picks one series,
**T1**, and follows exactly the pattern `ett.py` uses for ETTh1: read the loader
docstring there for the rolling-eval-convention rationale (splits are prefixes of one
another, not disjoint, so the honest move is to take the longest row and cut it in time
yourself). The two things worth restating here because they differ from ETT:

**1. No script-based-dataset problem here** — `australian_electricity_demand` already
ships as a plain dataset, so no `refs/convert/parquet` escape hatch is strictly needed,
but this module uses the same revision anyway for consistency with `ett.py` and because
it is the more-tested path in this repo.

**2. There are five series (`T1`-`T5`), not one.** This module always loads `T1`. The
choice is arbitrary among five near-identical demand series — if you want a different
one, change `ITEM_ID` below and nothing else changes shape-wise.

**3. The rolling-eval increment here is 60 steps (30 hours), not months.** ETT's
train/validation/test rows differ by whole months, which is what lets `ett.py` read
clean 12/4/4-month boundaries straight off the row lengths. This dataset's rows differ
by only 60 half-hourly steps between splits — a single short rolling-evaluation window,
not a usable train/val/test boundary. Reusing ETT's trick here would leave 30 hours each
for validation and test, nowhere near enough to window. So this module does **not**
follow `ett.py`'s pattern for splitting: it takes the single longest row (`test`, the
whole series) and cuts it in *calendar* time itself — the last full year is test, the
year before that is validation, everything earlier is train. Simpler than ETT's
convention, and correct for what this dataset's splits actually contain.

D15 (main_principles.md §3.8): replaces ETTh1 as `04_probabilistic`'s primary series.
Not because ETTh1 measured noisier point-to-point — it didn't — but because its
difficulty is regime drift between train/val/test spans, a confound that makes a first
calibration lesson harder to read than it needs to be. This series has 3x tighter CoV
and a clean daily+weekly pattern instead.
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
