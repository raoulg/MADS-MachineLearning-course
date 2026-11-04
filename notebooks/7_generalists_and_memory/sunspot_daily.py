"""Daily sunspot counts, windowed for forecasting -- the same series `03_decomposition_
and_nbeats` §3.7 uses at yearly resolution, at the resolution it was actually recorded.

`Monash-University/monash_tsf`, config `sunspot` — daily Wolf/Zurich sunspot number,
1818 onward, CC-BY-4.0. Two things worth knowing before windowing it, both checked
against the data rather than assumed:

**1. The record has real gaps, but only early on.** 3240 of the first ~11,300 days
(roughly 1818-1848) are missing — 19th-century observation was not continuous. Every
day from **1848-12-22 onward is complete**: 62,617 consecutive daily readings, 171.4
years. This module starts there; the earlier, gappy stretch is what
`03_decomposition_and_nbeats` §3.7 uses instead, because that section aggregates to
yearly means (a few missing days inside a year barely move an annual mean) rather than
windowing day by day (where a gap breaks a window outright).

**2. There is a real, physical ~27-day quasi-period, distinct from any calendar
season.** Verified by autocorrelation and FFT on the clean tail, not assumed: ACF has a
local maximum at lag 26-28 days (0.77, above both its lag-14 and lag-60 neighbours,
0.68 and 0.70), and the spectrum's short-period peaks cluster at 26.8-28.2 days. This
is the Sun's ~27-day synodic rotation period (as seen from Earth) — active regions
rotate into and out of view on roughly that schedule. `PERIOD = 27` below names it, the
way `ett.py`'s notebooks name 24 for a calendar day; nothing here assumes it, the
autocorrelation check is what a notebook using this module should show first.

Day-to-day persistence is still much stronger than the 27-day echo (ACF at lag 1 is
0.96 against 0.77 at lag 27) — closer to ETTh1's "naive beats seasonal-naive" shape
than to pedestrian's crisp daily cycle, which is exactly why this series replaces
ETTh1 as the windowing/baseline spine (main_principles.md §3.8): same pedagogical role
(a series where the obvious "season" is not the strong baseline), one dataset instead
of an unfamiliar six-covariate one, and a period a notebook can actually go find rather
than assume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

REPO = "Monash-University/monash_tsf"
REVISION = "refs/convert/parquet"
CONFIG = "sunspot"
FREQ = "D"

TARGET_NAME = "sunspot number"
PERIOD = 27  # ~solar synodic rotation, verified above -- not a calendar season
CLEAN_FROM = 11307  # first index with no NaN remaining ahead of it (1848-12-22)

TEST_YEARS = 10
VAL_YEARS = 10
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Sunspot:
    """The clean tail of the daily series: no missing days.

    values      (T,)      daily sunspot number
    timestamps  (T,)      daily DatetimeIndex
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
            f"Sunspot({len(self)} daily steps, {a:%Y-%m-%d} .. {b:%Y-%m-%d}, "
            f"split {self.n_train}/{self.n_val - self.n_train}/{len(self) - self.n_val})"
        )


def load_sunspot_daily() -> Sunspot:
    """The clean (post-1848) daily tail, split by calendar time: last 10 years test,
    the 10 years before that validation, the rest (151 years) train.
    """
    ds = load_dataset(
        REPO, revision=REVISION, data_files={"test": f"{CONFIG}/test/0000.parquet"}
    )["test"][0]
    values = np.asarray(ds["target"], dtype="float32")
    timestamps = pd.date_range(start=ds["start"], periods=len(values), freq=FREQ)

    values = values[CLEAN_FROM:]
    timestamps = timestamps[CLEAN_FROM:]
    assert not np.isnan(values).any(), "expected no missing days in the clean tail"

    n_test = int(TEST_YEARS * DAYS_PER_YEAR)
    n_val = int(VAL_YEARS * DAYS_PER_YEAR)
    n_train = len(values) - n_test - n_val

    return Sunspot(
        values=torch.tensor(values),
        timestamps=timestamps,
        n_train=n_train,
        n_val=n_train + n_val,
    )
