"""Solar power production, ten-minute resolution — the series `2_kernels_state_and_gates/03_decomposition_and_nbeats`
uses.

`Monash-University/monash_tsf`, config `solar_10_minutes` — 137 series, each one full
calendar year (2006-01-01 onward, 52,560 ten-minute steps, no gaps in any series checked),
CC-BY-4.0. Verified, not assumed: every series is exactly 365 days at 144 steps/day: 53.6%
of readings are exactly `0.0` (overnight), and the daily FFT peak at 144 steps (24h) is the
single largest peak in the spectrum, well above any weekly or longer component.

**Why this series.** Solar power has a period that is close to a deterministic function
of the sun's position: production follows the daily arc and sits at zero every night. A
period-144 Fourier basis is close to that generating process, so this is a series on which a
seasonal basis has something real to find. What it buys, measured against linear, generic
N-BEATS and the seasonal-naive baseline, is in the notebook; this module makes no claim
about which model wins.

**Two smaller things worth knowing before windowing it:**

1. **Night is exactly zero.** Night values are `0.0`, not "usually low". A model that
   learns "zero for roughly half of every 144 steps" gets a large, easy part of the loss
   for free; the harder part is the day-to-day height of the daytime hump (cloud cover).
2. **137 series are 137 different sites, not 137 repeats of one series**: real
   cross-series differences in site and local weather.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset

REPO = "Monash-University/monash_tsf"
REVISION = "refs/convert/parquet"
CONFIG = "solar_10_minutes"

TARGET_NAME = "solar power"
PERIOD = 144  # steps/day at 10-minute resolution, checked above
STEPS_PER_YEAR = 52560


def load_solar(
    w: int, h: int, n_series: int = 20, min_len_factor: int = 6
) -> list[torch.Tensor]:
    """The first `n_series` (of 137) whose length is at least `min_len_factor * (w + h)`.

    Every series in this config is the same length (one year), so in practice this just
    takes the first `n_series` by `item_id` order; the `min_len_factor` filter guards against
    a revision of the dataset that mixes lengths.
    """
    ds = load_dataset(
        REPO, revision=REVISION, data_files={"test": f"{CONFIG}/test/0000.parquet"}
    )["test"]
    out = []
    for row in ds:
        s = np.asarray(row["target"], dtype="float32")
        s = s[~np.isnan(s)]
        if len(s) >= min_len_factor * (w + h):
            out.append(torch.tensor(s))
        if len(out) >= n_series:
            break
    return out
