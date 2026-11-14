"""Solar power production, ten-minute resolution — the series `03_decomposition_and_nbeats`
and `06_search` use in place of `pedestrian_counts` (main_principles.md D20).

`Monash-University/monash_tsf`, config `solar_10_minutes` — 137 series, each one full
calendar year (2006-01-01 onward, 52,560 ten-minute steps, no gaps in any series checked),
CC-BY-4.0. Verified, not assumed: every series is exactly 365 days at 144 steps/day: 53.6%
of readings are exactly `0.0` (overnight), and the daily FFT peak at 144 steps (24h) is the
single largest peak in the spectrum, well above any weekly or longer component.

**Why this replaced `pedestrian_counts` as the primary demo in `03`.** The original choice
(main_principles.md §3.9) found N-BEATS losing to a flat linear map on pedestrian, at any
window/harmonic/hidden-width configuration tried — a real result, but one that left the
lesson's own N-BEATS-family architecture without a single case in the graded sequence where
it actually won. Direct user feedback, 2026-08-05: a notebook whose entire premise is
"decomposition inside the architecture" should show that premise paying off somewhere, not
just show a linear layer winning twice. Several Monash series were measured looking for a
genuine win (not a data-scarcity artifact — main_principles.md D20/§3.10 for the full
search and the ones that did not work: `nn5_daily`, `traffic_hourly`, `london_smart_meters`,
`tourism_monthly`). Solar is the one that worked, and the reason is physical rather than
statistical: overnight-to-day solar irradiance is close to a deterministic function of the
sun's position, not a noisy human behavioural pattern the way footfall or electricity demand
are. A period-144 Fourier basis is closer to the *actual generating process* here than an
unconstrained `W x H` linear map, and with enough pooled series (20 of 137, ~103,700
training windows — comfortably more than the linear model's own 41,616 parameters, so this
is not the earlier scarcity story repeating) N-BEATS wins outright: generic N-BEATS beats
both linear and the seasonal-naive baseline, confirmed at 8, 12 and 20 epochs across 5 seeds
each (D19's discipline — a single-config check is not a result).

**Two smaller things worth knowing before windowing it:**

1. **Amplitude, not just period, is deterministic.** Night values are not "usually low" —
   they are `0.0` past floating-point noise. A model that learns "near-zero for ~12 of every
   144 steps" gets a large, easy chunk of the loss for free; the harder part is the
   day-to-day variation in the daytime hump's height (cloud cover), which is where trend and
   the residual after the seasonal stack actually earn their keep.
2. **137 series are 137 different sites, not 137 repeats of one series.** Like
   `pedestrian_counts`'s 8 pooled locations, this is real cross-series heterogeneity (site
   latitude, local weather) `06_search`'s MoE routing question can plausibly exploit.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset

REPO = "Monash-University/monash_tsf"
REVISION = "refs/convert/parquet"
CONFIG = "solar_10_minutes"

TARGET_NAME = "solar power"
PERIOD = 144  # steps/day at 10-minute resolution -- verified above, not assumed
STEPS_PER_YEAR = 52560


def load_solar(
    w: int, h: int, n_series: int = 20, min_len_factor: int = 6
) -> list[torch.Tensor]:
    """The first `n_series` (of 137) whose length is at least `min_len_factor * (w + h)`.

    Every series in this config is the same length (one year), so in practice this just
    takes the first `n_series` by `item_id` order -- the `min_len_factor` filter exists for
    parity with `pedestrian_counts`'s loader (some of its 66 series were too short) and as a
    guard if a future revision of the dataset ever mixes lengths.
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
