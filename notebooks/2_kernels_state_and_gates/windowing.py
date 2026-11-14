"""Windowing and calendar helpers, shared by every notebook that cuts a series into windows.

Two functions, both deliberately small. Everything else — splitting, scaling, choosing
`W` and `H` — stays in the notebooks, because that is the lesson rather than the plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def window_index(n: int, size: int, stride: int = 1) -> torch.Tensor:
    """(n_windows, size) index matrix — the broadcast trick derived in `01_windowing`.

    `stride` exists because consecutive windows overlap by everything but one step: at
    W=96 two neighbours share 95/96 of their content, so striding by 24 costs almost no
    information and shrinks an epoch 24x.
    """
    starts = torch.arange(0, n - size + 1, stride).reshape(-1, 1)
    return torch.arange(size).reshape(1, -1) + starts


def calendar_features(timestamps: pd.DatetimeIndex) -> torch.Tensor:
    """(T, 4) — hour, day-of-week, day-of-month and day-of-year, each scaled to [-0.5, 0.5].

    The four time features the transformers-library forecasters expect in
    `past_time_features` / `future_time_features`. Scaling rather than one-hot keeps the
    tensor narrow; a *cyclical* encoding (sin/cos of the same quantities) is a design
    choice you get to make yourself.
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
