"""Features for a timeseries window that you can read, name, and argue about.

The counterpart of `5_attention/legal_patterns.py`. There, sixty regexes encoded what
somebody who has read a lot of Supreme Court opinions knows about them. Here, thirty
numbers encode what somebody who has stared at a lot of sensor data knows to look at
first: *how strong is the daily cycle, is it trending, how noisy is it, where in the
week are we, what were the loads doing.* Every one has a name and a reason, and any
of them can be wrong in a way you can point at — which is the whole difference
between a designed representation and a learned one.

Two granularities, because 04_composition needs both:

- `window_features(x)` → `(N, F)`, one vector per window. Fuse this *after* the
  learned branch has been reduced to a vector: `Concatenate2D`.
- `patch_features(x)` → `(N, C, Fp)`, one vector per patch of the window, with the
  time axis still intact. Fuse this *next to* the encoder's own per-patch states:
  `Concatenate3D`.

The alignment contract from `06_composition` applies unchanged and is just as easy to
break: patch *i* of the feature matrix describes patch *i* of the window only if both
were cut with the same `patch_len`. Nothing checks that for you — `torch.cat` sees two
tensors of matching rank and is happy.

**A caveat that is really a lesson.** A window of length W can only resolve periods up
to W. At W=96 (four days) the weekly cycle is invisible: it is not weak in the
spectrum, it is *absent from the axis*. `spectral_bands` names its widest band
`p>=36h` rather than pretending to a weekly number it cannot compute. If you want
weekly structure in the features, the fix is a longer window, not a cleverer feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor

#: period ranges in steps (hours, for ETTh1), as (name, low, high). A band contributes
#: the fraction of the window's spectral energy sitting at periods in [low, high).
SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("p<6h", 2.0, 6.0),
    ("p6-10h", 6.0, 10.0),
    ("p10-16h", 10.0, 16.0),
    ("p16-36h", 16.0, 36.0),  # the daily cycle lives here
    ("p>=36h", 36.0, float("inf")),
)

#: lags the autocorrelation is read at: one step back, one day back, two days back
ACF_LAGS: tuple[int, ...] = (1, 24, 48)

EPS = 1e-8


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def spectral_bands(x: Tensor) -> tuple[Tensor, list[str]]:
    """Energy per period band, plus spectral entropy and the dominant period.

    Mean-removed and Hann-tapered before the FFT (a rectangular window smears a
    strong low-frequency component across every bin — the standard leakage artefact,
    and the reason a raw `rfft` of a trending series looks like noise at every scale).
    Band energies are normalised to fractions, so a hot week and a cold week with the
    same *shape* get the same features.
    """
    n = x.shape[-1]
    taper = torch.hann_window(n, periodic=False, dtype=x.dtype, device=x.device)
    z = (x - x.mean(-1, keepdim=True)) * taper
    power = torch.fft.rfft(z, dim=-1).abs() ** 2
    power[..., 0] = 0.0  # DC is the mean we just removed

    k = torch.arange(power.shape[-1], device=x.device, dtype=x.dtype)
    periods = torch.where(k > 0, n / k.clamp(min=1), torch.full_like(k, float("inf")))
    total = power.sum(-1, keepdim=True).clamp(min=EPS)

    feats, names = [], []
    for name, lo, hi in SPECTRAL_BANDS:
        mask = (periods >= lo) & (periods < hi)
        feats.append((power * mask).sum(-1) / total.squeeze(-1))
        names.append(f"band:{name}")

    p = power / total
    entropy = -(p * torch.log(p.clamp(min=EPS))).sum(-1) / np.log(power.shape[-1])
    feats.append(entropy)
    names.append("spectral_entropy")

    dominant = periods[power.argmax(-1)]
    feats.append(torch.log1p(dominant.clamp(max=float(n))))
    names.append("log_dominant_period")

    feats.append(torch.log1p(total.squeeze(-1)))
    names.append("log_total_energy")

    return torch.stack(feats, dim=-1), names


def trend(x: Tensor) -> tuple[Tensor, list[str]]:
    """Least-squares slope over the window, and how much of the variance it explains.

    Two numbers, and they say different things: the slope is *how fast*, R^2 is
    *how much of what happened is that*. A steep slope with R^2 near zero is one
    spike, not a trend.
    """
    n = x.shape[-1]
    t = torch.linspace(-1.0, 1.0, n, dtype=x.dtype, device=x.device)
    tc = t - t.mean()
    xc = x - x.mean(-1, keepdim=True)
    slope = (xc * tc).sum(-1) / (tc * tc).sum().clamp(min=EPS)
    resid = xc - slope.unsqueeze(-1) * tc
    ss_tot = (xc**2).sum(-1).clamp(min=EPS)
    r2 = 1.0 - (resid**2).sum(-1) / ss_tot
    return torch.stack([slope, r2], dim=-1), ["trend_slope", "trend_r2"]


def level(x: Tensor) -> tuple[Tensor, list[str]]:
    """Where the series is and how far it swings — including the one number a naive
    forecaster uses on its own, `last`, and the one a mean-reverting forecaster uses,
    `last - mean`."""
    mean = x.mean(-1)
    last = x[..., -1]
    return (
        torch.stack(
            [mean, x.std(-1), x.amin(-1), x.amax(-1), last, last - mean],
            dim=-1,
        ),
        ["mean", "std", "min", "max", "last", "last_minus_mean"],
    )


def autocorrelation(
    x: Tensor, lags: tuple[int, ...] = ACF_LAGS
) -> tuple[Tensor, list[str]]:
    """Correlation of the window with itself, shifted. `acf_24` on hourly data is
    "how much does this hour look like the same hour yesterday" — the single number
    that says whether a seasonal-naive baseline will be hard to beat."""
    n = x.shape[-1]
    xc = x - x.mean(-1, keepdim=True)
    denom = (xc**2).sum(-1).clamp(min=EPS)
    feats, names = [], []
    for lag in lags:
        if lag >= n:  # not resolvable in this window; say so with a zero, do not invent
            feats.append(torch.zeros_like(denom))
        else:
            feats.append((xc[..., lag:] * xc[..., :-lag]).sum(-1) / denom)
        names.append(f"acf_{lag}")
    return torch.stack(feats, dim=-1), names


def calendar(origins: pd.DatetimeIndex) -> tuple[Tensor, list[str]]:
    """Where in the day and the week the *forecast origin* sits, encoded cyclically.

    sin/cos rather than a scaled integer, because 23:00 and 00:00 are one hour apart
    and `hour / 23` puts them at opposite ends. This is a design decision with a
    consequence you can measure — E3 asks you to.
    """
    origins_s = pd.Series(origins)
    hour = torch.tensor(origins_s.dt.hour.to_numpy(), dtype=torch.float32)
    dow = torch.tensor(origins_s.dt.dayofweek.to_numpy(), dtype=torch.float32)
    two_pi = 2 * np.pi
    return (
        torch.stack(
            [
                torch.sin(two_pi * hour / 24),
                torch.cos(two_pi * hour / 24),
                torch.sin(two_pi * dow / 7),
                torch.cos(two_pi * dow / 7),
                (dow >= 5).float(),
            ],
            dim=-1,
        ),
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
    )


# ---------------------------------------------------------------------------
# the two granularities
# ---------------------------------------------------------------------------
def window_features(
    x: Tensor,
    origins: pd.DatetimeIndex | None = None,
    covariates: Tensor | None = None,
    covariate_names: tuple[str, ...] = (),
) -> tuple[Tensor, list[str]]:
    """(N, W) -> (N, F), one readable vector per window.

    `origins[i]` is the timestamp of the last observed step of window *i* — the moment
    the forecast is made from. `covariates` is (N, W, C); only its last row is used,
    on the argument that a load reading from four days ago tells you less than the
    current one and costs a feature either way.
    """
    parts, names = [], []
    for fn in (spectral_bands, trend, level, autocorrelation):
        f, n = fn(x)
        parts.append(f)
        names += n
    if origins is not None:
        f, n = calendar(origins)
        parts.append(f)
        names += n
    if covariates is not None:
        parts.append(covariates[:, -1, :])
        names += [f"cov_last:{c}" for c in covariate_names] or [
            f"cov_last_{i}" for i in range(covariates.shape[-1])
        ]
    return torch.cat(parts, dim=-1), names


def patch_features(x: Tensor, patch_len: int = 24) -> tuple[Tensor, list[str]]:
    """(N, W) -> (N, C, Fp), one vector per patch, time axis intact.

    W must be divisible by `patch_len`. The per-patch set is deliberately smaller than
    the per-window one: a 24-step patch cannot support five spectral bands, and asking
    it to would produce five numbers that are mostly the same number.
    """
    n, w = x.shape
    if w % patch_len:
        raise ValueError(f"window {w} is not divisible by patch_len {patch_len}")
    patches = x.reshape(n, w // patch_len, patch_len)

    lvl, lvl_names = level(patches)
    trd, trd_names = trend(patches)
    acf, acf_names = autocorrelation(patches, lags=(1,))
    spec, spec_names = spectral_bands(patches)
    keep = [
        i
        for i, name in enumerate(spec_names)
        if name in ("spectral_entropy", "log_total_energy")
    ]
    spec, spec_names = spec[..., keep], [spec_names[i] for i in keep]

    return (
        torch.cat([lvl, trd, acf, spec], dim=-1),
        lvl_names + trd_names + acf_names + spec_names,
    )
