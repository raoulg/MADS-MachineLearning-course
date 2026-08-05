"""Shared plotting for this lesson's ray.tune search results.

Both 04_search.ipynb and 05_exercises.ipynb build a parallel-coordinates-style view of a
search's trials, and had grown two separate implementations of the same plot -- one
`go.Figure(go.Parcoords(...))` per notebook, each hand-rolling the categorical-to-code
mapping `px.parallel_categories` already does natively. One function instead, matching the
style lesson 5 settled on (`px.parallel_categories`, not `px.parallel_coordinates` as in
`4_tuning_networks/visualize.py`): categorical config values (like `encoder`) plot directly,
no manual code/tickvals bookkeeping, and it reads as one line per trial the same way.
"""

from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _round_sigfigs(series: pd.Series, sig: int) -> pd.Series:
    """Round to `sig` significant figures, not decimal places -- lr (~1e-4) and
    accuracy (~0.6) live on wildly different scales, and a fixed decimal count either
    truncates the small one to zero or leaves the large one noisy. 0 stays 0."""

    def round_one(x: float) -> float:
        if x == 0 or pd.isna(x):
            return x
        digits = sig - int(np.floor(np.log10(abs(x)))) - 1
        return round(x, digits)

    # Series.apply's stub is overloaded on the callable's return type and, for a plain
    # function like round_one, falls back to `DataFrame | Series` -- applying a
    # scalar-in/scalar-out function to a Series always gives a Series back, so this is
    # just narrowing the type checker to what pandas actually does at runtime.
    return cast(pd.Series, series.apply(round_one))


def search_parallel_plot(
    df: pd.DataFrame, columns: list[str], color: str = "accuracy", round_sig: int = 3
) -> go.Figure:
    """One line per trial, every listed column as an axis, coloured by `color`.

    Every numeric column (any int/float dtype, detected automatically -- not just the
    ones a caller thinks to flag) is rounded to `round_sig` significant figures and
    sorted ascending as a category. Without this, a raw `lr` or `accuracy` column plots
    one category per unique float value at full precision -- 24 trials, 24 illegible
    labels stacked on top of each other. Non-numeric columns (like `encoder`) are left
    exactly as they are.

    Args:
        df: a search's results, one row per trial (`results.get_dataframe()`, optionally
            column-renamed).
        columns: which columns to draw as axes. Include `color` in this list too if you
            want it to also appear as its own axis, not just the line colour.
        color: continuous column to colour lines by. Does not need to be in `columns`.
        round_sig: significant figures to round numeric axes to.
    """
    plot_df = df[columns].copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(plot_df[col]):
            plot_df[col] = _round_sigfigs(plot_df[col], round_sig)

    # Colour by the SAME (rounded) series as the matching dimension, not the raw one --
    # otherwise plotly cannot match it to the `color` column already in plot_df and adds
    # a second, redundant "color" dimension instead of reusing that axis.
    color_values = plot_df[color] if color in plot_df.columns else df[color]

    fig = px.parallel_categories(
        plot_df, color=color_values, color_continuous_scale=px.colors.sequential.Viridis
    )
    for dim in fig.data[0].dimensions:
        if dim.label in df.columns and pd.api.types.is_numeric_dtype(df[dim.label]):
            dim.categoryorder = "category ascending"
    return fig
