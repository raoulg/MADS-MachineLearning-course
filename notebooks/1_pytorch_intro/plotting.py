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


# ---------------------------------------------------------------------------------------
# 02_nonlinearity: how a ReLU network builds a curve, and what each activation looks like
# ---------------------------------------------------------------------------------------

from collections.abc import Callable, Sequence
from itertools import pairwise

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

UNIT_COLORS = ["#c8724f", "#5fb8b0", "#3b4a59", "#8a6fb3", "#b39b2f", "#4f7fc8"]


def plot_relu_buildup(
    x: torch.Tensor,
    lines: torch.Tensor,
    hinges: torch.Tensor,
    weighted: torch.Tensor,
    output: torch.Tensor,
) -> Figure:
    """Draw how a one-hidden-layer ReLU network builds its output, one hidden unit per
    column.

    `lines`, `hinges` and `weighted` have shape `(len(x), units)`: each unit's line
    before ReLU, after ReLU, and after scaling by its output weight. They are drawn as
    rows 1-3. `output` is the network's output (row-sum of `weighted` plus the output
    bias), drawn wide at the bottom. Shading marks every other stretch between two
    kinks; inside one stretch the output is a single straight line.
    """
    x = x.flatten()
    n = lines.shape[1]
    lim = 1.1 * float(torch.cat([lines, weighted]).abs().max())

    fig = plt.figure(figsize=(3.4 * n, 10))
    grid = fig.add_gridspec(4, n, height_ratios=[1, 1, 1, 1.3], hspace=0.6, wspace=0.25)
    rows = [
        ("line", r"$\theta_{{{i}0}} + \theta_{{{i}1}}x$", lines),
        (
            "after ReLU",
            r"$h_{i} = \mathrm{{ReLU}}[\theta_{{{i}0}} + \theta_{{{i}1}}x]$",
            hinges,
        ),
        ("weighted", r"$\phi_{i} h_{i}$", weighted),
    ]
    for r, (ylabel, title, curves) in enumerate(rows):
        for i in range(n):
            ax = fig.add_subplot(grid[r, i])
            ax.plot(x, curves[:, i], color=UNIT_COLORS[i % len(UNIT_COLORS)], lw=2.5)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(title.format(i=i + 1), fontsize=10)
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(-lim, lim)
            if i == 0:
                ax.set_ylabel(ylabel)

    # a kink sits where a unit's line crosses zero
    crossings = (torch.sign(lines[1:]) != torch.sign(lines[:-1])).any(dim=1)
    kinks = x[1:][crossings].tolist()
    lo, hi = float(x.min()), float(x.max())
    edges = [lo, *kinks, hi]

    ax = fig.add_subplot(grid[3, :])
    for k, (left, right) in enumerate(pairwise(edges)):
        if k % 2:
            ax.axvspan(left, right, color="lightgray", alpha=0.5, lw=0)
    ax.plot(x, output.flatten(), color="k", lw=2.5)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlim(lo, hi)
    terms = " + ".join(rf"\phi_{i + 1} h_{i + 1}" for i in range(n))
    ax.set_title(
        rf"output $y = \phi_0 + {terms}$: {len(edges) - 1} straight pieces", fontsize=11
    )
    ax.set_xlabel("input x")
    return fig


def plot_relu_sums(
    x: torch.Tensor,
    pieces: Sequence[torch.Tensor],
    totals: Sequence[torch.Tensor],
    target: torch.Tensor,
    titles: Sequence[str],
) -> Figure:
    """One column per network: its weighted hidden units on top, their sum against
    `target` below.

    `pieces[k]` has shape `(len(x), units)`: one column per hidden unit, already multiplied
    by its output weight. `totals[k]` is that network's output, the row-sum of `pieces[k]`
    plus the output bias.
    """
    x = x.flatten()
    n = len(pieces)
    fig, axes = plt.subplots(
        2, n, figsize=(3.6 * n, 6), sharex=True, sharey="row", squeeze=False
    )
    for col, (piece, total, title) in enumerate(zip(pieces, totals, titles)):
        top, bottom = axes[0, col], axes[1, col]
        for u in range(piece.shape[1]):
            top.plot(
                x,
                piece[:, u],
                lw=1.2,
                color=UNIT_COLORS[u % len(UNIT_COLORS)],
                alpha=0.85,
            )
        top.axhline(0, color="gray", lw=0.6, ls=":")
        top.set_title(title, fontsize=10)
        bottom.plot(x, target.flatten(), color="k", lw=1.5, ls="--", label="target")
        bottom.plot(x, total.flatten(), color="tab:red", lw=2, label="sum of the units")
        bottom.set_xlabel("input x")
    axes[0, 0].set_ylabel("each unit, weighted")
    axes[1, 0].set_ylabel("their sum + bias")
    axes[1, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_activation_fits(
    acts: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    fits: dict[str, tuple[torch.Tensor, float]],
    x: torch.Tensor,
    y: torch.Tensor,
) -> Figure:
    """One column per activation: the function and its slope on top, the sine fit made
    with it below.

    The slope comes from autograd, so it is exactly the factor a gradient flowing back
    through this activation gets multiplied by. `fits[name]` is `(prediction, mse)` for the
    model that used activation `name`; `x, y` are the training data.
    """
    n = len(acts)
    z = torch.linspace(-4, 4, 200)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4), sharey="row", squeeze=False)
    for col, (name, fn) in enumerate(acts.items()):
        top, bottom = axes[0, col], axes[1, col]
        zz = z.clone().requires_grad_(True)
        out = fn(zz)
        (slope,) = torch.autograd.grad(out.sum(), zz)
        top.plot(z, out.detach(), color="tab:blue", lw=2, label="f(z)")
        top.plot(z, slope, color="tab:orange", lw=1.5, ls="--", label="slope f'(z)")
        top.axhline(0, color="k", lw=0.5)
        top.axvline(0, color="k", lw=0.5)
        top.set_ylim(-1.5, 3)
        top.set_title(name)

        prediction, mse = fits[name]
        bottom.scatter(x.flatten(), y.flatten(), color="gray", s=4, alpha=0.4)
        bottom.plot(x.flatten(), prediction.flatten(), color="tab:red", lw=2)
        bottom.set_title(f"sine fit, MSE {mse:.3f}", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_stacking(
    vectors: list,
    labels: list[str],
    kind: str = "batch",
    source: str | None = None,
    max_dims: int = 48,
    title: str | None = None,
    wrap_width: int = 48,
):
    """Draw `torch.stack`: separate 1D vectors on the left, the 2D result on the right.

    The same drawing serves both meanings of a `(N, D)` tensor, and the left-hand side
    is where they differ:

    - `kind="batch"`: every vector is its own item, so every label gets its own box.
      The stacked result is `(B, D)`.
    - `kind="chunks"`: every vector is a piece of one item, so the labels sit together
      inside a single box named `source`. The stacked result is `(C, D)`.

    Args:
        vectors: N tensors (or arrays) of shape `(D,)`.
        labels: one short text per vector, shown left of its strip.
        kind: "batch" or "chunks".
        source: name of the single item the chunks came from (used when kind="chunks").
        max_dims: how many of the D numbers to draw; a 384-wide strip is unreadable.
        title: figure title.
        wrap_width: characters per line for the labels.
    """
    import textwrap

    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import FancyBboxPatch

    if kind not in ("batch", "chunks"):
        raise ValueError(f"kind must be 'batch' or 'chunks', got {kind!r}")
    stacked = torch.stack([torch.as_tensor(v) for v in vectors]).float()
    n, d = stacked.shape
    shown = stacked[:, :max_dims].numpy()
    vmin, vmax = float(shown.min()), float(shown.max())
    row = "item" if kind == "batch" else "chunk"
    letter = "B" if kind == "batch" else "C"
    colors = plt.cm.tab10(np.arange(n) % 10)

    fig = plt.figure(figsize=(13, 0.75 * n + 1.5))
    grid = fig.add_gridspec(
        n, 4, width_ratios=[3.6, 2.4, 0.9, 2.6], wspace=0.08, hspace=0.6
    )

    text_ax = fig.add_subplot(grid[:, 0])
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(n, 0)
    text_ax.axis("off")
    for i, label in enumerate(labels):
        box = None
        if kind == "batch":
            box = {
                "boxstyle": "round,pad=0.3",
                "facecolor": colors[i],
                "alpha": 0.18,
                "edgecolor": colors[i],
            }
        wrapped = textwrap.fill(label, width=wrap_width)
        text_ax.text(
            0.07, i + 0.5, wrapped, va="center", ha="left", fontsize=8.5, bbox=box
        )
        text_ax.plot(0.02, i + 0.5, "s", color=colors[i], markersize=8)
    if kind == "chunks":
        text_ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.08),
                1.0,
                n - 0.16,
                boxstyle="round,pad=0.02",
                fill=False,
                edgecolor="black",
                linewidth=1.4,
                clip_on=False,
            )
        )
    header = (
        f"{n} separate items" if kind == "batch" else f"1 item, cut into {n} pieces"
    )
    if kind == "chunks" and source:
        header = f"{source}, cut into {n} pieces"
    text_ax.set_title(header, loc="left", fontsize=10)

    for i in range(n):
        ax = fig.add_subplot(grid[i, 1])
        ax.imshow(shown[i : i + 1], cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(2)
        if i == 0:
            ax.set_title(f"{n} vectors, each ({d},)", fontsize=10)

    arrow_ax = fig.add_subplot(grid[:, 2])
    arrow_ax.axis("off")
    arrow_ax.annotate(
        "",
        xy=(0.95, 0.5),
        xytext=(0.05, 0.5),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "black"},
    )
    arrow_ax.text(
        0.5,
        0.56,
        "torch.stack",
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
    )

    mat_ax = fig.add_subplot(grid[:, 3])
    mat_ax.imshow(shown, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    mat_ax.set_yticks(range(n))
    mat_ax.set_yticklabels([f"{row} {i}" for i in range(n)])
    for i, tick in enumerate(mat_ax.get_yticklabels()):
        tick.set_color(colors[i])
        tick.set_fontweight("bold")
    mat_ax.yaxis.tick_right()
    mat_ax.set_xticks([])
    mat_ax.set_xlabel(
        f"first {min(max_dims, d)} of D={d} numbers" if d > max_dims else f"D={d}"
    )
    mat_ax.set_title(f"one tensor, ({letter}, D) = {tuple(stacked.shape)}", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    return fig


def plot_optimizer_paths(
    loss_fn, paths: dict, ax=None, title: str = "", lim: float = 4.5
):
    """Contour plot of a 2-parameter loss, with one optimizer trajectory per entry.

    Args:
        loss_fn: takes a `(2,)` tensor and returns a scalar tensor.
        paths: name -> `(steps, 2)` tensor of the weights after every step.
        ax: matplotlib axis to draw on; a new figure is made when omitted.
        title: axis title.
        lim: the plot shows [-lim, lim] on both axes.
    """
    import matplotlib.pyplot as plt
    import torch

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    grid = torch.linspace(-lim, lim, 200)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    zz = torch.stack(
        [loss_fn(torch.stack([x, y])) for x, y in zip(xx.flatten(), yy.flatten())]
    )
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=30, cmap="Greys", alpha=0.55)
    ax.contour(xx, yy, zz, levels=15, colors="grey", linewidths=0.4)
    for (name, path), color in zip(paths.items(), plt.get_cmap("tab10")(range(10))):
        p = path.detach().numpy()
        ax.plot(
            p[:, 0],
            p[:, 1],
            "-o",
            color=color,
            markersize=2.5,
            linewidth=1.4,
            label=f"{name}  (loss {float(loss_fn(path[-1])):.3g})",
        )
    ax.plot(*path[0].detach().numpy(), "k*", markersize=12)
    ax.plot(0, 0, "x", color="black", markersize=10, markeredgewidth=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("weight 1")
    ax.set_ylabel("weight 2")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return ax


UNIT_AXIS = "#e07a10"  # the colour of an axis of length 1, in every panel


def _stack(ax, planes, dx, dy, w, h, cmap=None, vmin=None, vmax=None):
    """Draw 2D planes as a stack of cards, the first one in front, the rest up and to the right."""
    from matplotlib.patches import Rectangle

    n = len(planes)
    for i in reversed(range(n)):
        x0, y0 = i * dx, i * dy
        ax.imshow(
            planes[i],
            extent=(x0, x0 + w, y0, y0 + h),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            zorder=n - i,
        )
        ax.add_patch(
            Rectangle(
                (x0, y0), w, h, fill=False, ec="black", lw=0.8, zorder=n - i + 0.5
            )
        )


def _bracket(ax, start, end, text, color, side=(0.12, -0.12)):
    """A line from `start` to `end` with end ticks, labelled `text` beside its middle."""
    import numpy as np

    (x0, y0), (x1, y1) = start, end
    ax.plot([x0, x1], [y0, y1], color=color, lw=2.5, solid_capstyle="butt", zorder=50)
    for x, y in (start, end):
        ax.plot(
            [x - side[0] / 2, x + side[0] / 2],
            [y - side[1] / 2, y + side[1] / 2],
            color=color,
            lw=2,
            zorder=50,
        )
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + 1.6 * np.array(side)
    ax.text(
        *mid,
        text,
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=51,
    )


def _panel(ax, title, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")


def plot_unit_axis(gray, rgb, chunked, batch, max_dims: int = 12):
    """Draw an axis of length 1 as a stack of one card.

    A stack axis is drawn as cards behind each other, with a bracket along the depth
    labelled with its length; the bracket is orange when that length is 1.

    Top row: a grey image as a flat `(H, W)` grid; the same numbers as a stack of one
    channel, `(H, W, 1)`; and the colour image, a stack of three, `(H, W, 3)`.
    Bottom row: one chunked item `(C, D)` and the same item as a batch of one,
    `(1, C, D)`; a batch `(B, D)` and the same rows as B items of one chunk each,
    `(B, 1, D)`.

    Args:
        gray: `(H, W)` grey image.
        rgb: `(H, W, 3)` colour image, uint8.
        chunked: `(C, D)` chunk vectors of one item.
        batch: `(B, D)` vectors of B items.
        max_dims: how many of the D numbers to draw.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    gray = np.asarray(gray, dtype=float)
    rgb = np.asarray(rgb)
    H, W = gray.shape
    w, h = 4.0, 4.0 * H / W
    dx, dy = 0.55, 0.45
    channels = []
    for k in range(3):
        only = np.zeros_like(rgb)
        only[:, :, k] = rgb[:, :, k]
        channels.append(only)

    fig = plt.figure(figsize=(16, 6.5))
    top_fig, bottom_fig = np.asarray(
        fig.subfigures(2, 1, height_ratios=[1.2, 1])
    ).ravel()
    top = top_fig.subplots(1, 3)
    bottom = bottom_fig.subplots(1, 4)
    image_box = ((-0.2, w + 3 * dx + 0.9), (-0.3, h + 3 * dy + 0.3))

    # top row: the image
    ax = top[0]
    ax.imshow(gray, cmap="gray", extent=(0, w, 0, h), aspect="auto")
    _panel(ax, f"(H, W) = ({H}, {W})\na grey image: one number per pixel", *image_box)

    ax = top[1]
    _stack(ax, [gray], dx, dy, w, h, cmap="gray")
    _bracket(ax, (w + 0.1, 0.0), (w + 0.1 + dx, dy), "1", UNIT_AXIS)
    _panel(
        ax, f"(H, W, 1) = ({H}, {W}, 1)\nthe same numbers, a stack of one", *image_box
    )

    ax = top[2]
    _stack(ax, channels, dx, dy, w, h)
    _bracket(ax, (w + 0.1, 0.0), (w + 0.1 + 3 * dx, 3 * dy), "3", "black")
    _panel(
        ax,
        f"(H, W, 3) = {tuple(rgb.shape)}\nthe colour image, a stack of three",
        *image_box,
    )

    # bottom row: the ladder
    chunked = np.asarray(chunked, dtype=float)[:, :max_dims]
    batch = np.asarray(batch, dtype=float)[:, :max_dims]
    vmin = float(min(chunked.min(), batch.min()))
    vmax = float(max(chunked.max(), batch.max()))
    C, B = chunked.shape[0], batch.shape[0]
    cell = 0.4
    gw = max_dims * cell
    dx, dy = 0.35, 0.7
    grid_box = (
        (-1.6, gw + B * dx + 1.4),
        (-0.2, max(C * cell + dy, cell + B * dy) + 0.2),
    )
    style = {"cmap": "viridis", "vmin": vmin, "vmax": vmax}

    ax = bottom[0]
    ax.imshow(chunked, extent=(0, gw, 0, C * cell), aspect="auto", **style)
    _panel(ax, f"(C, D)\none review, {C} chunks", *grid_box)

    ax = bottom[1]
    _stack(ax, [chunked], dx, dy, gw, C * cell, **style)
    _bracket(ax, (gw + 0.1, 0.0), (gw + 0.1 + dx, dy), "B = 1", UNIT_AXIS)
    _panel(ax, "(1, C, D)\na batch of one review", *grid_box)

    ax = bottom[2]
    ax.imshow(batch, extent=(0, gw, 0, B * cell), aspect="auto", **style)
    _panel(ax, f"(B, D)\n{B} reviews, one row each", *grid_box)

    ax = bottom[3]
    _stack(ax, [batch[i : i + 1] for i in range(B)], dx, dy, gw, cell, **style)
    ax.plot([-0.15, -0.15], [0, cell], color=UNIT_AXIS, lw=2.5)
    ax.text(
        -0.3,
        cell / 2,
        "C = 1",
        color=UNIT_AXIS,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="center",
    )
    _bracket(ax, (gw + 0.1, 0.0), (gw + 0.1 + B * dx, B * dy), f"B = {B}", "black")
    _panel(ax, f"(B, 1, D)\n{B} reviews of one chunk each", *grid_box)

    fig.text(
        0.01,
        -0.01,
        "orange: an axis of length 1. Adding or removing it moves no number; "
        "the address gets one more index, and that index can only be 0.",
        color=UNIT_AXIS,
        fontsize=12,
    )
    return fig
