"""Architecture diagrams and search-space pictures, drawn with matplotlib only.

Two ideas:

- `draw_architecture` draws a model as boxes and arrows, left to right, with the shape of
  the tensor written on every arrow. A stage is either one `Block` or a list of branches
  (each branch a list of `Block`s) that run side by side, which is how a model with two
  inputs is drawn.
- `trace` builds those blocks from a real model by pushing a tensor through it and
  recording every layer it passes, so the shapes on the arrows are the shapes the model
  actually produces, not the ones we meant it to produce.

`plot_search_space` draws a ray.tune `param_space` before any trial runs: one panel per
hyperparameter, filled with samples drawn from that axis's own distribution.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from torch import nn

# fill colour per kind of block; the legend in `draw_architectures` explains them
COLOURS = {
    "data": "#e7e7e7",  # a cache or other input: nothing is learned here
    "noise": "#fde9b8",  # augmentation, active only while training
    "hidden": "#f6c7a8",  # a hidden layer: what n_hidden_layers / hidden_dim add
    "proj": "#f3d9e6",  # a Linear that only changes a branch's width, so branches can be joined
    "layer": "#c9dcf2",  # the output Linear every head ends in
    "join": "#dccdf0",  # where two branches become one tensor
    "loss": "#cfe8cf",  # the loss, compared against the labels
}
LEGEND = {
    "data": "input vectors (cache)",
    "noise": "noise, training only",
    "hidden": "hidden layer",
    "proj": "projection",
    "layer": "output layer",
    "join": "join of two branches",
    "loss": "loss",
}

BOX_H = 0.9  # box height, in data units
BRANCH_DY = 1.5  # vertical distance between parallel branches
CHAR_W = 0.115  # width of one character of box text, in data units
SHAPE_CHAR_W = 0.105


@dataclass(frozen=True)
class Block:
    """One box: `label` inside it, `out` (the outgoing tensor's shape) on the arrow after it."""

    label: str
    out: str = ""
    kind: str = "layer"


Stage = Block | Sequence[Sequence[Block]]


def _box_width(label: str) -> float:
    return max(len(line) for line in label.split("\n")) * CHAR_W + 0.5


def _gap(*shapes: str) -> float:
    return max([1.3, *(len(s) * SHAPE_CHAR_W + 0.5 for s in shapes)])


def _arrow(
    ax: Axes, start: tuple[float, float], end: tuple[float, float], shape: str
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#444"
        )
    )
    if shape:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + 0.12
        ax.text(
            x,
            y,
            shape,
            ha="center",
            va="bottom",
            fontsize=8.5,
            family="monospace",
            color="#222",
        )


def _box(ax: Axes, x: float, y: float, block: Block) -> float:
    w = _box_width(block.label)
    ax.add_patch(
        FancyBboxPatch(
            (x, y - BOX_H / 2),
            w,
            BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=COLOURS.get(block.kind, "white"),
            ec="#555",
            lw=1,
        )
    )
    ax.text(x + w / 2, y, block.label, ha="center", va="center", fontsize=9)
    return w


def draw_architecture(
    stages: Sequence[Stage], ax: Axes | None = None, title: str | None = None
) -> Axes:
    """Draw `stages` left to right on `ax` (a new figure when None) and return the axes.

    Each arrow carries the shape of the tensor that travels along it: the `out` of the
    block it leaves. A stage that is a list of branches is drawn as parallel rows; the
    next single block joins them, with one arrow per branch.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2.4))

    tails: list[
        tuple[float, float, str]
    ] = []  # (x, y, shape) where the next arrows start
    x = 0.0
    y_min, y_max = -BOX_H, BOX_H

    for stage in stages:
        if isinstance(stage, Block):
            if tails:
                x += _gap(*(t[2] for t in tails)) if any(t[2] for t in tails) else 0.9
            for tx, ty, shape in tails:
                _arrow(ax, (tx, ty), (x, 0.0), shape)
            w = _box(ax, x, 0.0, stage)
            tails = [(x + w, 0.0, stage.out)]
            x += w
            continue

        branches = list(stage)
        ys = [BRANCH_DY * ((len(branches) - 1) / 2 - i) for i in range(len(branches))]
        y_min, y_max = min(y_min, ys[-1] - BOX_H), max(y_max, ys[0] + BOX_H)
        x0 = x + (_gap(*(t[2] for t in tails)) if tails else 0.0)
        new_tails = []
        for branch, y in zip(branches, ys):
            bx, incoming = x0, tails
            for i, block in enumerate(branch):
                if i > 0:
                    bx += _gap(incoming[0][2])
                for tx, ty, shape in incoming:
                    _arrow(ax, (tx, ty), (bx, y), shape)
                w = _box(ax, bx, y, block)
                incoming = [(bx + w, y, block.out)]
                bx += w
            new_tails.append(incoming[0])
        # each branch runs on horizontally to a common x, its shape written above that
        # line; the arrows into the join then carry no label of their own
        x = max(t[0] for t in new_tails) + _gap(*(t[2] for t in new_tails))
        tails = []
        for tx, ty, shape in new_tails:
            ax.plot([tx, x], [ty, ty], color="#444", lw=1.2)
            if shape:
                ax.text(
                    x - _gap(shape) / 2,
                    ty + 0.12,
                    shape,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    family="monospace",
                    color="#222",
                )
            tails.append((x, ty, ""))

    # a trailing shape on the last block gets a short arrow of its own
    for tx, ty, shape in tails:
        if shape:
            g = _gap(shape)
            _arrow(ax, (tx, ty), (tx + g, ty), shape)
            x = max(x, tx + g)

    ax.set_xlim(-0.2, x + 0.2)
    ax.set_ylim(y_min - 0.1, y_max + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)
    return ax


def draw_architectures(
    rows: dict[str, Sequence[Stage]], note: str | None = None, scale: float = 0.62
) -> Figure:
    """Several architectures, one row each, drawn at the same scale so they can be compared.

    `rows` maps a row title to its stages. `note` is printed under the drawings.
    """
    widths, heights = [], []
    for stages in rows.values():
        probe_fig, probe_ax = plt.subplots()
        draw_architecture(stages, ax=probe_ax)
        widths.append(probe_ax.get_xlim()[1] - probe_ax.get_xlim()[0])
        heights.append(probe_ax.get_ylim()[1] - probe_ax.get_ylim()[0] + 0.45)
        plt.close(probe_fig)

    width = max(widths)
    legend_h = 0.4
    note_h = 0.2 + 0.17 * (note.count("\n") + 1) if note else 0.0
    fig_h = sum(heights) * scale + legend_h + note_h
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(width * scale, fig_h),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    used_kinds: set[str] = set()
    for ax, (title, stages) in zip(axes[:, 0], rows.items()):
        draw_architecture(stages, ax=ax, title=title)
        ax.set_xlim(-0.2, width - 0.2)
        for stage in stages:
            blocks = (
                [stage]
                if isinstance(stage, Block)
                else [b for branch in stage for b in branch]
            )
            used_kinds.update(b.kind for b in blocks)

    handles = [
        Patch(fc=COLOURS[k], ec="#555", label=LEGEND[k])
        for k in COLOURS
        if k in used_kinds
    ]
    fig.tight_layout(rect=(0, note_h / fig_h, 1, 1 - legend_h / fig_h))
    fig.legend(
        handles=handles,
        loc="upper right",
        ncol=len(handles),
        fontsize=8.5,
        frameon=False,
    )
    if note:
        fig.text(
            0.01, 0.02, note, fontsize=9, va="bottom", ha="left", family="monospace"
        )
    return fig


def _shape(t: Any) -> str:
    dims = ["B", *map(str, t.shape[1:])]
    return f"({', '.join(dims)})"


def trace(model: nn.Module, x: Any) -> list[Block]:
    """The blocks `model` actually passes `x` through, in order, with their output shapes.

    Every leaf module (one without children) that runs becomes a block; a GELU straight
    after a Linear is drawn inside that Linear's box. Every Linear except the last is a
    hidden layer. The first dimension of every shape is written as B.
    """
    calls: list[tuple[nn.Module, str]] = []
    hooks = [
        m.register_forward_hook(lambda mod, _inp, out: calls.append((mod, _shape(out))))
        for m in model.modules()
        if not list(m.children())
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    n_linear = sum(isinstance(m, nn.Linear) for m, _ in calls)
    blocks: list[Block] = []
    seen_linear = 0
    for module, shape in calls:
        name = type(module).__name__
        if isinstance(module, nn.Linear):
            seen_linear += 1
            kind = "layer" if seen_linear == n_linear else "hidden"
            blocks.append(
                Block(
                    f"Linear\n{module.in_features} → {module.out_features}", shape, kind
                )
            )
        elif (
            isinstance(module, nn.GELU | nn.ReLU)
            and blocks
            and blocks[-1].label.startswith("Linear")
        ):
            prev = blocks[-1]
            blocks[-1] = Block(f"{prev.label}\n+ {name}", shape, prev.kind)
        elif name == "GaussianNoise":
            blocks.append(
                Block(f"GaussianNoise\nσ = {module.sigma:.2g}", shape, "noise")
            )
        elif name.startswith(("Concatenate", "Stack")):
            blocks.append(Block(name, shape, "join"))
        else:
            blocks.append(Block(name, shape, "layer"))
    return blocks


# ---------------------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------------------


def _is_domain(value: Any) -> bool:
    return hasattr(value, "sample") and hasattr(value, "get_sampler")


def sample_config(param_space: dict, seed: int = 0) -> dict:
    """One configuration drawn from `param_space`, the way a single trial would get one."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, value in param_space.items():
        out[key] = value.sample(random_state=rng) if _is_domain(value) else value
        if isinstance(out[key], np.generic):
            out[key] = out[key].item()
    return out


def plot_search_space(
    param_space: dict, trial: dict | None = None, n: int = 4000, seed: int = 0
) -> Figure:
    """One panel per searched hyperparameter, filled with `n` samples from its distribution.

    A choice shows how often each option is drawn; a continuous range shows a histogram,
    on a log axis when the range is log-uniform, which gets one extra panel showing the
    same samples on a linear axis. `trial`, when given, is marked in every panel in red:
    one trial is one value per panel. Constant entries are listed in the title.
    """
    rng = np.random.default_rng(seed)
    searched = {k: v for k, v in param_space.items() if _is_domain(v)}
    fixed = {k: v for k, v in param_space.items() if not _is_domain(v)}
    log_keys = [
        k for k, v in searched.items() if "Log" in type(v.get_sampler()).__name__
    ]

    n_panels = len(searched) + len(log_keys)
    n_cols = min(n_panels, 3)
    n_rows = -(-n_panels // n_cols)
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.9 * n_rows + 0.4), squeeze=False
    )
    for ax in grid.flat[n_panels:]:
        ax.axis("off")
    axes = iter(grid.flat)

    for key, domain in searched.items():
        samples = np.asarray(domain.sample(size=n, random_state=rng))
        picked = None if trial is None else trial[key]
        sampler = type(domain.get_sampler()).__name__

        if hasattr(domain, "categories"):
            ax = next(axes)
            labels = [str(c) for c in domain.categories]
            freq = [np.mean(samples.astype(str) == lab) for lab in labels]
            colours = [
                "crimson" if picked is not None and lab == str(picked) else "#7a9cc6"
                for lab in labels
            ]
            ax.bar(labels, freq, color=colours)
            ax.set_ylim(0, 1)
            ax.set_ylabel("share of trials")
            ax.set_title(f"{key}\nchoice of {len(labels)}", fontsize=10)
            if max(len(lab) for lab in labels) > 5:
                ax.tick_params(axis="x", rotation=35)
            continue

        lo, hi = domain.lower, domain.upper
        is_log = key in log_keys
        bins = np.geomspace(lo, hi, 25) if is_log else np.linspace(lo, hi, 25)
        ax = next(axes)
        ax.hist(samples, bins=bins, color="#7a9cc6")
        if is_log:
            ax.set_xscale("log")
        kind = "log-uniform" if is_log else sampler.strip("_").lower()
        ax.set_title(f"{key}\n{kind} on [{lo:g}, {hi:g}]", fontsize=10)
        ax.set_yticks([])
        if picked is not None:
            ax.axvline(picked, color="crimson", lw=2)
        if is_log:
            ax = next(axes)
            ax.hist(samples, bins=np.linspace(lo, hi, 25), color="#b9c9de")
            ax.set_title(f"{key}, same samples\non a linear axis", fontsize=10)
            ax.set_yticks([])
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            if picked is not None:
                ax.axvline(picked, color="crimson", lw=2)

    title = f"{n} draws from the search space"
    if fixed:
        title += "   (fixed: " + ", ".join(f"{k} = {v}" for k, v in fixed.items()) + ")"
    if trial is not None:
        title += "   red: one sampled trial"
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    return fig
