"""Figures for 01_the_cache.ipynb.

The notebook computes every number itself (projections, neighbours, timings, accuracies);
these functions only draw them, so the cells stay about the computation.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def projection_pair(
    pca: np.ndarray, tsne: np.ndarray, labels: np.ndarray, names: Sequence[str]
) -> Figure:
    """Two 2D views of the same vectors, side by side, coloured by class.

    The class labels are used for the colour only; neither projection saw them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    titles = [
        "PCA: the two directions of largest variance",
        "t-SNE: keeps each point's nearest neighbours close",
    ]
    for ax, coords, title in zip(axes, [pca, tsne], titles, strict=True):
        for k, name in enumerate(names):
            points = coords[labels == k]
            ax.scatter(points[:, 0], points[:, 1], s=5, color=cmap(k % 10), label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        markerscale=3,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def neighbour_grid(
    images: Sequence, titles: Sequence[Sequence[str]], same_class: np.ndarray
) -> Figure:
    """One row per query: the query image, then its nearest neighbours.

    `images[r][c]` is a PIL image, `titles[r][c]` its label, and `same_class[r][c]` says
    whether neighbour c has the query's label (a mismatch is titled in red).
    """
    rows, cols = len(images), len(images[0])
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.25 * cols, 1.4 * rows), dpi=80, squeeze=False
    )
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(images[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            colour = "black" if same_class[r][c] else "red"
            ax.set_title(titles[r][c], fontsize=7, color=colour)
    axes[0, 0].set_title(f"QUERY\n{titles[0][0]}", fontsize=7, fontweight="bold")
    for r in range(1, rows):
        axes[r, 0].set_title(titles[r][0], fontsize=7, fontweight="bold")
    fig.tight_layout()
    return fig


def cost_curve(
    runs: np.ndarray,
    in_loop_minutes: np.ndarray,
    cached_minutes: np.ndarray,
    mark_runs: int,
) -> Figure:
    """Total wall-clock time against the number of heads trained, for both pipelines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        runs,
        in_loop_minutes,
        lw=2,
        color="tab:red",
        label="encoder inside the training loop",
    )
    ax.plot(
        runs,
        cached_minutes,
        lw=2,
        color="tab:blue",
        label="encode once, train heads on the cache",
    )
    ax.axvline(mark_runs, color="grey", lw=1, ls="--")
    ax.text(
        mark_runs,
        ax.get_ylim()[1] * 0.97,
        f" {mark_runs} trials\n (notebook 04's search)",
        fontsize=8,
        va="top",
        color="grey",
    )
    for series in (in_loop_minutes, cached_minutes):
        y = series[mark_runs - 1]
        ax.plot(mark_runs, y, "o", color="black", ms=4)
        ax.annotate(
            f"{y:.1f} min",
            (mark_runs, y),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("number of heads trained (experiments, search trials)")
    ax.set_ylabel("total time (minutes)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def encoder_bars(results: list[dict], datasets: Sequence[str]) -> Figure:
    """One panel per dataset: 1-NN and linear-head accuracy per encoder, chance as a line."""
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4.4), sharey=True, squeeze=False
    )
    width = 0.38
    for ax, ds in zip(axes[0], datasets, strict=True):
        rows = [r for r in results if r["dataset"] == ds]
        x = np.arange(len(rows))
        knn = ax.bar(
            x - width / 2,
            [r["knn"] for r in rows],
            width,
            color="tab:grey",
            label="1-NN, no training",
        )
        head = ax.bar(
            x + width / 2,
            [r["head"] for r in rows],
            width,
            color="tab:blue",
            label="trained linear head",
        )
        ax.bar_label(head, fmt="%.2f", fontsize=7, padding=2)
        ax.axhline(rows[0]["chance"], color="black", lw=1, ls=":")
        ax.text(
            len(rows) - 0.5, rows[0]["chance"] + 0.01, "chance", fontsize=7, ha="right"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r["encoder"].replace("_1.0_224", "") for r in rows],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("test accuracy")
    fig.legend(
        handles=[knn, head],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.09),
    )
    fig.tight_layout()
    return fig
