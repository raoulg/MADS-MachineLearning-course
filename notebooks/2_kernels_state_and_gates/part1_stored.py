"""The stored results of Part 1 of `04_exercise.ipynb`, for a student who runs it with
`QUICK = True`: three pictures in `img/`, drawn from one full run of Part 1 in the store.

    uv run notebooks/2_kernels_state_and_gates/part1_stored.py

It reads the latest session of `lesson2-notebook-runs` and the trials of the same
session of `lesson2-notebook`, so run Part 1 in full first. Nothing is trained here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optuna.trial import FrozenTrial, TrialState

from hpo.measure import Comparison
from hpo.record import StudyLog

OBJECTIVE = "macro_recall"
CLASSES = ["N", "S", "V"]
AXES = ["kernel_size", "width", "depth", "downsample", "skip", "gate", "lr"]
IMG = Path(__file__).parent / "img"


def repeats(trial: FrozenTrial) -> np.ndarray:
    """The objective of every repeat of a logged run (step = repeat)."""
    return np.array(
        [trial.intermediate_values[k] for k in sorted(trial.intermediate_values)]
    )


def compare_on(ax: plt.Axes, rows: list[Comparison]) -> None:
    """The notebook's `compare`: mean ± std and every repeat, the first row on top."""
    for y, row in enumerate(rows):
        ax.errorbar(row.mean, y, xerr=row.std, fmt="o", color="tab:blue", capsize=4)
        ax.scatter(row.values, [y] * len(row.values), color="tab:blue", alpha=0.3, s=14)
        label = f"{row.mean:.3f} ± {row.std:.3f}"
        if row.points is not None:
            label += f"   {row.points:+.2f} pt"
        ax.annotate(
            label,
            (row.mean, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(rows)), [r.name for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel(OBJECTIVE)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    IMG.mkdir(exist_ok=True)
    runs = StudyLog.load("lesson2-notebook-runs").study.trials
    session = max(t.user_attrs["session"] for t in runs)
    named = {
        t.user_attrs["name"]: t for t in runs if t.user_attrs["session"] == session
    }
    search = [
        t
        for t in StudyLog.load("lesson2-notebook").study.trials
        if t.user_attrs.get("session") == session and t.state == TrialState.COMPLETE
    ]
    print(f"session {session}: {len(search)} trials, runs {sorted(named)}")
    df = pd.DataFrame([{OBJECTIVE: t.value, **t.params} for t in search])

    # E3: every trial of the search, one panel per axis
    fig, panels = plt.subplots(1, len(AXES), figsize=(18, 3), sharey=True)
    for ax, axis in zip(panels, AXES, strict=True):
        values = df[axis].astype(str) if df[axis].dtype == bool else df[axis]
        ax.scatter(values, df[OBJECTIVE], alpha=0.6)
        ax.set_xlabel(axis)
        if axis in ("width", "lr"):
            ax.set_xscale("log")
    panels[0].set_ylabel(OBJECTIVE)
    plt.suptitle(
        f"stored: the {len(df)} trials of one full run of the search",
        x=0.1,
        ha="left",
    )
    plt.tight_layout()
    fig.savefig(IMG / "part1-search.png", dpi=110)
    plt.close(fig)

    # E4: the winner rerun against the handed-out design
    design = repeats(named["handed-out design"])
    winner = repeats(named["winner"])
    best = {
        k: v
        for k, v in named["winner"].params.items()
        if k in AXES and not isinstance(v, float)
    } | {"lr": f"{named['winner'].params['lr']:.1e}"}
    fig, ax = plt.subplots(figsize=(8, 2.2))
    compare_on(
        ax,
        [
            Comparison.of(f"handed-out design, {len(design)} runs", design),
            Comparison.of(
                f"the winner rerun, {len(winner)} runs", winner, versus=design
            ),
        ],
    )
    ax.set_title(
        f"stored: the winner's one run in the search {df[OBJECTIVE].max():.3f}\n"
        + ", ".join(f"{k} {v}" for k, v in best.items()),
        loc="left",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(IMG / "part1-winner.png", dpi=110)
    plt.close(fig)

    # E5: the same winner on the plain and on the balanced loss
    balanced = named["winner, balanced loss"]
    fig, (ax, table) = plt.subplots(
        2, 1, figsize=(8, 3.2), gridspec_kw={"height_ratios": [2.2, 1]}
    )
    compare_on(
        ax,
        [
            Comparison.of(f"winner, plain loss, {len(winner)} runs", winner),
            Comparison.of(
                f"winner, balanced loss, {len(winner)} runs",
                repeats(balanced),
                versus=winner,
            ),
        ],
    )
    ax.set_title("stored: the loss steers, the metric judges", loc="left", fontsize=9)
    rows = []
    for name, trial in [("plain loss", named["winner"]), ("balanced loss", balanced)]:
        m = trial.user_attrs["metrics"]
        rows.append(
            [name, f"{m['accuracy']:.3f}", f"{m['macro_recall']:.3f}"]
            + [f"{m['recall'][c]:.2f}" for c in CLASSES]
        )
    table.axis("off")
    table.table(
        cellText=rows,
        colLabels=["", "accuracy", "macro recall"] + [f"recall {c}" for c in CLASSES],
        loc="center",
        cellLoc="center",
    )
    plt.tight_layout()
    fig.savefig(IMG / "part1-balanced.png", dpi=110)
    plt.close(fig)
    print(f"written: {', '.join(p.name for p in sorted(IMG.glob('part1-*.png')))}")


if __name__ == "__main__":
    main()
