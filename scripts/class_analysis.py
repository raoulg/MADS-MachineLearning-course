# %% [markdown]
# # What the class did
#
# The opening analysis for a lesson, over the runs `collect_runs.py` imported into the
# class store. Runs as a script:
#
#     uv run scripts/class_analysis.py --lesson 1
#
# or cell by cell (`# %%`) in VS Code or Jupyter, to show it in class. It prints the tables
# and writes the figures to artefacts/class-analysis/lesson<N>/.
#
# It reads the conventions `runlog` writes: a `role` tag (baseline, trial, final, ablation,
# hypothesis), `origin` (`notebook` for the runs as handed out, `study` for a student's own
# study, which also carries `study`, `search_space`, `searched_axes`, `space_<axis>`,
# `space_is_default` and `budget`), `dataset`, the score metric with `<metric>_std` and
# `n_seeds` when there were seeds, `n_params`, and the one-line `architecture` tag. Runs
# without them still show up in the counts; they just have less to say.
#
# Scores of different datasets are never ranked or plotted against each other: every table
# and figure that compares scores does so per dataset, and runs without a `dataset` tag
# form the dataset `unknown`.
#
# ## Reading the runs as a parallel-coordinates plot
#
# In `mlflow ui` on the class store: open an experiment, the **Chart** view, then **Add
# chart -> Parallel coordinates**. Filter first, on one dataset and one study
# (`tags.dataset = 'dtd' and tags.study = 'depth-on-eurosat'`, or `tags.origin =
# 'notebook'` for the handed-out runs). Put the params on the axes: `encoder` and
# `embed_dim` (the vectors a run starts from), `n_hidden_layers`, `hidden_dim`, `sigma`,
# `norm`, `skip`, `gate`, `lr`; and the metrics `accuracy`, `n_params` and `train_seconds`
# (seconds in training steps, evaluation not counted) as the outcomes. Every lesson 1 run
# logs all of them, so a study that varied one switch differs from its neighbours on one
# axis. `tags.searched_axes LIKE '%hidden_dim%'` finds the studies that searched an axis.

# %%
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
import mlflow
import pandas as pd

COURSE_ROOT = (
    Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
)
parser = argparse.ArgumentParser(
    description="Opening analysis of the class runs for one lesson."
)
parser.add_argument("--lesson", type=int, default=1)
parser.add_argument(
    "--metric", default="accuracy", help="the score to compare (default: %(default)s)"
)
parser.add_argument(
    "--tracking-uri",
    default=f"sqlite:///{COURSE_ROOT / 'artefacts' / 'mlflow' / 'class.db'}",
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.002,
    help="a handed-out run further than this from the class median counts as deviating "
    "(default: %(default)s)",
)
args, _ = parser.parse_known_args(sys.argv[1:] if "__file__" in globals() else [])

LESSON, METRIC = args.lesson, args.metric
OUT = COURSE_ROOT / "artefacts" / "class-analysis" / f"lesson{LESSON}"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option(
    "display.width", 200, "display.max_columns", 30, "display.max_colwidth", 70
)

mlflow.set_tracking_uri(args.tracking_uri)
experiments = [
    e for e in mlflow.search_experiments() if e.name.startswith(f"lesson{LESSON}-")
]
if not experiments:
    sys.exit(
        f"no lesson{LESSON}-* experiments in {args.tracking_uri}; run scripts/collect_runs.py first"
    )

raw = pd.DataFrame(
    mlflow.search_runs([e.experiment_id for e in experiments], output_format="pandas")
)
names = {e.experiment_id: e.name for e in experiments}


def col(prefix: str, name: str) -> pd.Series:
    key = f"{prefix}.{name}"
    return raw[key] if key in raw else pd.Series([None] * len(raw), index=raw.index)


runs = pd.DataFrame(
    {
        "student": col("tags", "student"),
        "experiment": raw["experiment_id"].map(names),
        "name": col("tags", "source_run_name"),
        "role": col("tags", "role").fillna("-"),
        "origin": col("tags", "origin").fillna("-"),
        "dataset": col("tags", "dataset").fillna("unknown"),
        "study": col("tags", "study"),
        "search_space": col("tags", "search_space"),
        "searched_axes": col("tags", "searched_axes"),
        "space_is_default": col("tags", "space_is_default"),
        "budget": col("tags", "budget"),
        "score": pd.to_numeric(col("metrics", METRIC), errors="coerce"),
        "std": pd.to_numeric(col("metrics", f"{METRIC}_std"), errors="coerce"),
        "n_seeds": pd.to_numeric(col("metrics", "n_seeds"), errors="coerce"),
        "n_params": pd.to_numeric(col("metrics", "n_params"), errors="coerce"),
        "architecture": col("tags", "architecture"),
        "start": raw["start_time"],
    }
)
params = raw[[c for c in raw.columns if c.startswith("params.")]].rename(
    columns=lambda c: c[len("params.") :]
)
# The search space per axis, as `runlog.space_tags` writes it: tags.space_<axis>.
space = raw[[c for c in raw.columns if c.startswith("tags.space_")]].rename(
    columns=lambda c: c[len("tags.space_") :]
)
space = space.drop(columns=[c for c in ["is_default"] if c in space.columns])
print(
    f"lesson {LESSON}: {runs.student.nunique()} students, {len(runs)} runs, metric `{METRIC}`, "
    f"datasets {sorted(runs.dataset.unique())}\n"
)


def sigma(a: pd.Series, b: pd.Series) -> float:
    """(b - a) in units of the standard error of the difference, as the notebooks print it:
    `std` is the spread over seeds (ddof 0), so std**2 / (n - 1) is var(ddof 1) / n."""
    se2 = 0.0
    for run in (a, b):
        n = run["n_seeds"]
        if not (n and n > 1) or pd.isna(run["std"]):
            return float("nan")
        se2 += run["std"] ** 2 / (n - 1)
    return float((b["score"] - a["score"]) / np.sqrt(se2)) if se2 > 0 else float("nan")


def latest(sel: pd.DataFrame) -> pd.Series | None:
    sel = sel[sel.score.notna()].sort_values("start")
    return None if sel.empty else sel.iloc[-1]


def pm(run: pd.Series | None) -> str:
    if run is None:
        return "-"
    if pd.isna(run["std"]):
        return f"{run['score']:.3f}"
    return f"{run['score']:.3f}±{run['std']:.3f}"


# %% [markdown]
# ## The studies
#
# One row per student and study: the axes it searched, whether its space is the handed-out
# one, and how it came out. `baseline` is the study's own baseline run, or else the
# student's handed-out baseline on the same dataset. `sigma` is the gain in units of the
# standard error of the difference: inside about two, the difference is not measured,
# whatever its sign. The second table is each study's space, axis by axis; `-` is an axis
# the study does not have.

# %%
study_runs = runs[runs.origin == "study"]
rows, axes_rows = [], []
for (student, study), group in study_runs.groupby(["student", "study"], dropna=False):
    final = latest(group[group.role == "final"])
    dataset = final["dataset"] if final is not None else group.dataset.mode().iloc[0]
    base = latest(group[(group.role == "baseline") & (group.dataset == dataset)])
    if base is None:
        base = latest(
            runs[
                (runs.student == student)
                & (runs.origin == "notebook")
                & (runs.role == "baseline")
                & (runs.dataset == dataset)
            ]
        )
    first = group.iloc[0]
    budget = json.loads(first["budget"]) if isinstance(first["budget"], str) else {}
    both = final is not None and base is not None
    rows.append(
        {
            "student": student,
            "study": study,
            "dataset": dataset,
            "searched": first["searched_axes"] or "-",
            "default": first["space_is_default"],
            "trials": f"{int((group.role == 'trial').sum())}/{budget.get('n_trials', '?')}",
            "baseline": pm(base),
            "final": pm(final),
            "gain_pt": 100 * (final["score"] - base["score"]) if both else np.nan,
            "sigma": sigma(base, final) if both else np.nan,
        }
    )
    axes_rows.append(
        {"student": student, "study": study, **space.loc[first.name].fillna("-")}
    )
studies = pd.DataFrame(rows)
if studies.empty:
    print("no study runs (origin=study) submitted\n")
else:
    print(
        studies.sort_values(["dataset", "sigma"], ascending=[True, False])
        .round(2)
        .to_string(index=False),
        "\n",
    )
    per_axis = pd.DataFrame(axes_rows).set_index(["student", "study"])
    per_axis = per_axis.loc[:, (per_axis != "-").any()]
    print(per_axis.sort_index().to_string(), "\n")
    handed_out = runs[(runs.origin == "notebook") & runs.search_space.notna()]
    if len(handed_out):
        print(f"handed-out space: {handed_out.search_space.iloc[0]}\n")

# %% [markdown]
# ## Who submitted what

# %%
print(
    pd.crosstab(
        runs.student, [runs.origin, runs.role], margins=True, margins_name="total"
    ).to_string(),
    "\n",
)

# %% [markdown]
# ## The runs as handed out
#
# Every student ran the same handed-out notebooks, so these runs should agree. Per run
# name: the class median, how many students ran it with the same parameters, and how many
# of those differ from the median by more than the tolerance. A student whose handed-out
# baseline differs has a setup problem worth a look. `other_params` counts students whose
# run of that name had different parameters (an unseeded search draws different trials),
# which are left out of the comparison.

# %%
notebook = runs[(runs.origin == "notebook") & runs.score.notna()].copy()
if notebook.empty:
    print("no handed-out runs (origin=notebook) submitted\n")
else:
    notebook["params_key"] = (
        params.loc[notebook.index].astype(str).agg("|".join, axis=1)
    )
    rows = []
    deviating: list[str] = []
    for (exp, name, dataset), group in notebook.groupby(
        ["experiment", "name", "dataset"]
    ):
        group = group.sort_values("start").groupby("student").tail(1)
        usual = group.params_key.mode().iloc[0]
        same = group[group.params_key == usual]
        median = same.score.median()
        off = same[(same.score - median).abs() > args.tolerance]
        deviating += [f"{s} ({exp}/{name})" for s in off.student]
        rows.append(
            {
                "experiment": exp,
                "name": name,
                "dataset": dataset,
                "role": group.role.iloc[0],
                "students": len(same),
                "median": median,
                "deviating": len(off),
                "other_params": len(group) - len(same),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")
    print(
        "deviating by more than",
        args.tolerance,
        ":",
        ", ".join(deviating) if deviating else "nobody",
        "\n",
    )

# %% [markdown]
# ## How much identical code varies
#
# Per handed-out run (the trials left out): one dot per student, with that student's own
# seed spread as a bar, and the class median as a dashed line. Every panel has its own
# axis, so read the spread within a panel, not across them. Dots that scatter more than
# their bars says machines differ; bars wider than the scatter says seeds do.

# %%
shown = notebook[notebook.role != "trial"] if len(notebook) else notebook
panels = sorted(shown.groupby(["experiment", "name", "dataset"]).groups)
if panels:
    students = sorted(shown.student.dropna().unique())
    ncols = min(4, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 0.22 * len(students) * nrows + 1.2 * nrows),
        squeeze=False,
        sharey=True,
    )
    y = {s: i for i, s in enumerate(students)}
    for ax, (exp, name, dataset) in zip(axs.flat, panels):
        group = shown[
            (shown.experiment == exp)
            & (shown.name == name)
            & (shown.dataset == dataset)
        ]
        group = group.sort_values("start").groupby("student").tail(1)
        ax.errorbar(
            group.score,
            group.student.map(y),
            xerr=group["std"].fillna(0),
            fmt="o",
            ms=4,
            capsize=2,
            lw=1,
        )
        ax.axvline(group.score.median(), color="k", ls="--", lw=0.8)
        ax.set_title(
            f"{name}\n{exp.removeprefix(f'lesson{LESSON}-')}, {dataset}", fontsize=8
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis="x", alpha=0.3)
    for ax in axs.flat[len(panels) :]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_yticks(range(len(students)), students, fontsize=7)
    fig.suptitle(f"lesson {LESSON}: the handed-out runs, per student ({METRIC})")
    fig.tight_layout()
    fig.savefig(OUT / "handed_out_spread.png", dpi=150)

# %% [markdown]
# ## Baseline against final, per student and dataset
#
# The final is the run a student stands behind, rerun over seeds: the handed-out search's
# winner, and the final of each study. `sigma` as in the studies table.


# %%
def finals_table(sel: pd.DataFrame) -> pd.DataFrame:
    out = []
    pairs = sel[["student", "dataset"]].drop_duplicates().itertuples(index=False)
    for student, dataset in pairs:
        group = sel[(sel.student == student) & (sel.dataset == dataset)]
        base = latest(group[group.role == "baseline"])
        for (origin, study), finals in group[group.role == "final"].groupby(
            ["origin", group.study.fillna("-")]
        ):
            final = latest(finals)
            best_trial = group[
                (group.role == "trial")
                & (group.origin == origin)
                & (group.study.fillna("-") == study)
            ].score.max()
            if final is None:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "student": student,
                    "from": "handed out" if origin == "notebook" else study,
                    "baseline": base["score"] if base is not None else np.nan,
                    "final": final["score"],
                    "final_std": final["std"],
                    "gain_pt": 100 * (final["score"] - base["score"])
                    if base is not None
                    else np.nan,
                    "sigma": sigma(base, final) if base is not None else np.nan,
                    "trial_minus_final_pt": 100 * (best_trial - final["score"]),
                }
            )
    return pd.DataFrame(out)


finals = finals_table(runs)
for dataset, group in finals.groupby("dataset") if len(finals) else []:
    print(f"-- dataset {dataset}")
    print(
        group.drop(columns="dataset")
        .sort_values("gain_pt", ascending=False)
        .round(4)
        .to_string(index=False),
        "\n",
    )
print(
    "trial_minus_final_pt: how much of the single-seed search maximum did not survive the rerun.\n"
)

for dataset, t in finals.groupby("dataset") if len(finals) else []:
    t = t.dropna(subset=["baseline", "final"]).sort_values("final")
    if t.empty:
        continue
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.5))
    y = list(range(len(t)))
    ax.hlines(y, t.baseline, t.final, color="0.7")
    ax.plot(t.baseline, y, "o", label="baseline")
    ax.errorbar(t.final, y, xerr=t.final_std, fmt="o", label="final", capsize=3)
    ax.set_yticks(y, [f"{s} / {f}" for s, f in zip(t.student, t["from"])])
    ax.set_xlabel(METRIC)
    ax.legend()
    ax.set_title(f"{dataset}: baseline and final per student, with seed spread")
    fig.tight_layout()
    fig.savefig(OUT / f"baseline_vs_final_{dataset}.png", dpi=150)

# %% [markdown]
# ## Score against parameters, per dataset
#
# The study runs, coloured per student, with the handed-out baselines drawn on top. A cloud
# with no slope says capacity bought nothing; the grey band, two seed spreads around the
# baselines, is what a run has to clear.

# %%
sized = runs.dropna(subset=["score", "n_params"])
for dataset, group in sized.groupby("dataset"):
    mine = group[group.origin == "study"]
    bases = group[(group.origin == "notebook") & (group.role == "baseline")]
    if mine.empty and bases.empty:
        continue
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = plt.get_cmap(
        "tab20"
    )  # 20 distinct colours: one per student in a class of 15
    for i, (student, s) in enumerate(mine.groupby("student")):
        ax.scatter(
            s.n_params,
            s.score,
            label=f"{student} ({len(s)})",
            alpha=0.7,
            s=22,
            color=colours(i % 20),
        )
    if len(bases):
        mid, spread = bases.score.mean(), bases["std"].mean()
        ax.axhline(mid, color="k", ls="--", lw=1, zorder=5)
        if not np.isnan(spread):
            ax.axhspan(
                mid - 2 * spread,
                mid + 2 * spread,
                color="k",
                alpha=0.08,
                label="baselines: mean +/- 2 seed std",
                zorder=4,
            )
        ax.scatter(
            bases.n_params,
            bases.score,
            marker="*",
            s=160,
            color="k",
            label=f"baseline ({len(bases)})",
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(METRIC)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"lesson {LESSON}, {dataset}: {METRIC} against parameters")
    fig.tight_layout()
    fig.savefig(OUT / f"score_vs_params_{dataset}.png", dpi=150)

# %% [markdown]
# ## Which architectures appear, per dataset
#
# The distinct architectures across the class, by the one-line signature, with how many
# students built each and the best and median score it reached. A class of 15 builds many;
# the table keeps the best `TOP`.

# %%
TOP = 15
arch = runs.dropna(subset=["architecture", "score"])
for dataset, group in arch.groupby("dataset"):
    summary = (
        group.groupby("architecture")
        .agg(
            students=("student", "nunique"),
            runs=("score", "size"),
            best=("score", "max"),
            median=("score", "median"),
            n_params=("n_params", "first"),
        )
        .sort_values("best", ascending=False)
    )
    summary.index = [a if len(a) <= 90 else a[:87] + "..." for a in summary.index]
    print(f"-- dataset {dataset}: {len(summary)} architectures, the best {TOP} shown")
    print(summary.head(TOP).round(4).to_string(), "\n")

# %% [markdown]
# ## Which choices each study explored
#
# Per student and study, for every parameter its trials varied: how often each value was
# tried and what it scored. An axis whose medians do not separate did not matter in that
# search, whatever the best value was.


# %%
def quartile_labels(numeric: pd.Series) -> pd.Series:
    """Four bins of equal count, each labelled by the range of values it holds."""
    bins = pd.qcut(numeric, 4, duplicates="drop")
    low = numeric.min()  # qcut widens the lowest edge a little below the smallest value
    return bins.map(lambda b: f"{max(b.left, low):.3g} to {b.right:.3g}").astype(str)


trials = runs[(runs.role == "trial") & (runs.origin == "study")]
for (student, study), group in trials.groupby(["student", "study"]):
    print(f"== {student} / {study}  [{', '.join(sorted(group.dataset.unique()))}]")
    for name in params.columns:
        values = params.loc[group.index, name].dropna()
        if values.nunique() < 2:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        key = values
        if numeric.notna().all() and values.nunique() > 4:
            # a continuous axis: four bins of equal count, labelled by their range
            key = quartile_labels(numeric)
        elif numeric.notna().all() and (numeric % 1 != 0).any():
            key = numeric.map(lambda x: f"{x:.3g}")
        elif numeric.notna().all():
            key = numeric.astype(int)  # sorts 64, 256, 1024 as numbers
        stats = runs.loc[values.index].groupby(key).score.agg(["size", "median", "max"])
        print(f"-- {name}")
        print(stats.round(4).to_string(), "\n")

# %%
print(f"figures in {OUT}")
if "__file__" not in globals():
    plt.show()
