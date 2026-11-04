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
# It reads the conventions `runlog.log_run` writes: a `role` tag (baseline, trial, final,
# ablation, hypothesis), the score metric with `<metric>_std` when there were seeds,
# `n_params`, and the one-line `architecture` tag. Runs without them still show up in the
# counts; they just have less to say.

# %%
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
args, _ = parser.parse_known_args(sys.argv[1:] if "__file__" in globals() else [])

LESSON, METRIC = args.lesson, args.metric
OUT = COURSE_ROOT / "artefacts" / "class-analysis" / f"lesson{LESSON}"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option(
    "display.width", 160, "display.max_columns", 20, "display.max_colwidth", 60
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
        "score": pd.to_numeric(col("metrics", METRIC), errors="coerce"),
        "std": pd.to_numeric(col("metrics", f"{METRIC}_std"), errors="coerce"),
        "n_params": pd.to_numeric(col("metrics", "n_params"), errors="coerce"),
        "architecture": col("tags", "architecture"),
        "start": raw["start_time"],
    }
)
params = raw[[c for c in raw.columns if c.startswith("params.")]].rename(
    columns=lambda c: c[len("params.") :]
)
print(
    f"lesson {LESSON}: {runs.student.nunique()} students, {len(runs)} runs, metric `{METRIC}`\n"
)

# %% [markdown]
# ## Who submitted what

# %%
print(
    pd.crosstab(
        runs.student, runs.role, margins=True, margins_name="total"
    ).to_string(),
    "\n",
)

# %% [markdown]
# ## Baseline against final, per student
#
# The final is the run a student stands behind, rerun over seeds. `sigma` is the
# difference in units of the pooled seed spread: inside about two, it is not a result.


# %%
def latest(role: str) -> pd.DataFrame:
    sel = runs[(runs.role == role) & runs.score.notna()].sort_values("start")
    return sel.groupby("student").tail(1).set_index("student")[["score", "std"]]


base, final = latest("baseline"), latest("final")
best_trial = (
    runs[(runs.role == "trial") & runs.score.notna()].groupby("student").score.max()
)
table = base.join(final, lsuffix="_baseline", rsuffix="_final", how="outer").join(
    best_trial.rename("best_trial")
)
table["gain_pt"] = 100 * (table.score_final - table.score_baseline)
pooled = (table.std_baseline**2 + table.std_final**2) ** 0.5
table["sigma"] = (table.score_final - table.score_baseline) / pooled
table["trial_minus_final_pt"] = 100 * (table.best_trial - table.score_final)
print(table.sort_values("gain_pt", ascending=False).round(4).to_string(), "\n")
print(
    "trial_minus_final_pt: how much of the single-seed search maximum did not survive the rerun.\n"
)

if table.score_baseline.notna().any() and table.score_final.notna().any():
    t = table.dropna(subset=["score_baseline", "score_final"]).sort_values(
        "score_final"
    )
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.5))
    y = range(len(t))
    ax.hlines(y, t.score_baseline, t.score_final, color="0.7")
    ax.errorbar(
        t.score_baseline, y, xerr=t.std_baseline, fmt="o", label="baseline", capsize=3
    )
    ax.errorbar(t.score_final, y, xerr=t.std_final, fmt="o", label="final", capsize=3)
    ax.set_yticks(list(y), t.index)
    ax.set_xlabel(METRIC)
    ax.legend()
    ax.set_title("baseline and final per student, with seed spread")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_vs_final.png", dpi=150)

# %% [markdown]
# ## Score against parameters
#
# Every run with an architecture attached. A cloud with no slope says capacity bought
# nothing; the grey band, two seed spreads around the baselines, is what a run has to clear.

# %%
sized = runs.dropna(subset=["score", "n_params"])
if len(sized):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for role, group in sized.groupby("role"):
        ax.scatter(
            group.n_params,
            group.score,
            label=f"{role} ({len(group)})",
            alpha=0.35 if role == "trial" else 0.9,
            s=18 if role == "trial" else 40,
        )
    if len(base):
        mid, spread = base.score.mean(), base["std"].mean()
        ax.axhline(mid, color="k", ls="--", lw=1)
        ax.axhspan(
            mid - 2 * spread,
            mid + 2 * spread,
            color="k",
            alpha=0.08,
            label="baselines: mean +/- 2 seed std",
        )
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(METRIC)
    ax.legend(fontsize=8)
    ax.set_title(f"lesson {LESSON}: {METRIC} against parameters, every submitted run")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_params.png", dpi=150)

# %% [markdown]
# ## Which architectures appear
#
# The distinct architectures across the class, by the one-line signature, with how many
# students built each and the best and median score it reached.

# %%
arch = runs.dropna(subset=["architecture", "score"])
if len(arch):
    summary = (
        arch.groupby("architecture")
        .agg(
            students=("student", "nunique"),
            runs=("score", "size"),
            best=("score", "max"),
            median=("score", "median"),
            n_params=("n_params", "first"),
        )
        .sort_values("best", ascending=False)
    )
    print(summary.round(4).to_string(), "\n")

# %% [markdown]
# ## Which choices the searches explored
#
# For every parameter the trials varied: how often each value was tried and what it
# scored. An axis whose medians do not separate did not matter, whatever the best value
# was.

# %%
trials = runs.role == "trial"
for name in params.columns:
    values = params.loc[trials, name].dropna()
    if values.nunique() < 2:
        continue
    numeric = pd.to_numeric(values, errors="coerce")
    key = (
        pd.qcut(numeric, 4, duplicates="drop").astype(str)
        if numeric.notna().all() and values.nunique() > 8
        else values
    )
    stats = runs.loc[values.index].groupby(key).score.agg(["size", "median", "max"])
    stats["students"] = runs.loc[values.index].groupby(key).student.nunique()
    print(f"-- {name}")
    print(stats.round(4).to_string(), "\n")

# %%
print(f"figures in {OUT}")
if "__file__" not in globals():
    plt.show()
