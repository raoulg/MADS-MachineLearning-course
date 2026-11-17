"""The class view: one page that shows what the whole class did in a lesson.

    uv run scripts/class_analysis.py --lesson 1          # fetch main, write the page, print its path
    uv run scripts/class_analysis.py --lesson 1 --open   # ... and open it in the browser
    uv run scripts/class_analysis.py --demo              # the page on made-up data

The class's runs are the `lesson<N>/<student>/<run_id>.json` files on the class
repository's `main`; they are read with git exactly as `collect_runs.py` reads them. The
result is one self-contained HTML page, `artefacts/class-analysis/lesson<N>/index.html`,
read from top to bottom in the lesson: who submitted what; the handed-out baselines side
by side, each with its spread between runs; baseline against final per student, with the
spread drawn; the handed-out search pooled over the class; each student's own study;
the ideas of the class as a graph of what builds on what; which run built on whose;
which architectures appeared and what they cost.

optuna-dashboard (`make dashboard LESSON=<N>` opens both) shows one study at a time; this
page is what it cannot show: the class pooled, and a student, a role or a dataset as the
thing to filter on.

Scores of different datasets are never compared with each other: every picture that
compares scores does so per dataset. Every figure reads one metric, named on it: by
default the objective most studies of the lesson search, or `--metric macro_recall`, or
one class of a metric per class, `--metric recall/F`. A run that does not log the
metric has no score on the page.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import textwrap
import webbrowser
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_submission import COURSE, Submission, explain, idea_parents, is_number

COURSE_ROOT = Path(__file__).resolve().parents[1]
METRIC = "accuracy"

ROLE_COLORS = {
    "baseline": "#8a8f98",
    "trial": "#9db7d5",
    "final": "#1f5fa8",
    "ablation": "#d98e3b",
    "hypothesis": "#7a5aa6",
}
GAIN = "#2a8a5a"
LOSS = "#b8463c"
INK = "#1d2330"
MUTED = "#6a7280"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# The params every study of lesson 1 can search, in the order of the parallel plot.
AXES = [
    "n_hidden_layers",
    "hidden_dim",
    "activation",
    "norm",
    "skip",
    "dropout",
    "sigma",
    "lr",
]
CATEGORICAL = {"activation", "norm", "skip", "encoder", "dataset", "gate"}


# ---- the runs as a table ---------------------------------------------------------------


def origin_of(experiment: str) -> str:
    """`notebook` for the studies of the notebook as handed out, `study` for a student's
    own, `other` for anything else."""
    if re.fullmatch(r"lesson\d+-notebook(-runs)?", experiment):
        return "notebook"
    if re.match(r"lesson\d+-study", experiment):
        return "study"
    return "other"


def study_of(experiment: str) -> str | None:
    """The student's name for a study: `lesson1-study-sam-depth-runs` -> `sam-depth`."""
    m = re.fullmatch(r"lesson\d+-study-(.+?)(-runs)?", experiment)
    return m.group(1) if m else None


def word(value: object, default: str = "") -> str:
    """A user attr that is read as text: `default` when a submission holds another type.
    The format leaves user attrs free, so the page reads none of them on trust."""
    return value if isinstance(value, str) else default


def plain(text: object) -> str:
    """Text of a student for a label plotly draws: plotly reads a label as HTML, so the
    characters that would open a tag or a link are escaped."""
    return html.escape(str(text))


def page_metric(subs: Iterable[Submission]) -> str:
    """The metric a page reads when none is asked for: the objective most studies of
    the lesson search, accuracy when there is no study."""
    counts = Counter(sub.objective() for sub in subs)
    return counts.most_common(1)[0][0] if counts else METRIC


def metric_names(subs: Iterable[Submission]) -> list[str]:
    """Every metric any of `subs` logs, scalars first, as `--metric` takes them."""
    seen: dict[str, None] = {}
    for sub in subs:
        for name in sub.metric_names():
            seen.setdefault(name, None)
    return sorted(seen, key=lambda n: ("/" in n, n))


def curve_of(sub: Submission, metric: str) -> list[float] | None:
    """`metric` after every epoch of a run: the curve a trial logs as the user attr of
    that name, or, for a trial of a search on `metric`, the values it reported per
    epoch. None for a run without one."""
    attrs = sub.run.attrs
    logged = attrs.get(metric)
    if isinstance(logged, list) and logged and all(is_number(v) for v in logged):
        return [float(v) for v in logged]
    if attrs.get("role") == "trial" and metric == sub.objective():
        return [p.value for p in sub.run.intermediate] or None
    return None


def runs_frame(subs: Iterable[Submission], metric: str = METRIC) -> pd.DataFrame:
    """One row per submitted run, with the params as `p_<name>` columns. `score` and
    `std` are `metric` (a scalar such as `macro_recall`, or one class as `recall/F`) and
    its spread between runs, `curve` the metric per epoch; a run that does not log it
    has no score."""
    rows = []
    for sub in subs:
        run = sub.run
        attrs = run.attrs
        study_attrs = run.study.attrs
        budget = study_attrs.get("budget")
        row = {
            "student": sub.student,
            "lesson": sub.lesson,
            "ref": sub.ref(),
            "experiment": run.experiment,
            "origin": origin_of(run.experiment),
            "study": study_of(run.experiment),
            "name": run.run_name,
            "role": word(attrs.get("role"), "-"),
            "dataset": word(attrs.get("dataset")) or "unknown",
            "session": word(attrs.get("session")) or None,
            "status": run.status,
            # every run is read on one metric, whatever its study searched
            "score": sub.metric(metric),
            "std": sub.spread(metric),
            "objective": sub.objective(),
            "n": attrs.get("n_repeats"),
            "n_params": attrs.get("n_params"),
            "train_seconds": attrs.get("train_seconds"),
            "architecture": word(attrs.get("architecture")) or None,
            "builds_on": word(attrs.get("builds_on")),
            "search_space": word(study_attrs.get("search_space")) or None,
            "searched_axes": study_attrs.get("searched_axes"),
            "space_is_default": study_attrs.get("space_is_default"),
            "idea": word(study_attrs.get("idea")),
            "idea_name": word(study_attrs.get("idea_name")),
            "idea_author": word(study_attrs.get("idea_author")),
            "idea_parents": word(study_attrs.get("idea_parents")),
            "expects": word(study_attrs.get("expects")),
            "budget": budget if isinstance(budget, dict) else None,
            "curve": curve_of(sub, metric),
            "epochs_run": len(run.intermediate),
            "start": run.start_time,
            "design": "design.py" in run.artifacts,
            "design_script": word(run.artifacts.get("design.py")) or None,
        }
        row.update({f"p_{k}": v for k, v in run.params.items()})
        # a trial's fixed axes (the dataset, ...) are in its config, not its params
        config = attrs.get("config")
        for k, v in (config if isinstance(config, dict) else {}).items():
            row.setdefault(f"p_{k}", v)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("score", "std", "n", "n_params", "train_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # A student who ran a handed-out notebook twice submits both sessions; only the latest
    # counts, so the class compares one execution per student.
    handed = df.origin == "notebook"
    latest = (
        df[handed]
        .groupby(["student", "experiment"])
        .session.transform("max")
        .reindex(df.index)
    )
    return df[~(handed & df.session.notna() & (df.session != latest))].reset_index(
        drop=True
    )


def sigma(diff_a: pd.Series, diff_b: pd.Series) -> float:
    """`a` against `b` in pooled standard errors of their repeats; NaN without a spread."""
    se = math.sqrt(
        (diff_a["std"] ** 2) / diff_a["n"] + (diff_b["std"] ** 2) / diff_b["n"]
    )
    return (diff_a["score"] - diff_b["score"]) / se if se > 0 else float("nan")


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline against final, per student and study: both scores with their spread
    between runs, the gain in points and in sigmas, and how much of the best single trial
    is left in the final. Own studies only; one row per student and study."""
    rows = []
    own = df[df.origin == "study"]
    for (student, study), group in own.groupby(["student", "study"]):
        base = group[(group.role == "baseline") & group.score.notna()]
        final = group[(group.role == "final") & group.score.notna()]
        if base.empty or final.empty:
            continue
        b, f = base.sort_values("start").iloc[-1], final.sort_values("start").iloc[-1]
        trials = group[(group.role == "trial") & (group.status == "COMPLETE")]
        best = trials.score.max() if len(trials) else float("nan")
        gain = f.score - b.score
        rows.append(
            {
                "student": student,
                "study": study,
                "dataset": f.dataset,
                "objective": f.objective,
                "base": b.score,
                "base_std": b["std"],
                "base_n": b["n"],
                "final": f.score,
                "final_std": f["std"],
                "final_n": f["n"],
                "gain_pt": 100 * gain,
                "sigma": sigma(f, b),
                "best_trial": best,
                "survived": gain / (best - b.score) if best > b.score else float("nan"),
                "n_trials": len(trials),
                "pruned": int((group.role == "trial").sum() - len(trials)),
                "search_space": f.search_space,
                "searched_axes": f.searched_axes,
            }
        )
    return pd.DataFrame(rows)


def handed_out_baselines(df: pd.DataFrame) -> pd.DataFrame:
    sel = df[(df.origin == "notebook") & (df.role == "baseline") & df.score.notna()]
    return sel.sort_values("score").reset_index(drop=True)


def handed_out_trials(df: pd.DataFrame) -> pd.DataFrame:
    """The trials of the handed-out search, pooled over the class: one search of the
    space many times the size. Pruned trials are not scores and stay out."""
    sel = df[
        (df.origin == "notebook")
        & (df.role == "trial")
        & (df.status == "COMPLETE")
        & df.score.notna()
    ]
    return sel.reset_index(drop=True)


def links(df: pd.DataFrame) -> pd.DataFrame:
    """Who built on whom: per `builds_on` entry, the run, the classmate's run it names
    and both scores. A target that is not among the submissions has no score."""
    by_ref = df.set_index("ref")
    rows = []
    for _, r in df[df.builds_on.astype(bool)].iterrows():
        for ref in r.builds_on.split(","):
            target = by_ref.loc[ref] if ref in by_ref.index else None
            if isinstance(target, pd.DataFrame):
                target = target.iloc[0]
            rows.append(
                {
                    "student": r.student,
                    "run": r["name"],
                    "builds_on": ref.split("/")[1],
                    "on_run": None if target is None else target["name"],
                    "score": r.score,
                    "on_score": np.nan if target is None else target.score,
                    "found": target is not None,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["diff_pt"] = 100 * (out.score - out.on_score)
    return out


IDEA_COLUMNS = [
    "name", "idea", "author", "parents", "missing", "students", "n_runs", "student",
    "study", "dataset", "objective", "base", "base_std", "base_n", "final", "final_std",
    "final_n", "gain_pt", "sigma", "verdict", "expects", "budget", "axes", "round",
]  # fmt: skip


def number(value: object) -> float:
    """`value` as a float; NaN for anything that is not a finite number."""
    try:
        out = float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def standard_error(*runs: tuple[object, object]) -> float:
    """The pooled standard error of the means of `runs`, each (spread, repeats); NaN
    when one of them has no spread."""
    total = 0.0
    for std, n in runs:
        s, k = number(std), number(n)
        if math.isnan(s) or math.isnan(k) or k <= 0:
            return float("nan")
        total += s**2 / k
    return math.sqrt(total) if total > 0 else float("nan")


def verdict(sig: float | None) -> str:
    """A result at a glance: `better` or `worse` than its baseline by more than two
    sigmas, `inside` the spread between runs, or `unmeasured`."""
    if sig is None or math.isnan(sig):
        return "unmeasured"
    if sig >= 2:
        return "better"
    return "worse" if sig <= -2 else "inside"


# What an idea's row takes from the study that stands for it, when it has a result.
RESULT = (
    "base",
    "base_std",
    "base_n",
    "final",
    "final_std",
    "final_n",
    "gain_pt",
    "sigma",
)


def first_of(values: Iterable[object], kind: type) -> object | None:
    """The first of `values` that is a non-empty `kind`."""
    return next((v for v in values if isinstance(v, kind) and v), None)


def ideas(df: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """One row per idea of the class: its name, words, author, parents (the names it
    builds on; `missing` those not among the submissions), the students who ran it and
    their number of runs, its round (the column of the graph, `idea_layout`), and the
    study that stands for it: the one `comparisons` has with the most trials, or else
    the one with the most trials. Of that study: its result (the winner against its own
    baseline, in points and in sigmas, both with their spread and repeats), its dataset
    and objective, what its author `expects`, its `budget` and the axes it searched.

    An idea is its name, so the handed-out idea of author `course` is one row for the
    whole class; it has no result of its own, the class's baselines and pooled search
    are its outcome, and its axes are those of the handed-out search."""
    if df.empty or "idea_name" not in df:
        return pd.DataFrame(columns=IDEA_COLUMNS)
    carried = df[df.idea_name.astype(bool)]
    names = set(carried.idea_name)
    idea_of = {
        (s, st): n
        for s, st, n in zip(carried.student, carried.study, carried.idea_name)
        if isinstance(st, str)
    }
    result: dict[str, SimpleNamespace] = {}
    for r in rows(cmp):
        name = idea_of.get((r.student, r.study))
        if name is not None and (
            name not in result or r.n_trials > result[name].n_trials
        ):
            result[name] = r
    out = []
    for name, group in carried.groupby("idea_name", sort=False):
        first = group.iloc[0]
        parents = idea_parents(first.idea_parents)
        c = result.get(str(name))
        if c is not None:
            student, study = c.student, c.study
        else:
            own = group[group.study.map(lambda s: isinstance(s, str))]
            counts = own[own.role == "trial"].groupby(["student", "study"]).size()
            if counts.empty:
                counts = own.groupby(["student", "study"]).size()
            student, study = counts.idxmax() if len(counts) else (None, None)
        if study is None:
            ran = group[(group.origin == "notebook") & (group.role == "trial")]
        else:
            ran = group[(group.student == student) & (group.study == study)]
        axes = first_of(ran.searched_axes, str)
        out.append(
            {
                "name": name,
                "idea": first.idea,
                "author": first.idea_author,
                "parents": parents,
                "missing": [p for p in parents if p not in names],
                "students": sorted(group.student.unique()),
                "n_runs": len(group),
                "student": student,
                "study": study,
                "dataset": None if c is None else c.dataset,
                "objective": None if c is None else c.objective,
                **{key: np.nan if c is None else getattr(c, key) for key in RESULT},
                "verdict": verdict(None if c is None else c.sigma),
                "expects": first_of(ran.expects, str) or "",
                "budget": first_of(ran.budget, dict),
                "axes": str(axes).split(",") if axes else [],
            }
        )
    out.sort(key=lambda r: (r["author"] != COURSE, r["author"], r["name"]))
    frame = pd.DataFrame(out, columns=IDEA_COLUMNS)
    place = idea_layout(frame)
    frame["round"] = [place[n][0] for n in frame.name]
    return frame


EDGE_COLUMNS = [
    "parent", "child", "parent_author", "child_author", "diff_pt", "band_pt", "sigma",
    "verdict", "number", "why",
]  # fmt: skip


def edges(ids: pd.DataFrame) -> pd.DataFrame:
    """One row per arrow of the graph, from a parent to the idea that builds on it. The
    number on it, `diff_pt`, is the child's winner minus the parent's winner in points;
    `sigma` is that difference in pooled standard errors of the two winners' repeats,
    `band_pt` two of those standard errors in points, and `verdict` reads `sigma` as
    `verdict` does. An arrow has no number (`number` False, `why` says why) from a
    parent that is not submitted, from the course's idea (the child's box holds its
    winner against that baseline already), from or to an idea without a winner, and
    between winners of different datasets or metrics."""
    by = {r.name: r for r in rows(ids)}
    out = []
    for child in rows(ids):
        for p in child.parents:
            parent = by.get(p)
            row = {
                "parent": p, "child": child.name, "child_author": child.author,
                "parent_author": None if parent is None else parent.author,
                "diff_pt": np.nan, "band_pt": np.nan, "sigma": np.nan,
                "verdict": "unmeasured", "number": False, "why": "",
            }  # fmt: skip
            if parent is None:
                row["why"] = "the parent is not among the submissions"
            elif parent.author == COURSE:
                row["why"] = (
                    "the handed-out idea: the box holds the gain on its baseline"
                )
            elif math.isnan(number(parent.final)) or math.isnan(number(child.final)):
                row["why"] = "no winner against a baseline on both ends"
            elif parent.dataset != child.dataset:
                row["why"] = f"different datasets ({parent.dataset}, {child.dataset})"
            elif parent.objective != child.objective:
                row["why"] = (
                    f"different metrics ({parent.objective}, {child.objective})"
                )
            else:
                diff = child.final - parent.final
                se = standard_error(
                    (child.final_std, child.final_n), (parent.final_std, parent.final_n)
                )
                row.update(
                    diff_pt=100 * diff,
                    band_pt=200 * se,
                    sigma=diff / se,
                    verdict=verdict(diff / se),
                    number=True,
                )
            out.append(row)
    return pd.DataFrame(out, columns=EDGE_COLUMNS)


def rows(df: pd.DataFrame) -> list[SimpleNamespace]:
    """The rows of `df` as objects with the columns as attributes."""
    return [SimpleNamespace(**r) for r in df.to_dict("records")]


# ---- the pictures ----------------------------------------------------------------------


def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    layout = {"margin": {"l": 70, "r": 30, "t": 50, "b": 55}, **layout}
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": FONT, "size": 13, "color": INK},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        **layout,
    )
    fig.update_xaxes(
        showline=True, linecolor="#c9ced6", gridcolor="#eef0f3", zeroline=False
    )
    fig.update_yaxes(
        showline=True, linecolor="#c9ced6", gridcolor="#eef0f3", zeroline=False
    )
    return fig


def fig_who_submitted(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["student", "role"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for role, color in ROLE_COLORS.items():
        if role in counts:
            fig.add_bar(
                y=counts.index, x=counts[role], name=role, orientation="h",
                marker_color=color,
            )  # fmt: skip
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="submitted runs")
    return style(fig, max(260, 26 * len(counts) + 110))


def fig_baselines(df: pd.DataFrame, metric: str = METRIC) -> go.Figure:
    """The handed-out baseline of every student, each with its spread between runs: how
    far one model fluctuates, and so how big a difference has to be to mean anything."""
    base = handed_out_baselines(df)
    fig = go.Figure()
    if base.empty:
        return style(fig, 200)
    median = base.score.median()
    spread = base["std"].median()
    fig.add_hrect(
        y0=median - 2 * spread, y1=median + 2 * spread, fillcolor="#1f5fa8",
        opacity=0.10, line_width=0,
    )  # fmt: skip
    fig.add_hline(y=median, line_color="#1f5fa8", line_width=1, line_dash="dot")
    fig.add_scatter(
        x=base.student, y=base.score, mode="markers",
        marker={"size": 11, "color": ROLE_COLORS["final"]},
        error_y={"type": "data", "array": base["std"].fillna(0), "color": "#6f93c0", "thickness": 2},
        hovertemplate="%{x}: %{y:.4f}<extra></extra>", showlegend=False,
    )  # fmt: skip
    fig.add_annotation(
        x=1, xref="paper", y=median + 2 * spread, yref="y", xanchor="right", yanchor="bottom",
        text=f"band: class median ± 2 × the median spread between runs ({100 * spread:.2f} pt)",
        showarrow=False, font={"size": 12, "color": MUTED},
    )  # fmt: skip
    fig.update_yaxes(title=f"handed-out baseline, {metric}")
    fig.update_xaxes(categoryorder="array", categoryarray=list(base.student))
    return style(fig, 380)


def fig_baseline_vs_final(cmp: pd.DataFrame, metric: str = METRIC) -> go.Figure:
    """Per student's study: the baseline and the final, each with its spread between
    runs. A gain inside the spread is drawn as no gain."""
    fig = go.Figure()
    if cmp.empty:
        return style(fig, 200)
    cmp = cmp.sort_values("gain_pt").reset_index(drop=True)
    labels = [plain(f"{r.student} · {r.study}") for r in rows(cmp)]
    for i, r in enumerate(rows(cmp)):
        real = (not math.isnan(r.sigma)) and r.sigma >= 2
        color = GAIN if real else (LOSS if r.gain_pt < 0 and r.sigma <= -2 else MUTED)
        fig.add_scatter(
            x=[r.base, r.final], y=[labels[i]] * 2, mode="lines",
            line={"color": color, "width": 3}, showlegend=False, hoverinfo="skip",
        )  # fmt: skip
    for key, std, name, color in (
        ("base", "base_std", "baseline", ROLE_COLORS["baseline"]),
        ("final", "final_std", "final", ROLE_COLORS["final"]),
    ):
        fig.add_scatter(
            x=cmp[key], y=labels, mode="markers", name=name,
            marker={"size": 11, "color": color},
            error_x={"type": "data", "array": cmp[std].fillna(0), "color": color, "thickness": 2},
            customdata=cmp[["gain_pt", "sigma"]],
            hovertemplate="%{y}<br>%{x:.4f}<br>gain %{customdata[0]:+.2f} pt, %{customdata[1]:+.1f} σ<extra></extra>",
        )  # fmt: skip
    fig.update_xaxes(title=f"{metric} (bars: spread between runs)")
    return style(fig, max(260, 34 * len(cmp) + 120))


def parcoords(
    trials: pd.DataFrame, axes: list[str], color: str = "score", metric: str = METRIC
) -> go.Figure:
    dims = []
    for axis in axes:
        col = trials.get(f"p_{axis}")
        if col is None or col.isna().all():
            continue
        if axis in CATEGORICAL or col.dtype == object or col.dtype == bool:
            cats = sorted(col.dropna().astype(str).unique())
            codes = col.astype(str).map({c: i for i, c in enumerate(cats)})
            dims.append(
                {
                    "label": axis, "values": codes,
                    "tickvals": list(range(len(cats))), "ticktext": cats,
                    "range": [-0.2, max(len(cats) - 0.8, 0.8)],
                }
            )  # fmt: skip
        elif axis in ("lr", "hidden_dim") and (col.dropna() > 0).all():
            dims.append(
                {"label": f"{axis} (log)", "values": np.log10(col.clip(lower=1e-12))}
            )
        else:
            dims.append({"label": axis, "values": col})
    dims.append({"label": metric, "values": trials[color]})
    fig = go.Figure(
        go.Parcoords(
            line={
                "color": trials[color], "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": metric, "thickness": 12},
            },
            dimensions=dims,
        )
    )  # fmt: skip
    return style(fig, 430, margin={"l": 70, "r": 70, "t": 60, "b": 30})


def fig_axes(
    trials: pd.DataFrame, baseline_median: float | None, metric: str = METRIC
) -> go.Figure:
    """The metric against every searched axis of the pooled handed-out search: a dot per
    trial, and the median per bin or per value."""
    axes = [a for a in AXES if f"p_{a}" in trials and trials[f"p_{a}"].nunique() > 1]
    n_cols = 4
    n_rows = max(1, -(-len(axes) // n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=axes, shared_yaxes=True,
                        horizontal_spacing=0.025, vertical_spacing=0.16)  # fmt: skip
    for i, axis in enumerate(axes):
        row, col = divmod(i, n_cols)
        x = trials[f"p_{axis}"]
        if axis in CATEGORICAL:
            x = x.astype(str)
            fig.add_scatter(
                x=x, y=trials.score, mode="markers", showlegend=False,
                marker={"size": 5, "color": "#1f5fa8", "opacity": 0.25},
                row=row + 1, col=col + 1,
            )  # fmt: skip
            med = trials.groupby(x).score.median()
            fig.add_scatter(
                x=med.index, y=med.values, mode="markers", showlegend=False,
                marker={"size": 13, "color": "white", "line": {"color": INK, "width": 2}},
                row=row + 1, col=col + 1,
            )  # fmt: skip
        else:
            xs = pd.to_numeric(x)
            log = axis in ("lr", "hidden_dim")
            fig.add_scatter(
                x=xs, y=trials.score, mode="markers", showlegend=False,
                marker={"size": 5, "color": "#1f5fa8", "opacity": 0.25},
                row=row + 1, col=col + 1,
            )  # fmt: skip
            key = np.log10(xs) if log else xs
            if key.nunique() > 8:
                bins = pd.cut(key, 6)
                mid = key.groupby(bins, observed=True).mean()
                med = trials.score.groupby(bins, observed=True).median()
                fig.add_scatter(
                    x=(10 ** mid if log else mid), y=med, mode="lines+markers",
                    showlegend=False, line={"color": INK, "width": 2},
                    marker={"size": 7, "color": INK}, row=row + 1, col=col + 1,
                )  # fmt: skip
            else:
                med = trials.score.groupby(xs).median()
                fig.add_scatter(
                    x=med.index, y=med.values, mode="lines+markers", showlegend=False,
                    line={"color": INK, "width": 2}, marker={"size": 7, "color": INK},
                    row=row + 1, col=col + 1,
                )  # fmt: skip
            if log:
                fig.update_xaxes(type="log", row=row + 1, col=col + 1)
        if baseline_median is not None:
            fig.add_hline(
                y=baseline_median,
                line_dash="dot",
                line_color=MUTED,
                row=row + 1,
                col=col + 1,
            )
    fig.update_yaxes(title_text=metric, col=1)
    return style(fig, 250 * n_rows + 60)


def fig_studies(df: pd.DataFrame, cmp: pd.DataFrame, metric: str = METRIC) -> go.Figure:
    """One small panel per student's study: the metric of its trials in the order they
    ran, the baseline with its band of two standard errors, the final as a star."""
    if cmp.empty:
        return style(go.Figure(), 200)
    n = len(cmp)
    n_cols = min(3, n)
    n_rows = -(-n // n_cols)
    titles = [plain(f"{r.student} · {r.study}") for r in rows(cmp)]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, shared_yaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.18)  # fmt: skip
    for i, r in enumerate(rows(cmp)):
        row, col = divmod(i, n_cols)
        rr, cc = row + 1, col + 1
        group = df[
            (df.student == r.student) & (df.study == r.study) & (df.role == "trial")
        ]
        group = group.sort_values("start").reset_index(drop=True)
        kept = group[group.status == "COMPLETE"]
        stopped = group[group.status == "PRUNED"]
        se = 2 * (r.base_std or 0) / math.sqrt(3)
        fig.add_hrect(y0=r.base - se, y1=r.base + se, fillcolor=MUTED, opacity=0.18,
                      line_width=0, row=rr, col=cc)  # fmt: skip
        fig.add_hline(y=r.base, line_color=MUTED, line_width=1, row=rr, col=cc)
        fig.add_scatter(x=stopped.index, y=stopped.score, mode="markers", showlegend=False,
                        marker={"size": 6, "color": "#c8ced8", "symbol": "x"}, row=rr, col=cc)  # fmt: skip
        fig.add_scatter(x=kept.index, y=kept.score, mode="markers", showlegend=False,
                        marker={"size": 8, "color": ROLE_COLORS["trial"],
                                "line": {"color": "#4d79a8", "width": 1}}, row=rr, col=cc)  # fmt: skip
        fig.add_scatter(x=[len(group)], y=[r.final], mode="markers", showlegend=False,
                        error_y={"type": "data", "array": [r.final_std or 0], "color": ROLE_COLORS["final"]},
                        marker={"size": 15, "color": ROLE_COLORS["final"], "symbol": "star"},
                        row=rr, col=cc)  # fmt: skip
    fig.update_xaxes(title_text="trial", showticklabels=True)
    fig.update_yaxes(title_text=metric, col=1)
    return style(fig, 300 * n_rows + 40)


COURSE_TONE = "#8a8f98"
# One hue per student's ideas; green and red are kept for the result.
AUTHOR_COLORS = [
    "#1f5fa8", "#d98e3b", "#7a5aa6", "#2b8fa3", "#c2618f", "#8a7a2a", "#4f6fd0",
    "#a0522d", "#5b6b8c", "#b07ac4", "#3d7a8a", "#c98a1b", "#6d4c9f", "#a35d7a",
    "#2f6f9f",
]  # fmt: skip
VERDICT_STYLE = {  # border colour, border width, mark
    "better": (GAIN, 2.5, "▲"),
    "worse": (LOSS, 2.5, "▼"),
    "inside": (MUTED, 1.5, "≈"),
    "unmeasured": ("#c9ced6", 1.0, ""),
}
ARROW_STYLE = {  # colour, width
    "better": (GAIN, 3.0),
    "worse": (LOSS, 3.0),
    "inside": ("#9aa1ab", 1.8),
    "unmeasured": ("#9aa1ab", 1.8),
}
SELECTED = "#b0452f"
LAYER_PX, BOX_PX = 280, 230  # the width of a column, and of a box in it
ROW_PX, BOX_H_PX = 150, 116  # the height of a row, and of a box in it
HEAD_PX = 30  # the column titles above the boxes
PAD_PX = 3  # room for a selected box's border


def signed(value: float, digits: int = 1) -> str:
    """`value` with its sign, a minus sign rather than a hyphen; no sign on a value that
    rounds to zero."""
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        return f"{0:.{digits}f}"
    return text.replace("-", "−")


def tint(color: str, alpha: float) -> str:
    """`color` (#rrggbb) mixed with white: `alpha` of the colour."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    mix = (round(255 - alpha * (255 - c)) for c in (r, g, b))
    return "#" + "".join(f"{c:02x}" for c in mix)


def author_colors(ids: pd.DataFrame) -> dict[str, str]:
    """The colour of every author of an idea; the course's root in a neutral tone."""
    authors = sorted({a for a in ids.author if a != COURSE})
    colors = {a: AUTHOR_COLORS[i % len(AUTHOR_COLORS)] for i, a in enumerate(authors)}
    return {COURSE: COURSE_TONE, **colors}


def idea_layout(ids: pd.DataFrame) -> dict[str, tuple[int, float]]:
    """(layer, row) of every idea and of every parent that is not submitted: a root in
    layer 0, an idea one layer right of its deepest parent, a parent that is not
    submitted in the layer left of its first child. Inside a layer the ideas sit in the
    order of their parents' rows, and every layer is centred on the widest one."""
    parents = dict(zip(ids.name, ids.parents, strict=True))
    depth: dict[str, int] = {}

    def depth_of(name: str, seen: frozenset[str] = frozenset()) -> int:
        if name not in depth:
            known = [p for p in parents[name] if p in parents and p not in seen]
            depth[name] = 1 + max(
                (depth_of(p, seen | {name}) for p in known), default=-1
            )
            if any(p not in parents for p in parents[name]):
                depth[name] = max(depth[name], 1)
        return depth[name]

    for name in parents:
        depth_of(name)
    for name, ps in parents.items():
        for p in ps:
            if p not in parents:
                depth[p] = min(depth.get(p, depth[name] - 1), depth[name] - 1)
    authors = dict(zip(ids.name, ids.author, strict=True))
    layers: dict[int, list[str]] = {}
    for name, d in depth.items():
        layers.setdefault(d, []).append(name)
    widest = max((len(v) for v in layers.values()), default=0)
    place: dict[str, tuple[int, float]] = {}
    children: dict[str, list[str]] = {}
    for name, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(name)
    for d in sorted(layers):

        def key(name: str) -> tuple[float, bool, str, str]:
            # a parent not submitted has no parents of its own: it sits by its children
            near = [place[p][1] for p in parents.get(name, []) if p in place]
            if name not in parents:
                near = [place[c][1] for c in children[name] if c in place]
            centre = sum(near) / len(near) if near else -1.0
            author = authors.get(name, "~")
            return centre, author != COURSE, author, name

        group = sorted(layers[d], key=key)
        targets = [key(name)[0] for name in group]
        ys: list[float] = []
        if all(t < 0 for t in targets):
            offset = (widest - len(group)) / 2
            ys = [offset + i for i in range(len(group))]
        else:
            # by its parents, one row apart, the group centred on where it wants to be
            for t in targets:
                ys.append(max(t, ys[-1] + 1) if ys else max(t, 0.0))
            known = [t for t in targets if t >= 0]
            shift = sum(known) / len(known) - sum(ys) / len(ys)
            ys = [y + shift for y in ys]
            ys = [y - min(0.0, ys[0]) for y in ys]
        for name, y in zip(group, ys, strict=True):
            place[name] = (d, y)
    return place


def cut(text: str, width: int, lines: int) -> list[str]:
    """`text` wrapped at `width` characters, at most `lines` lines; cut with an ellipsis."""
    wrapped = textwrap.wrap(text, width) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1][: width - 1].rstrip() + "…"
    return wrapped


def name_lines(name: str, width: int = 27) -> list[str]:
    """An idea's name on one line, or on two split at a hyphen when it is longer than
    `width`: a name is typed by a classmate, so it is never cut."""
    if len(name) <= width:
        return [name]
    cuts = [i for i, c in enumerate(name) if c == "-"]
    at = min(cuts, key=lambda i: abs(i - len(name) / 2))
    return [name[: at + 1], name[at + 1 :]]


def score_line(r: SimpleNamespace) -> tuple[str, str]:
    """The last line of an idea's box, and its colour: the winner against its own
    baseline, in points and sigmas."""
    if r.verdict == "unmeasured" and math.isnan(number(r.gain_pt)):
        return "no winner against a baseline yet", MUTED
    border, _, mark = VERDICT_STYLE[r.verdict]
    sig = "" if math.isnan(number(r.sigma)) else f" · {r.sigma:.1f}σ"
    text = f"{r.final:.3f} · {mark} {signed(r.gain_pt)} pt{sig}".replace("·  ", "· ")
    return text, (border if r.verdict in ("better", "worse") else MUTED)


def fig_ideas(
    ids: pd.DataFrame,
    eds: pd.DataFrame | None = None,
    elsewhere: Mapping[str, int] | None = None,
    root_score: str = "",
) -> go.Figure:
    """The ideas of the class as a graph: one box per idea, an arrow from each parent to
    the idea that builds on it, in columns by the longest path from the course's idea on
    the left (`idea_layout`). A box holds the idea's name, author, words, what its author
    expected, and its winner against its own baseline (`root_score` for the course's
    idea); its colour says whose idea it is, its border what the winner gave. An arrow
    carries the number of `edges`. A parent that is not among the submissions is an
    open, dashed box; `elsewhere` names the lesson of a parent that is an idea of
    another lesson. Every box answers a click with its name (the markers' customdata),
    and the annotation `selected` waits, hidden, for the page to place it."""
    elsewhere = elsewhere or {}
    eds = edges(ids) if eds is None else eds
    fig = go.Figure()
    if ids.empty:
        return style(fig, 160)
    place = idea_layout(ids)
    colors = author_colors(ids)
    n_layers = 1 + max(d for d, _ in place.values())
    n_rows = max(math.ceil(y + 1) for _, y in place.values())
    bw, bh = BOX_PX / LAYER_PX, BOX_H_PX / ROW_PX
    px, py_ = 1 / LAYER_PX, 1 / ROW_PX  # one pixel across, one down
    submitted = set(ids.name)

    def box(name: str, **kw) -> None:
        d, y = place[name]
        fig.add_shape(
            type="rect", x0=d - bw / 2, x1=d + bw / 2, y0=y - bh / 2, y1=y + bh / 2,
            name=name, **kw,
        )  # fmt: skip

    def text_in(name: str, lines: list[str], color: str = INK) -> None:
        d, y = place[name]
        fig.add_annotation(
            x=d - bw / 2 + 14 * px, y=y - bh / 2 + 9 * py_, xanchor="left",
            yanchor="top", text="<br>".join(lines), align="left", showarrow=False,
            font={"size": 11.5, "color": color},
        )  # fmt: skip

    for d in range(n_layers):
        fig.add_annotation(
            x=d - bw / 2, y=1, xref="x", yref="paper", xanchor="left", yanchor="bottom",
            text="HANDED OUT" if d == 0 else f"ROUND {d}", showarrow=False,
            font={"size": 12, "color": MUTED},
        )  # fmt: skip
    for e in rows(eds):
        (pd_, py), (cd, cy) = place[e.parent], place[e.child]
        color, width = ARROW_STYLE[e.verdict if e.number else "unmeasured"]
        fig.add_annotation(
            x=cd - bw / 2, y=cy, ax=pd_ + bw / 2, ay=py, xref="x", yref="y",
            axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
            arrowwidth=width, arrowcolor=color, text="", name=f"{e.parent}->{e.child}",
            standoff=2,
        )  # fmt: skip
        if e.number:
            mark = VERDICT_STYLE[e.verdict][2]
            sig = "" if math.isnan(e.sigma) else f", {e.sigma:+.1f} σ"
            fig.add_annotation(
                x=(pd_ + bw / 2 + cd - bw / 2) / 2, y=(py + cy) / 2, yshift=10,
                text=f"<b>{mark} {signed(e.diff_pt)}</b>".replace("<b> ", "<b>"),
                showarrow=False, bgcolor="rgba(255,255,255,0.9)",
                font={"size": 11.5, "color": color if e.verdict != "inside" else MUTED},
                hovertext=plain(f"{e.diff_pt:+.2f} pt on its parent's winner{sig}"),
                name=f"label:{e.parent}->{e.child}",
            )  # fmt: skip
    for r in rows(ids):
        color = colors.get(r.author, COURSE_TONE)
        border, width, _ = VERDICT_STYLE[r.verdict]
        d, y = place[r.name]
        box(r.name, fillcolor=tint(color, 0.14), line={"color": border, "width": width})
        fig.add_shape(
            type="rect", x0=d - bw / 2, x1=d - bw / 2 + 5 * px,
            y0=y - bh / 2, y1=y + bh / 2, fillcolor=color, line_width=0,
            name=f"stripe:{r.name}",
        )  # fmt: skip
        names = name_lines(r.name)
        words = cut(r.idea, 34, 2)
        room = 7 - len(names) - len(words) - 2
        if r.author == COURSE:
            said = []
            last, last_color = root_score, MUTED
        else:
            said = (
                cut(f"expected: {r.expects}", 35, min(2, room))
                if r.expects
                else ["expected: not said"]
            )
            last, last_color = score_line(r)
        text_in(
            r.name,
            [
                *(f"<b>{html.escape(n)}</b>" for n in names),
                f"<span style='color:{color}'><b>{html.escape(r.author)}</b></span>",
                *(html.escape(w) for w in words),
                *(f"<i style='color:{MUTED}'>{html.escape(s)}</i>" for s in said),
                f"<b style='color:{last_color}'>{html.escape(last)}</b>",
            ],
        )
    for name in place:
        if name in submitted:
            continue
        box(
            name, fillcolor="white", line={"color": MUTED, "width": 1.2, "dash": "dash"}
        )
        where = (
            f"an idea of lesson {elsewhere[name]}"
            if name in elsewhere
            else "not among the submissions"
        )
        text_in(
            name,
            [
                *(f"<b>{html.escape(n)}</b>" for n in name_lines(name)),
                f"<i>{where}</i>",
            ],
            MUTED,
        )
    fig.add_annotation(
        x=0, y=0, yshift=8, yanchor="bottom", text="<b>selected</b>", showarrow=False,
        font={"size": 11, "color": SELECTED}, visible=False, name="selected",
    )  # fmt: skip
    # invisible marks across each box carry the hover and the click
    hover = []
    for r in rows(ids):
        if r.author == COURSE:
            res = "handed out: its baselines and its search are further down"
        elif r.verdict == "unmeasured":
            res = "no baseline against final measured"
        else:
            res = (
                f"{r.study}: final {r.final:.4f} against baseline {r.base:.4f}, "
                f"{r.gain_pt:+.2f} pt, {r.sigma:+.1f} σ ({r.verdict})"
            )
        parts = [
            f"<b>{html.escape(r.name)}</b> · {html.escape(r.author)}",
            "<br>".join(html.escape(w) for w in textwrap.wrap(r.idea, 50)),
            "builds on: " + (plain(", ".join(r.parents)) or "nothing (a root)"),
            f"ran by {len(r.students)}: " + plain(", ".join(r.students)),
            f"{r.n_runs} runs",
            html.escape(res),
            "click: show this idea in the window below",
        ]
        hover.append((r.name, "<br>".join(parts)))
    for name in place:
        if name not in submitted:
            hover.append(
                (
                    name,
                    f"<b>{html.escape(name)}</b><br>an idea of lesson {elsewhere[name]}"
                    if name in elsewhere
                    else f"<b>{html.escape(name)}</b><br>named as a parent, not submitted",
                )
            )
    marks = [
        (place[n][0] + dx * bw, place[n][1] + dy * bh, text, n)
        for n, text in hover
        for dx in (-0.36, -0.12, 0.12, 0.36)
        for dy in (-0.22, 0.22)
    ]
    xs, ys, texts, ns = zip(*marks, strict=True)
    fig.add_scatter(
        x=xs, y=ys, mode="markers", text=texts, customdata=ns,
        hovertemplate="%{text}<extra></extra>",
        marker={"size": 54, "symbol": "square", "color": "rgba(0,0,0,0)"},
        showlegend=False,
    )  # fmt: skip
    pad = (bw / 2) + PAD_PX * px
    fig.update_xaxes(range=[-pad, n_layers - 1 + pad], visible=False, fixedrange=True)
    fig.update_yaxes(range=[n_rows - 0.5, -0.5], visible=False, fixedrange=True)
    width = (n_layers - 1) * LAYER_PX + BOX_PX + 2 * PAD_PX
    margin = {"l": 0, "r": 0, "t": HEAD_PX, "b": 6}
    return style(
        fig, ROW_PX * n_rows + HEAD_PX + 6, width=width, margin=margin,
        hovermode="closest", hoverlabel={"align": "left"}, dragmode=False,
    )  # fmt: skip


# ---- the window: one experiment, or the whole class ------------------------------------


def idea_trials(df: pd.DataFrame, ids: pd.DataFrame, name: str) -> pd.DataFrame:
    """The trials of the study that stands for the idea `name` (the pooled handed-out
    search for the course's idea), in the order they ran, with a column `winner`: True
    for the finished trial with the best score."""
    [r] = rows(ids[ids.name == name])
    if df.empty:
        return df.assign(winner=pd.Series(dtype=bool))
    trial = (df.idea_name == name) & (df.role == "trial")
    if not isinstance(r.study, str):  # the course's idea has no study of its own
        sel = df[trial & (df.origin == "notebook")]
    else:
        sel = df[trial & (df.student == r.student) & (df.study == r.study)]
    sel = sel.sort_values("start").reset_index(drop=True)
    done = sel[(sel.status == "COMPLETE") & sel.score.notna()]
    winner = done.score.idxmax() if len(done) else None
    return sel.assign(winner=sel.index == winner)


def class_trials(df: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """The trials of every student's idea (`idea_trials`), with the idea's name and
    author as `idea_name` and `author`."""
    parts = [
        idea_trials(df, ids, r.name).assign(author=r.author)
        for r in rows(ids)
        if r.author != COURSE
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return df.iloc[:0].assign(
            winner=pd.Series(dtype=bool), author=pd.Series(dtype=object)
        )
    return pd.concat(parts, ignore_index=True)


def shared_axes(ids: pd.DataFrame) -> list[str]:
    """The axes every student's idea searched, in the order the first one lists them."""
    lists = [r.axes for r in rows(ids) if r.author != COURSE and r.axes]
    if not lists:
        return []
    return [a for a in lists[0] if all(a in other for other in lists[1:])]


def gain_rows(
    ids: pd.DataFrame, eds: pd.DataFrame, name: str | None = None
) -> list[dict]:
    """The bars of the gain picture, top to bottom. For the idea `name`: its winner
    against its own baseline, then against the winner of each parent the course did not
    hand out, then every other arrow of the class, `faded`. For the class (`name` None):
    every arrow that carries a number. Each bar: `label`, `value` and `band` (two
    standard errors) in points, `verdict`, `text`, and `why` for a bar without a value."""
    out = []

    def arrow(e: SimpleNamespace, label: str, faded: bool) -> dict:
        mark = VERDICT_STYLE[e.verdict][2]
        return {
            "label": label, "value": e.diff_pt if e.number else np.nan,
            "band": e.band_pt, "verdict": e.verdict, "faded": faded,
            "text": f"{mark} {signed(e.diff_pt)}".strip() if e.number else "",
            "why": "" if e.number else e.why,
        }  # fmt: skip

    by = {r.name: r for r in rows(ids)}
    mine = []
    if name is not None:
        r = by[name]
        if r.author != COURSE:
            band = 200 * standard_error(
                (r.base_std, r.base_n), (r.final_std, r.final_n)
            )
            sig = "" if math.isnan(number(r.sigma)) else f" · {r.sigma:.1f}σ"
            measured = not math.isnan(number(r.gain_pt))
            out.append(
                {
                    "label": "vs its baseline",
                    "value": r.gain_pt if measured else np.nan,
                    "band": band, "verdict": r.verdict, "faded": False,
                    "text": f"{signed(r.gain_pt)}{sig}" if measured else "",
                    "why": "" if measured else "no winner against a baseline yet",
                }
            )  # fmt: skip
        mine = [e for e in rows(eds) if e.child == name and e.parent_author != COURSE]
        for e in mine:
            parent = by.get(e.parent)
            label = (
                f"vs {parent.author}"
                if parent is not None
                else f"vs {cut(e.parent, 16, 1)[0]}"
            )
            bar = arrow(e, label, False)
            if parent is None:
                bar["why"] = "not submitted"
            out.append(bar)
    shown = {(e.parent, e.child) for e in mine}
    for e in rows(eds):
        if e.number and (e.parent, e.child) not in shown:
            out.append(
                arrow(e, f"{e.parent_author} → {e.child_author}", name is not None)
            )
    return out


def fits(value: float, reach: float, text: str, plot_px: float = 160) -> bool:
    """Whether `text` fits inside a bar of `value` on an axis from -`reach` to `reach`
    that is at least `plot_px` wide."""
    return abs(value) / (2 * reach) * plot_px > 6.5 * len(text) + 6


def room(values: list[float], bars: list[dict], reach: float) -> list[float]:
    """The x range of the gain picture: -`reach` to `reach`, wider on the side where a
    label beside its bar needs the room (on a plot area of 160 px)."""
    low, high = -reach, reach
    for v, b in zip(values, bars, strict=True):
        text = b["text"] or b["why"]
        k = (6.5 * len(text) + 8) / 160
        if fits(v, reach, b["text"]) or k >= 0.8:
            continue
        if v >= 0:
            high = max(high, (v - k * low) / (1 - k))
        else:
            low = min(low, (v - k * high) / (1 - k))
    return [low, high]


def fig_gain(bars: list[dict], note: str = "") -> go.Figure:
    """`gain_rows` as horizontal bars in points, each on the grey band of its spread:
    a bar inside the band is no gain."""
    fig = go.Figure()
    n = len(bars)
    for i, b in enumerate(bars):
        band = number(b["band"])
        if not math.isnan(band):
            fig.add_shape(
                type="rect", x0=-band, x1=band, y0=i - 0.42, y1=i + 0.42, layer="below",
                fillcolor="#eef0f3", line_width=0,
            )  # fmt: skip
    color = {"better": GAIN, "worse": LOSS}
    values = [number(b["value"]) for b in bars]
    fills = [
        tint(color.get(b["verdict"], "#9aa1ab"), 0.3 if b["faded"] else 1.0)
        for b in bars
    ]
    finite = [abs(v) for v in values if not math.isnan(v)]
    finite += [number(b["band"]) for b in bars if not math.isnan(number(b["band"]))]
    reach = max(finite, default=1.0) * 1.5 + 0.6
    shown = [0 if math.isnan(v) else v for v in values]
    fig.add_bar(
        x=shown, y=list(range(n)),
        orientation="h", marker_color=fills, width=0.55,
        text=[b["text"] or b["why"] for b in bars],
        textposition=[
            "inside" if fits(v, reach, b["text"]) else "outside"
            for v, b in zip(shown, bars, strict=True)
        ],
        insidetextanchor="end", textangle=0, constraintext="none",
        textfont={"size": 11},
        outsidetextfont={"size": 11, "color": [MUTED if b["faded"] else INK for b in bars]},
        insidetextfont={"size": 11, "color": "white"},
        hovertemplate="%{text}<extra></extra>", cliponaxis=False,
    )  # fmt: skip
    if n:
        fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_yaxes(
        tickvals=list(range(n)),
        ticktext=[plain(b["label"]) for b in bars],
        range=[n - 0.5, -0.5] if n else [0.5, -0.5],
        showgrid=False, showline=False,
    )  # fmt: skip
    fig.update_xaxes(
        range=room(shown, bars, reach),
        nticks=5,
        tickfont={"size": 10},
        title_text="points",
        title_font={"size": 11},
        title_standoff=2,
    )
    if not n:
        empty_note(fig, note or "no arrow carries a number yet")  # the page's own text
    ticks = [len(b["label"]) for b in bars] or [10]
    left = min(110, 6 * max(ticks) + 10)
    return style(
        fig, 236, margin={"l": left, "r": 10, "t": 6, "b": 40}, showlegend=False
    )


def with_curve(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials of `trials` that logged a curve."""
    if trials.empty or "curve" not in trials:
        return trials.iloc[:0]
    return trials[
        trials.curve.map(lambda c: isinstance(c, list) and len(c) > 0).astype(bool)
    ]


def curve_trace(trials: pd.DataFrame) -> tuple[list, list]:
    """The curves of `trials` as one line with gaps: x the epoch from 1, y the metric."""
    xs: list = []
    ys: list = []
    for c in trials.curve:
        xs += [*range(1, len(c) + 1), None]
        ys += [*c, None]
    return xs, ys


def fig_curves(
    trials: pd.DataFrame,
    color: str,
    baseline: tuple[float, float] | None,
    metric: str = METRIC,
) -> go.Figure:
    """The metric after every epoch of every trial in `trials`: the winner bold in
    `color`, a pruned trial ending in a cross, the baseline with its spread as a band."""
    fig = go.Figure()
    has = with_curve(trials)
    if baseline is not None and len(has):
        add_baseline(fig, baseline)
    rest = has[~has.winner]
    xs, ys = curve_trace(rest)
    fig.add_scatter(
        x=xs, y=ys, mode="lines", line={"color": "#9db7d5", "width": 1.2},
        opacity=0.85, hoverinfo="skip", showlegend=False,
    )  # fmt: skip
    stopped = has[has.status == "PRUNED"]
    fig.add_scatter(
        x=[len(c) for c in stopped.curve], y=[c[-1] for c in stopped.curve],
        mode="markers", marker={"symbol": "x-thin", "size": 8, "color": "#6a7280",
                                "line": {"width": 1.5, "color": "#6a7280"}},
        hovertemplate="pruned after epoch %{x}<extra></extra>", showlegend=False,
    )  # fmt: skip
    best = has[has.winner]
    xs, ys = curve_trace(best)
    fig.add_scatter(
        x=xs, y=ys, mode="lines+markers", line={"color": color, "width": 3},
        marker={"size": 4, "color": color},
        hovertemplate="the winner, epoch %{x}: %{y:.4f}<extra></extra>",
        showlegend=False,
    )  # fmt: skip
    if has.empty:
        empty_note(fig, "no trial of this study logged its curves")
    fig.update_xaxes(title_text="epoch", title_standoff=2, title_font={"size": 11})
    return style(fig, 236, margin={"l": 46, "r": 10, "t": 6, "b": 40}, showlegend=False)


def fig_class_curves(
    trials: pd.DataFrame,
    colors: Mapping[str, str],
    baseline: tuple[float, float] | None,
) -> go.Figure:
    """The winning trial's curve of every student's idea, in its author's colour."""
    fig = go.Figure()
    won = with_curve(trials)
    won = won[won.winner.astype(bool)]
    if baseline is not None and len(won):
        add_baseline(fig, baseline)
    for r in rows(won):
        fig.add_scatter(
            x=list(range(1, len(r.curve) + 1)), y=r.curve, mode="lines",
            line={"color": colors.get(r.author, COURSE_TONE), "width": 2},
            name=plain(f"{r.author} · {r.idea_name}"),
            hovertemplate="%{fullData.name}<br>epoch %{x}: %{y:.4f}<extra></extra>",
        )  # fmt: skip
    if won.empty:
        empty_note(fig, "no winner logged its curves")
    fig.update_xaxes(title_text="epoch", title_standoff=2, title_font={"size": 11})
    return style(fig, 236, margin={"l": 46, "r": 10, "t": 6, "b": 40}, showlegend=False)


def add_baseline(fig: go.Figure, baseline: tuple[float, float]) -> None:
    score, spread = baseline
    spread = 0.0 if math.isnan(number(spread)) else spread
    fig.add_hrect(
        y0=score - spread, y1=score + spread, fillcolor="#8a8f98", opacity=0.16,
        line_width=0, layer="below",
    )  # fmt: skip
    fig.add_hline(
        y=score, line_color="#8a8f98", line_dash="dot", line_width=1.2,
        annotation_text="baseline ± spread", annotation_position="bottom right",
        annotation_font={"size": 10, "color": "#8a8f98"},
    )  # fmt: skip


def empty_note(fig: go.Figure, text: str) -> None:
    """`text` in the middle of a picture that has nothing to draw, without its axes."""
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, text=text,
        font={"size": 12, "color": MUTED},
    )  # fmt: skip


def scaled(trials: pd.DataFrame, axes: list[str]) -> tuple[pd.DataFrame, list[tuple]]:
    """The finished trials of `trials` on `axes` and their score, each axis scaled to
    0..1: a number from its lowest to its highest value (on a log scale when it spans
    more than a factor 20), a choice spread evenly over its values. With the scaled
    columns, per axis (name, lowest, highest) as text. An axis no trial carries is left
    out."""
    done = trials[(trials.status == "COMPLETE") & trials.score.notna()]
    out = pd.DataFrame(index=done.index)
    labels = []
    for axis in [*axes, "score"]:
        col = done.get(axis if axis == "score" else f"p_{axis}")
        if col is None or col.isna().all():
            continue
        values = pd.to_numeric(col, errors="coerce")
        if col.dtype == bool or col.dtype == object or values.isna().any():
            cats = sorted(col.dropna().astype(str).unique())
            codes = col.astype(str).map({c: i for i, c in enumerate(cats)})
            out[axis] = codes / max(len(cats) - 1, 1) if len(cats) > 1 else 0.5
            labels.append((axis, cats[0], cats[-1]))
            continue
        low, high = values.min(), values.max()
        log = low > 0 and high / low > 20
        key = np.log10(values) if log else values
        span = key.max() - key.min()
        out[axis] = (key - key.min()) / span if span > 0 else 0.5
        labels.append((axis, tick(low), tick(high)))
    return out, labels


def tick(value: float) -> str:
    """A number at the end of an axis, short: two digits below 0.01, three above."""
    return f"{value:.2g}" if abs(value) < 0.01 else f"{value:.3g}"


def fig_lines(
    trials: pd.DataFrame,
    axes: list[str],
    metric: str = METRIC,
    color: str = "#7a5aa6",
    by: Mapping[str, str] | None = None,
) -> go.Figure:
    """Parallel coordinates drawn as lines: one line per finished trial over `axes`,
    ending in its score, the winner bold in `color`. With `by` (author -> colour) every
    line takes its author's colour, as the whole class is drawn."""
    fig = go.Figure()
    if not axes:
        empty_note(fig, "the studies share no searched axis")
        return style(fig, 236, margin={"l": 16, "r": 16, "t": 24, "b": 40})
    values, labels = scaled(trials, axes)
    names = [a for a, _, _ in labels]
    if values.empty or names == ["score"]:
        empty_note(fig, "no finished trial over the searched axes")
        return style(fig, 236, margin={"l": 16, "r": 16, "t": 24, "b": 40})
    xs = list(range(len(names)))
    winner = trials.winner.reindex(values.index).fillna(False).astype(bool)
    groups: dict[str, list] = {}
    for i, v in values.iterrows():
        key = "winner" if winner[i] and by is None else (
            trials.author[i] if by is not None else "rest"
        )  # fmt: skip
        groups.setdefault(key, []).append([*v[names].tolist(), None])
    for key, lines in groups.items():
        y = [p for line in lines for p in line]
        x = (xs + [None]) * len(lines)
        if key == "winner":
            line = {"color": color, "width": 3}
        elif by is not None:
            line = {"color": by.get(key, COURSE_TONE), "width": 1.2}
        else:
            line = {"color": "#8fa9cc", "width": 1.2}
        fig.add_scatter(
            x=x, y=y, mode="lines", line=line, hoverinfo="skip", showlegend=False,
            opacity=1.0 if key == "winner" else 0.75, connectgaps=False,
        )  # fmt: skip
    if by is not None:  # the winners on top, bold
        for i, v in values[winner].iterrows():
            fig.add_scatter(
                x=xs, y=v[names].tolist(), mode="lines", showlegend=False,
                line={"color": by.get(trials.author[i], COURSE_TONE), "width": 2.6},
                hoverinfo="skip",
            )  # fmt: skip
    crowded = len(labels) > 5  # every other label a line further out
    for i, (axis, low, high) in enumerate(labels):
        step = 12 if crowded and i % 2 else 0
        fig.add_shape(
            type="line", x0=i, x1=i, y0=0, y1=1, line={"color": INK, "width": 1}
        )
        title = metric if axis == "score" else axis
        fig.add_annotation(x=i, y=1, yshift=4 + step, yanchor="bottom", text=plain(title),
                           showarrow=False, font={"size": 10, "color": INK})  # fmt: skip
        fig.add_annotation(x=i, y=0, yshift=-4 - step, yanchor="top", showarrow=False,
                           text=plain(f"{low} … {high}" if low != high else low),
                           font={"size": 9.5, "color": MUTED})  # fmt: skip
    fig.update_xaxes(visible=False, range=[-0.35, len(names) - 0.65])
    fig.update_yaxes(visible=False, range=[-0.04, 1.04])
    room = 36 if crowded else 24
    return style(
        fig, 236, margin={"l": 16, "r": 16, "t": room, "b": room}, showlegend=False
    )


def plural(n: object, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def budget_text(budget: object, n_trials: int) -> str:
    """The budget of a study in words: its minutes and its trials."""
    if isinstance(budget, dict) and is_number(budget.get("minutes")):
        minutes = float(budget["minutes"])
        trials = budget.get("n_trials", n_trials)
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes:g} {unit}, {plural(trials, 'trial')}"
    return plural(n_trials, "trial")


def panel_head(
    r: SimpleNamespace | None, color: str, trials: pd.DataFrame, n: int
) -> str:
    """The line on top of the window: which idea it shows, or the whole class."""
    if r is None:
        return (
            "<h2>The whole class</h2><span class='cap'>"
            f"{n} ideas · every arrow as a bar, the winners' curves by student, the "
            "trials of every study over the axes they share · click a box above to "
            "see one idea</span>"
        )
    where = "handed out" if r.author == COURSE else f"round {r.round}"
    said = (
        f" · <i>expected: {html.escape(r.expects)}</i>"
        if r.expects
        else (" · <i>expected: not said</i>" if r.author != COURSE else "")
    )
    pooled = " pooled over the class" if r.author == COURSE else ""
    return (
        f"<h2>{html.escape(r.name)}</h2>"
        f"<span class='who' style='color:{color}'>{html.escape(r.author)} · {where}</span>"
        f"<span class='cap'>{html.escape(r.idea)}{said} · "
        f"{html.escape(budget_text(r.budget, len(trials)))}{pooled}</span>"
    )


def baseline_of(df: pd.DataFrame) -> tuple[float, float] | None:
    """The handed-out baseline of the class: the median of the students' baselines and
    the median of their spreads; None without one."""
    base = handed_out_baselines(df)
    if base.empty:
        return None
    return float(base.score.median()), float(base["std"].median())


def panels(
    df: pd.DataFrame,
    ids: pd.DataFrame,
    eds: pd.DataFrame,
    metric: str = METRIC,
) -> dict[str, dict]:
    """What the window shows, per idea name and for the whole class (the key ""): the
    line on top (`head`, HTML) and the three pictures, gain, curves and lines."""
    colors = author_colors(ids)
    class_base = baseline_of(df)
    everyone = class_trials(df, ids)
    out = {
        "": {
            "head": panel_head(None, INK, everyone, int((ids.author != COURSE).sum())),
            "gain": fig_gain(gain_rows(ids, eds)),
            "curves": fig_class_curves(everyone, colors, class_base),
            "lines": fig_lines(everyone, shared_axes(ids), metric, by=colors),
        }
    }
    for r in rows(ids):
        color = colors.get(r.author, COURSE_TONE)
        trials = idea_trials(df, ids, r.name)
        if r.author == COURSE:
            base, bars = class_base, []
            note = (
                "handed out: each box above holds<br>its idea's gain on this baseline"
            )
        else:
            measured = not math.isnan(number(r.base))
            base, bars, note = (
                ((r.base, r.base_std) if measured else None),
                gain_rows(ids, eds, r.name),
                "",
            )
        out[r.name] = {
            "head": panel_head(r, color, trials, len(ids)),
            "gain": fig_gain(bars, note),
            "curves": fig_curves(trials, color, base, metric),
            "lines": fig_lines(trials, r.axes, metric, color),
        }
    return out


def fig_links(lk: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if lk.empty:
        return style(fig, 160)
    lk = lk[lk.found].sort_values("diff_pt").reset_index(drop=True)
    labels = [plain(f"{r.student} → {r.builds_on}") for r in rows(lk)]
    fig.add_bar(
        y=labels, x=lk.diff_pt, orientation="h",
        marker_color=[GAIN if d > 0 else LOSS for d in lk.diff_pt],
        hovertemplate="%{y}<br>%{x:+.2f} pt<extra></extra>",
    )  # fmt: skip
    fig.update_xaxes(
        title="score of the run, minus the score of the run it builds on (points)"
    )
    return style(fig, max(200, 34 * len(lk) + 100))


def fig_architectures(df: pd.DataFrame, metric: str = METRIC) -> go.Figure:
    """Score against parameters for every scored run, per dataset the biggest one: what a
    larger head costs and what it buys."""
    sel = df[(df.role.isin(["trial", "final", "baseline"])) & df.score.notna() & df.n_params.notna()
             & (df.status == "COMPLETE")]  # fmt: skip
    fig = go.Figure()
    if sel.empty:
        return style(fig, 200)
    dataset = sel.dataset.value_counts().index[0]
    sel = sel[sel.dataset == dataset]
    for role in ("trial", "baseline", "final"):
        part = sel[sel.role == role]
        fig.add_scatter(
            x=part.n_params, y=part.score, mode="markers", name=role,
            marker={"size": 9 if role != "trial" else 6, "color": ROLE_COLORS[role],
                    "opacity": 0.9 if role != "trial" else 0.45},
            text=(part.student + " · " + part["name"].astype(str)).map(plain), hovertemplate="%{text}<br>%{x:,.0f} params<br>%{y:.4f}<extra></extra>",
        )  # fmt: skip
    fig.update_xaxes(title=f"parameters (log), runs on {dataset}", type="log")
    fig.update_yaxes(title=metric)
    return style(fig, 400)


# ---- the page --------------------------------------------------------------------------


def tile(value: str, label: str) -> str:
    return f'<div class="tile"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'


def section(title: str, caption: str, fig: go.Figure | None, note: str = "") -> str:
    body = (
        ""
        if fig is None
        else fig.to_html(
            full_html=False, include_plotlyjs=False, config={"displaylogo": False}
        )
    )
    return (
        f'<section><h2>{html.escape(title)}</h2><p class="cap">{html.escape(caption)}</p>'
        f"{body}{note}</section>"
    )


def missing(text: str) -> str:
    """A section's line in place of a picture that has nothing to draw."""
    return f'<p class="missing">{html.escape(text)}</p>'


HINT = (
    "<p class='cap hint'>Click a box to filter: the window at the bottom then shows "
    "that experiment. With nothing selected it shows the whole class: every arrow, "
    "every winner's curve by student, every trial.</p>"
)
NO_HANDED_OUT = (
    "No run of the handed-out notebook was submitted: the class handed in only "
    "studies of their own, so there is nothing to show here."
)

CSS = f"""
body {{ font-family: {FONT}; color: {INK}; margin: 0; background: #f6f7f9; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 28px 24px 0; }}
h1 {{ font-size: 28px; margin: 0 0 4px; }}
h2 {{ font-size: 20px; margin: 0 0 2px; }}
section {{ background: white; border-radius: 10px; padding: 20px 22px 14px; margin: 18px 0;
          box-shadow: 0 1px 3px rgba(20,30,50,.08); }}
.cap {{ margin: 0 0 6px; color: {MUTED}; font-size: 14px; }}
.sub {{ color: {MUTED}; margin: 0 0 18px; }}
.missing {{ color: {MUTED}; font-size: 14px; background: #f6f7f9; border-radius: 6px;
           padding: 10px 14px; margin: 8px 0 4px; }}
.tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 6px; }}
.tile {{ background: white; border-radius: 10px; padding: 12px 18px; min-width: 120px;
        box-shadow: 0 1px 3px rgba(20,30,50,.08); }}
.tile b {{ display: block; font-size: 26px; }}
.tile span {{ color: {MUTED}; font-size: 13px; }}
code {{ background: #eef0f3; padding: 1px 5px; border-radius: 4px; font-size: 12.5px;
       overflow-wrap: anywhere; }}
.wide {{ overflow-x: auto; }}
.hint {{ margin: 4px 2px; }}
.study {{ border-top: 1px solid #eef0f3; padding: 12px 0 6px; }}
.study h3 {{ font-size: 15px; margin: 0; }}
.study h3 span {{ color: {MUTED}; font-weight: 500; }}
.kv {{ display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 6px 10px;
      font-size: 14px; margin-top: 8px; }}
.kv > b {{ color: {MUTED}; font-weight: 500; font-size: 13px; }}
.chips span {{ display: inline-block; background: #eef0f3; border-radius: 4px; padding: 2px 7px;
              margin: 2px 4px 2px 0; font-size: 12.5px; font-family: ui-monospace, Menlo, monospace; }}
.chips .on {{ background: #dff0e4; color: #1c6b3a; }}
.chips .off {{ background: #f1f1f1; color: #9aa0a8; text-decoration: line-through; }}
details pre {{ background: #f6f7f9; border-radius: 6px; padding: 8px 10px; font-size: 12px;
              overflow-x: auto; max-height: 320px; }}
summary {{ cursor: pointer; color: {MUTED}; font-size: 13px; }}
.dock {{ position: sticky; bottom: 0; background: white; border-top: 3px solid #9aa1ab;
        border-radius: 10px 10px 0 0; box-shadow: 0 -4px 18px rgba(20,30,50,.16);
        padding: 12px 22px 8px; margin: 18px -6px 0; z-index: 5; }}
.dock h2 {{ font-size: 16px; margin: 0; }}
.dock h3 {{ font-size: 13px; margin: 8px 0 2px; }}
.dock .head {{ display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }}
.dock .head .cap {{ margin: 0; }}
.dock .who {{ font-weight: 600; }}
.dock .x {{ margin-left: auto; color: {MUTED}; font: inherit; font-size: 13px; background: white;
           border: 1px solid #c9cdd3; border-radius: 6px; padding: 1px 9px; cursor: pointer; }}
.dock .x[hidden] {{ display: none; }}
.dock .foot {{ text-align: center; color: {MUTED}; font-size: 11px; margin: 0; }}
.three {{ display: grid; grid-template-columns: .85fr 1fr 1.15fr; gap: 22px; }}
.three > div {{ min-width: 0; }}
@media (max-width: 800px) {{
  .three {{ grid-template-columns: 1fr; }}
  .dock {{ position: static; margin: 18px 0 0; }}
  main {{ padding: 20px 16px 0; }}
  section {{ padding: 16px 14px 10px; }}
}}
"""

OPTION = re.compile(r"""@option\(\s*["']([^"']+)["']""")
ON = (True, "true", "on", "yes", 1)
OFF = (False, "false", "off", "no", 0)


def chip_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def study_notes(df: pd.DataFrame, cmp: pd.DataFrame) -> str:
    """Per student's study, under the studies picture: the searched space, the winner's
    config (the searched axes first), the options of the design (`@option` in the design
    script) on or off in the winner, the budget and the design script itself."""
    blocks = []
    for r in rows(cmp):
        group = df[(df.student == r.student) & (df.study == r.study)]
        finals = group[group.role == "final"].sort_values("start")
        if finals.empty:
            continue
        final = finals.iloc[-1]
        params = {
            k[2:]: v
            for k, v in final.items()
            if k.startswith("p_") and not (isinstance(v, float) and math.isnan(v))
            and v is not None
        }  # fmt: skip
        space = first_of(group.search_space, str)
        axes = str(first_of(group.searched_axes, str) or "").split(",")
        order = [a for a in axes if a in params] + [a for a in params if a not in axes]
        winner = "".join(
            f"<span{' class=on' if a in axes else ''}>{html.escape(a)}="
            f"{html.escape(chip_value(params[a]))}</span>"
            for a in order
        )
        script = final.design_script or first_of(group.design_script, str)
        options = list(dict.fromkeys(OPTION.findall(str(script or ""))))
        chips = []
        for o in options:
            v = params.get(o)
            state = "on" if v in ON else ("off" if v in OFF else "")
            chips.append(
                f"<span class='{state}'>{html.escape(o)}</span>"
                if state
                else f"<span>{html.escape(o)}: not in the winner's config</span>"
            )
        budget = first_of(group.budget, dict)
        n_trials = int((group.role == "trial").sum())
        spent = budget_text(budget, n_trials)
        if isinstance(budget, dict) and isinstance(budget.get("measured_on"), str):
            spent += f", the cost of a trial measured on {budget['measured_on']}"
        design = (
            f"<details><summary>design.py · {len(str(script).splitlines())} lines"
            f"</summary><pre>{html.escape(str(script))}</pre></details>"
            if script
            else "<span class='cap'>not submitted</span>"
        )
        idea = first_of(group.idea_name, str)
        blocks.append(
            f"<div class='study'><h3>{html.escape(r.student)} · {html.escape(str(r.study))}"
            + (f" <span>{html.escape(str(idea))}</span>" if idea else "")
            + "</h3><div class='kv'>"
            + f"<b>space</b><div><code>{html.escape(str(space or 'not submitted'))}</code></div>"
            + f"<b>winner</b><div class='chips'>{winner or '—'}</div>"
            + "<b>options</b><div class='chips'>"
            + ("".join(chips) or "<i class='cap'>the design marks no @option</i>")
            + "</div>"
            + f"<b>budget</b><div>{html.escape(spent)}</div>"
            + f"<b>design</b><div>{design}</div></div></div>"
        )
    return "".join(blocks)


def ideas_note(
    ids: pd.DataFrame,
    elsewhere: Mapping[str, int] | None = None,
    eds: pd.DataFrame | None = None,
    root_score: str = "",
) -> str:
    """The graph of the ideas, scrolling sideways inside its section when it is wider
    than the page; one line when no run carries an idea."""
    if ids.empty:
        return missing("No submitted run carries an idea yet.")
    fig = fig_ideas(ids, eds, elsewhere, root_score).to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="ideas-graph",
        config={"displaylogo": False, "displayModeBar": False},
    )
    return f'<div class="wide">{fig}</div>'


PANEL_NOTES = {
    "class": [
        "points · each arrow: the child's winner on its parent's · grey band: the spread",
        "each idea's winning trial, in its author's colour",
        "every trial over the axes all studies searched · bold: the winners",
    ],
    "idea": [
        "points · grey band: inside the spread · faded: the class's other arrows",
        "bold: the winner · ×: pruned",
        "one line per trial · bold: the winner",
    ],
}


def panels_json(
    ps: Mapping[str, dict], colors: Mapping[str, str], ids: pd.DataFrame
) -> str:
    """The window's data for the page's script, as JSON that is safe inside a script tag."""
    authors = dict(zip(ids.name, ids.author, strict=True))
    parts = []
    for name, p in ps.items():
        color = colors.get(authors.get(name, ""), "#9aa1ab") if name else "#9aa1ab"
        notes = PANEL_NOTES["idea" if name else "class"]
        figs = ",".join(f'"{k}":{p[k].to_json()}' for k in ("gain", "curves", "lines"))
        parts.append(
            f'{json.dumps(name)}:{{"head":{json.dumps(p["head"])},'
            f'"color":{json.dumps(color)},"notes":{json.dumps(notes)},{figs}}}'
        )
    return ("{" + ",".join(parts) + "}").replace("</", "<\\/")


DOCK = """
<div class="dock" id="dock">
<div class="head"><span id="dock-head" style="display:contents"></span>
<button class="x" id="dock-all" hidden>show the whole class ✕</button></div>
<div class="three">
<div><h3>Gain</h3><div id="panel-gain"></div><p class="foot" id="note-gain"></p></div>
<div><h3>Learning curves · {metric} per epoch</h3><div id="panel-curves"></div><p class="foot" id="note-curves"></p></div>
<div><h3>Parallel coordinates</h3><div id="panel-lines"></div><p class="foot" id="note-lines"></p></div>
</div></div>
"""

# The window's filter: a click on a box shows that idea, a second click or the button
# the whole class; the URL's #<idea-name> opens the page on that idea.
SCRIPT = """
window.addEventListener('DOMContentLoaded', function () {
  const P = JSON.parse(document.getElementById('panels').textContent);
  const CONFIG = {displaylogo: false, responsive: true, displayModeBar: false};
  const graph = document.getElementById('ideas-graph');
  const dock = document.getElementById('dock');
  const all = document.getElementById('dock-all');
  const border = {};
  if (graph && graph.layout) {
    graph.layout.shapes.forEach(function (s) {
      if (s.name && s.name.indexOf('stripe:') !== 0 && s.name in P) {
        border[s.name] = [s.line.color, s.line.width];
      }
    });
  }
  let current = null;
  function mark(name) {
    if (!graph || !graph.layout) return;
    const update = {};
    graph.layout.shapes.forEach(function (s, i) {
      if (!(s.name in border)) return;
      const on = s.name === name;
      update['shapes[' + i + '].line.color'] = on ? '#1d2330' : border[s.name][0];
      update['shapes[' + i + '].line.width'] = on ? 3 : border[s.name][1];
      if (on) {
        const k = graph.layout.annotations.findIndex(function (a) { return a.name === 'selected'; });
        update['annotations[' + k + '].x'] = (s.x0 + s.x1) / 2;
        update['annotations[' + k + '].y'] = Math.min(s.y0, s.y1);
      }
    });
    const k = graph.layout.annotations.findIndex(function (a) { return a.name === 'selected'; });
    if (k >= 0) update['annotations[' + k + '].visible'] = name !== '';
    Plotly.relayout(graph, update);
  }
  function show(name) {
    if (!(name in P)) name = '';
    if (name === current) return;
    current = name;
    const p = P[name];
    document.getElementById('dock-head').innerHTML = p.head;
    ['gain', 'curves', 'lines'].forEach(function (key, i) {
      Plotly.react('panel-' + key, p[key].data, p[key].layout, CONFIG);
      document.getElementById('note-' + key).textContent = p.notes[i];
    });
    all.hidden = name === '';
    dock.style.borderTopColor = p.color;
    mark(name);
    const url = location.pathname + location.search + (name ? '#' + encodeURIComponent(name) : '');
    try { history.replaceState(null, '', url); } catch (e) { /* a page opened from disk may refuse */ }
  }
  function fromHash() {
    let name = '';
    try { name = decodeURIComponent(location.hash.slice(1)); } catch (e) { name = ''; }
    show(name);
  }
  if (graph && graph.on) {
    graph.on('plotly_click', function (ev) {
      const name = ev.points && ev.points[0] && ev.points[0].customdata;
      if (name in P) show(name === current ? '' : name);
    });
  }
  all.addEventListener('click', function () { show(''); });
  window.addEventListener('hashchange', fromHash);
  fromHash();
});
"""


def class_tiles(
    df: pd.DataFrame, ids: pd.DataFrame, eds: pd.DataFrame, metric: str
) -> str:
    """The tiles on top: students, ideas and their rounds, how often an idea builds on a
    classmate's, runs, the best winner beside the baseline, and the metric."""
    own = ids[ids.author != COURSE]
    rounds = int(own["round"].max()) if len(own) else 0
    authors = dict(zip(ids.name, ids.author, strict=True))
    on_classmate = sum(
        1
        for e in rows(eds)
        if authors.get(e.parent) not in (None, COURSE, e.child_author)
    )
    measured = own[own.final.notna()]
    if len(measured):
        dataset = measured.dataset.value_counts().index[0]
        on = measured[measured.dataset == dataset]
        best = f"{on.final.max():.3f}"
        base = baseline_of(df[df.dataset == dataset])
        base_score = base[0] if base is not None else float(on.base.median())
        several = measured.dataset.nunique() > 1
        label = f"best so far{f' on {dataset}' if several else ''} (baseline {base_score:.3f})"
    else:
        best, label = "—", "best so far: no winner measured yet"
    return "".join(
        [
            tile(str(df.student.nunique()), "students"),
            tile(
                str(len(own)), f"ideas, in {rounds} round{'s' if rounds != 1 else ''}"
            ),
            tile(str(on_classmate), "times built on a classmate"),
            tile(str(len(df)), "runs"),
            tile(best, label),
            tile(metric, "the metric every figure reads"),
        ]
    )


def build_page(
    df: pd.DataFrame,
    lesson: int,
    source: str = "",
    metric: str = METRIC,
    others: Iterable[str] = (),
    elsewhere: Mapping[str, int] | None = None,
) -> str:
    cmp = comparisons(df)
    trials = handed_out_trials(df)
    base = handed_out_baselines(df)
    base_median = float(base.score.median()) if len(base) else None
    lk = links(df)
    ids = ideas(df, cmp)
    eds = edges(ids)
    class_base = baseline_of(df)
    root_score = (
        f"baseline {class_base[0]:.3f} ± {class_base[1]:.3f}"
        if class_base is not None
        else "no handed-out baseline submitted"
    )

    other = [m for m in others if m != metric and "/" not in m]
    per_class: dict[str, int] = {}
    for m in others:
        head, _, _ = m.partition("/")
        if "/" in m:
            per_class[head] = per_class.get(head, 0) + 1
    other += [
        f"{head} per class ({n} classes, as {head}/&lt;class&gt;)"
        for head, n in per_class.items()
    ]
    reads = (
        f"<p class='sub'>Every figure reads {html.escape(metric)}. The runs also log: "
        f"{', '.join(html.escape(m).replace('&amp;lt;', '&lt;').replace('&amp;gt;', '&gt;') for m in other)}; "
        f"read one with <code>make dashboard LESSON={lesson} METRIC=&lt;name&gt;</code>.</p>"
        if other
        else ""
    )
    no_search = (
        NO_HANDED_OUT
        if base.empty and trials.empty
        else "No trial of the handed-out search was submitted."
    )
    parts = [
        section(
            "The ideas of the class",
            "One box per idea, from left to right in the order they were built: a "
            "column is the longest path from the handed-out idea. Colour: whose idea. "
            "The number on a box: its winner against its own baseline. The number on an "
            "arrow: the child's winner against that parent's winner, in points; grey ≈ "
            "is inside the spread between runs of the two, green ▲ and red ▼ clear two "
            "sigma; no number between different datasets or metrics. Dashed: a parent "
            "that is not among the submissions. To build on an idea, name it in "
            "study.py: IDEA = Idea.of('...', builds_on='blissful-bee-18').",
            None,
            ideas_note(ids, elsewhere, eds, root_score),
        ),
        HINT,
        section(
            "Who submitted what",
            "Every submitted run, per student and role. A student with only baselines "
            "ran the handed-out part and no study of their own.",
            fig_who_submitted(df),
        ),
        section(
            "The handed-out baselines, side by side",
            "The same model on the same data, on every student's machine. Each dot is "
            "one student's baseline; the bar is the spread between their own runs. "
            "How far the dots lie apart is how large a difference has to be to mean anything.",
            fig_baselines(df, metric) if len(base) else None,
            "" if len(base) else missing(NO_HANDED_OUT),
        ),
        section(
            "Baseline against final, per study",
            "Each student's own study. Grey line: a gain inside the spread between runs "
            "(under two sigma). Green: a gain that clears it.",
            fig_baseline_vs_final(cmp, metric) if len(cmp) else None,
            ""
            if len(cmp)
            else missing("No study has both a baseline and a final submitted yet."),
        ),
        section(
            "The handed-out search, pooled over the class",
            f"{len(trials)} trials from {trials.student.nunique() if len(trials) else 0} students, every one an "
            "independent draw from the same space: one random search many times the "
            "size. One line is one trial; drag along an axis to keep a range.",
            parcoords(trials, AXES, metric=metric) if len(trials) else None,
            "" if len(trials) else missing(no_search),
        ),
        section(
            "Axis by axis",
            f"{metric} against each searched axis. Faint dots: every trial. Line or "
            "ringed dots: the median per bin or value. Dotted: the class's median baseline.",
            fig_axes(trials, base_median, metric) if len(trials) else None,
            "" if len(trials) else missing(no_search),
        ),
        section(
            "Each student's own study",
            "Trials in the order they ran (crosses: stopped early by the pruner). Grey "
            "band: the baseline ± two standard errors of its runs. Star: the final, rerun "
            "a few times. Under the picture, per study: the space it searched, the "
            "winner's config (green: the searched axes), the options of the design on or "
            "off in the winner, the budget and the design script.",
            fig_studies(df, cmp, metric) if len(cmp) else None,
            study_notes(df, cmp)
            if len(cmp)
            else missing("No study has both a baseline and a final submitted yet."),
        ),
        section(
            "Who built on whom",
            "A run that names a classmate's run in builds_on, and what it scored against that run.",
            fig_links(lk) if len(lk) else None,
            "" if len(lk) else missing("Nobody has built on a classmate's run yet."),
        ),
        section(
            "What a larger head costs and buys",
            f"Every scored run of the most common dataset: {metric} against parameters.",
            fig_architectures(df, metric),
        ),
    ]
    window = ""
    if len(ids):
        ps = panels(df, ids, eds, metric)
        window = (
            DOCK.format(metric=html.escape(metric))
            + f"<script type='application/json' id='panels'>{panels_json(ps, author_colors(ids), ids)}</script>"
            + f"<script>{SCRIPT}</script>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Lesson {lesson}: what the class did</title><style>{CSS}</style>"
        f"<script>{get_plotlyjs()}</script></head><body><main>"
        f"<h1>Lesson {lesson}: what the class did</h1>"
        f"<p class='sub'>{html.escape(source)}</p>{reads}"
        f"<div class='tiles'>{class_tiles(df, ids, eds, metric)}</div>"
        + "".join(parts)
        + window
        + "<div style='height:24px'></div></main></body></html>"
    )


# ---- reading the class's runs ----------------------------------------------------------


def ideas_elsewhere(subs: Iterable[Submission], lesson: int) -> dict[str, int]:
    """The name of every idea of another lesson than `lesson`, with its lesson: an idea
    may build on one of an earlier lesson."""
    out: dict[str, int] = {}
    for sub in subs:
        idea = sub.idea()
        if idea is not None and sub.lesson != lesson:
            out.setdefault(idea[0], sub.lesson)
    return out


def read_class(
    repo: Path, lesson: int, fetch: bool
) -> tuple[list[Submission], str, dict[str, int]]:
    """The valid submissions of `lesson` on the class repository's main, where they came
    from, and the ideas of the other lessons (`ideas_elsewhere`); invalid files are
    reported and skipped."""
    from collect_runs import class_ref, read_submission, submission_files

    ref = class_ref(repo, fetch=fetch)
    subs, others = [], []
    for rel, mode in submission_files(repo, ref, None):
        try:
            sub = read_submission(repo, ref, rel, mode)
        except ValueError as e:
            if rel.startswith(f"lesson{lesson}/"):
                print(f"skipped {rel}: " + "; ".join(explain(e)))
            continue
        (subs if sub.lesson == lesson else others).append(sub)
    source = f"{len(subs)} runs in {repo} at {ref.removeprefix('refs/remotes/')}"
    return subs, source, ideas_elsewhere(others, lesson)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lesson", type=int, default=1)
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(
            os.environ.get("MADS_RUNS_REPO", COURSE_ROOT.parent / "mads-runs")
        ),
    )
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--demo", action="store_true", help="made-up data of a class")
    parser.add_argument(
        "--out",
        type=Path,
        help="the page (default: artefacts/class-analysis/lesson<N>/index.html)",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the page in the browser"
    )
    parser.add_argument(
        "--metric",
        help="the metric every figure reads: accuracy, macro_recall, or one class of a "
        "metric per class, as recall/F (default: the objective most studies search)",
    )
    args = parser.parse_args()

    if args.demo:
        from class_demo import demo_submissions

        subs, source, elsewhere = (
            demo_submissions(args.lesson),
            "made-up data: eight invented students",
            {},
        )
    else:
        repo = args.runs_repo.resolve()
        subs, source, elsewhere = read_class(repo, args.lesson, fetch=not args.no_pull)
    if not subs:
        sys.exit(f"no runs of lesson {args.lesson} to show")
    metric = args.metric or page_metric(subs)
    known = metric_names(subs)
    if metric not in known:
        sys.exit(f"no run logs the metric {metric!r}; the runs log {known}")
    df = runs_frame(subs, metric)
    out = (
        args.out
        or COURSE_ROOT
        / "artefacts"
        / "class-analysis"
        / f"lesson{args.lesson}"
        / "index.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(df, args.lesson, source, metric, known, elsewhere))
    by_role = Counter(df.role)
    print(f"{len(df)} runs of {df.student.nunique()} students ({dict(by_role)})")
    print(f"page: {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
