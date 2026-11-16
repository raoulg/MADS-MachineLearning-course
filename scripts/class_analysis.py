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
compares scores does so per dataset.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import sys
import textwrap
import webbrowser
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_submission import COURSE, Submission, explain, idea_parents

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


def runs_frame(subs: Iterable[Submission], metric: str = METRIC) -> pd.DataFrame:
    """One row per submitted run, with the params as `p_<name>` columns."""
    rows = []
    for sub in subs:
        run = sub.run
        attrs = run.attrs
        study_attrs = run.study.attrs
        row = {
            "student": sub.student,
            "lesson": sub.lesson,
            "ref": sub.ref(),
            "experiment": run.experiment,
            "origin": origin_of(run.experiment),
            "study": study_of(run.experiment),
            "name": run.run_name,
            "role": attrs.get("role", "-"),
            "dataset": attrs.get("dataset") or "unknown",
            "session": attrs.get("session"),
            "status": run.status,
            "score": run.value,
            "std": attrs.get(f"{metric}_std"),
            "n": attrs.get("n_repeats"),
            "n_params": attrs.get("n_params"),
            "train_seconds": attrs.get("train_seconds"),
            "architecture": attrs.get("architecture"),
            "builds_on": attrs.get("builds_on", ""),
            "search_space": sub.run.study.attrs.get("search_space"),
            "searched_axes": sub.run.study.attrs.get("searched_axes"),
            "space_is_default": sub.run.study.attrs.get("space_is_default"),
            "idea": study_attrs.get("idea", ""),
            "idea_name": study_attrs.get("idea_name", ""),
            "idea_author": study_attrs.get("idea_author", ""),
            "idea_parents": study_attrs.get("idea_parents", ""),
            "epochs_run": len(run.intermediate),
            "start": run.start_time,
            "design": "design.py" in run.artifacts,
        }
        row.update({f"p_{k}": v for k, v in run.params.items()})
        # a trial's fixed axes (the dataset, ...) are in its config, not its params
        for k, v in (attrs.get("config") or {}).items():
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
                "base": b.score,
                "base_std": b["std"],
                "final": f.score,
                "final_std": f["std"],
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
    "name", "idea", "author", "parents", "missing", "students", "n_runs", "study",
    "base", "final", "gain_pt", "sigma", "verdict",
]  # fmt: skip


def verdict(sig: float | None) -> str:
    """A result at a glance: `better` or `worse` than its baseline by more than two
    sigmas, `inside` the spread between runs, or `unmeasured`."""
    if sig is None or math.isnan(sig):
        return "unmeasured"
    if sig >= 2:
        return "better"
    return "worse" if sig <= -2 else "inside"


def ideas(df: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """One row per idea of the class: its name, words, author, parents (the names it
    builds on; `missing` those not among the submissions), the students who ran it and
    their number of runs, and its result where `comparisons` has one for a study of the
    idea: the final against the baseline, in points and in sigmas. An idea is its name,
    so the handed-out idea of author `course` is one row for the whole class."""
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
        if name is not None:
            result.setdefault(name, r)
    out = []
    for name, group in carried.groupby("idea_name", sort=False):
        first = group.iloc[0]
        parents = idea_parents(first.idea_parents)
        c = result.get(str(name))
        out.append(
            {
                "name": name,
                "idea": first.idea,
                "author": first.idea_author,
                "parents": parents,
                "missing": [p for p in parents if p not in names],
                "students": sorted(group.student.unique()),
                "n_runs": len(group),
                "study": None if c is None else c.study,
                "base": np.nan if c is None else c.base,
                "final": np.nan if c is None else c.final,
                "gain_pt": np.nan if c is None else c.gain_pt,
                "sigma": np.nan if c is None else c.sigma,
                "verdict": verdict(None if c is None else c.sigma),
            }
        )
    out.sort(key=lambda r: (r["author"] != COURSE, r["author"], r["name"]))
    return pd.DataFrame(out, columns=IDEA_COLUMNS)


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


def fig_baselines(df: pd.DataFrame) -> go.Figure:
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
    fig.update_yaxes(title="handed-out baseline, accuracy")
    fig.update_xaxes(categoryorder="array", categoryarray=list(base.student))
    return style(fig, 380)


def fig_baseline_vs_final(cmp: pd.DataFrame) -> go.Figure:
    """Per student's study: the baseline and the final, each with its spread between
    runs. A gain inside the spread is drawn as no gain."""
    fig = go.Figure()
    if cmp.empty:
        return style(fig, 200)
    cmp = cmp.sort_values("gain_pt").reset_index(drop=True)
    labels = [f"{r.student} · {r.study}" for r in rows(cmp)]
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
    fig.update_xaxes(title="accuracy (bars: spread between runs)")
    return style(fig, max(260, 34 * len(cmp) + 120))


def parcoords(trials: pd.DataFrame, axes: list[str], color: str = "score") -> go.Figure:
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
    dims.append({"label": METRIC, "values": trials[color]})
    fig = go.Figure(
        go.Parcoords(
            line={
                "color": trials[color], "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": METRIC, "thickness": 12},
            },
            dimensions=dims,
        )
    )  # fmt: skip
    return style(fig, 430, margin={"l": 70, "r": 70, "t": 60, "b": 30})


def fig_axes(trials: pd.DataFrame, baseline_median: float | None) -> go.Figure:
    """Accuracy against every searched axis of the pooled handed-out search: a dot per
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
    fig.update_yaxes(title_text=METRIC, col=1)
    return style(fig, 250 * n_rows + 60)


def fig_studies(df: pd.DataFrame, cmp: pd.DataFrame) -> go.Figure:
    """One small panel per student's study: the accuracy of its trials in the order they
    ran, the baseline with its band of two standard errors, the final as a star."""
    if cmp.empty:
        return style(go.Figure(), 200)
    n = len(cmp)
    n_cols = min(3, n)
    n_rows = -(-n // n_cols)
    titles = [f"{r.student} · {r.study}" for r in rows(cmp)]
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
    fig.update_yaxes(title_text=METRIC, col=1)
    return style(fig, 300 * n_rows + 40)


COURSE_TONE = "#8a8f98"
# One hue per student's ideas; green and red are kept for the result.
AUTHOR_COLORS = [
    "#1f5fa8", "#d98e3b", "#7a5aa6", "#2b8fa3", "#c2618f", "#8a7a2a", "#4f6fd0",
    "#a0522d", "#5b6b8c", "#b07ac4", "#3d7a8a", "#c98a1b", "#6d4c9f", "#a35d7a",
    "#2f6f9f",
]  # fmt: skip
VERDICT_STYLE = {  # border colour, border width, mark
    "better": (GAIN, 3.0, "▲"),
    "worse": (LOSS, 3.0, "▼"),
    "inside": (MUTED, 1.5, "≈"),
    "unmeasured": ("#c9ced6", 1.0, ""),
}
LAYER_PX, BOX_PX = 250, 186  # the width of a layer, and of a box in it
ROW_PX, BOX_H_PX = 100, 80  # the height of a row, and of a box in it


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
        offset = (widest - len(group)) / 2
        for i, name in enumerate(group):
            place[name] = (d, offset + i)
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


def fig_ideas(ids: pd.DataFrame) -> go.Figure:
    """The ideas of the class as a graph: one box per idea, an arrow from each parent to
    the idea that builds on it, in layers by depth from the course's root on the left.
    The colour says whose idea it is, the border what it gave against its baseline. A
    parent that is not among the submissions is an open, dashed box."""
    fig = go.Figure()
    if ids.empty:
        return style(fig, 160)
    place = idea_layout(ids)
    colors = author_colors(ids)
    n_layers = 1 + max(d for d, _ in place.values())
    n_rows = max(math.ceil(y + 1) for _, y in place.values())
    bw, bh = BOX_PX / LAYER_PX, BOX_H_PX / ROW_PX
    line = 15  # px between the lines of a box's text

    def box(name: str, **kw) -> None:
        d, y = place[name]
        fig.add_shape(
            type="rect", x0=d - bw / 2, x1=d + bw / 2, y0=y - bh / 2, y1=y + bh / 2,
            name=name, **kw,
        )  # fmt: skip

    for r in rows(ids):
        color = colors.get(r.author, COURSE_TONE)
        border, width, mark = VERDICT_STYLE[r.verdict]
        d, y = place[r.name]
        box(r.name, fillcolor=tint(color, 0.14), line={"color": border, "width": width})
        fig.add_shape(
            type="rect", x0=d - bw / 2, x1=d - bw / 2 + 6 / LAYER_PX,
            y0=y - bh / 2, y1=y + bh / 2, fillcolor=color, line_width=0,
            name=f"stripe:{r.name}",
        )  # fmt: skip
        name = name_lines(r.name)
        words = cut(r.idea, 30, 3 - len(name))
        size = 12 if len(r.name) <= 22 else 10.5
        text = [
            *(
                f"<span style='font-size:{size}px'><b>{html.escape(n)}</b></span>"
                for n in name
            ),
            f"<span style='color:{color}'>{html.escape(r.author)}</span>",
            *(html.escape(w) for w in words),
        ]
        fig.add_annotation(
            x=d - bw / 2 + 12 / LAYER_PX, y=y, xanchor="left", yanchor="middle",
            text="<br>".join(text), align="left", showarrow=False,
            font={"size": 11, "color": INK},
        )  # fmt: skip
        if r.verdict != "unmeasured":
            fig.add_annotation(
                x=d + bw / 2 - 8 / LAYER_PX, y=y, xanchor="right", yanchor="middle",
                yshift=(len(text) / 2 - len(name) - 0.5) * line,  # the author's line
                text=f"<b>{mark} {r.gain_pt:+.1f} pt</b>", showarrow=False,
                font={"size": 11, "color": border if r.verdict != "inside" else MUTED},
            )  # fmt: skip
    for name in place:
        if name in set(ids.name):
            continue
        d, y = place[name]
        box(
            name, fillcolor="white", line={"color": MUTED, "width": 1.2, "dash": "dash"}
        )
        fig.add_annotation(
            x=d - bw / 2 + 12 / LAYER_PX, y=y, xanchor="left", yanchor="middle",
            text="<br>".join(f"<b>{html.escape(n)}</b>" for n in name_lines(name))
            + "<br><i>not among the submissions</i>",
            align="left", showarrow=False, font={"size": 11, "color": MUTED},
        )  # fmt: skip
    for r in rows(ids):
        for p in r.parents:
            (pd_, py), (cd, cy) = place[p], place[r.name]
            fig.add_annotation(
                x=cd - bw / 2, y=cy, ax=pd_ + bw / 2, ay=py, xref="x", yref="y",
                axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
                arrowwidth=1.4, arrowcolor="#9aa1ab", text="", name=f"{p}->{r.name}",
                standoff=2,
            )  # fmt: skip
    # invisible marks across each box carry the hover
    hover = []
    for r in rows(ids):
        d, y = place[r.name]
        if r.verdict == "unmeasured":
            res = "no baseline against final measured"
        else:
            res = (
                f"{r.study}: final {r.final:.4f} against baseline {r.base:.4f}, "
                f"{r.gain_pt:+.2f} pt, {r.sigma:+.1f} σ ({r.verdict})"
            )
        parts = [
            f"<b>{html.escape(r.name)}</b> · {html.escape(r.author)}",
            "<br>".join(html.escape(w) for w in textwrap.wrap(r.idea, 50)),
            "builds on: " + (", ".join(r.parents) or "nothing (a root)"),
            f"ran by {len(r.students)}: " + ", ".join(r.students),
            f"{r.n_runs} runs",
            html.escape(res),
        ]
        for dx in (-0.3, 0.0, 0.3):
            hover.append((d + dx * bw, y, "<br>".join(parts)))
    for name in place:
        if name not in set(ids.name):
            d, y = place[name]
            text = f"<b>{html.escape(name)}</b><br>named as a parent, not submitted"
            hover += [(d + dx * bw, y, text) for dx in (-0.3, 0.0, 0.3)]
    xs, ys, texts = zip(*hover, strict=True)
    fig.add_scatter(
        x=xs, y=ys, mode="markers", text=texts, hovertemplate="%{text}<extra></extra>",
        marker={"size": 60, "symbol": "square", "color": "rgba(0,0,0,0)"},
        showlegend=False,
    )  # fmt: skip
    fig.update_xaxes(range=[-0.5, n_layers - 0.5], visible=False, fixedrange=True)
    fig.update_yaxes(range=[n_rows - 0.5, -0.5], visible=False, fixedrange=True)
    margin = {"l": 10, "r": 10, "t": 10, "b": 10}
    return style(
        fig, ROW_PX * n_rows + 20, width=LAYER_PX * n_layers + 20, margin=margin,
        hovermode="closest", hoverlabel={"align": "left"},
    )  # fmt: skip


def fig_links(lk: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if lk.empty:
        return style(fig, 160)
    lk = lk[lk.found].sort_values("diff_pt").reset_index(drop=True)
    labels = [f"{r.student} → {r.builds_on}" for r in rows(lk)]
    fig.add_bar(
        y=labels, x=lk.diff_pt, orientation="h",
        marker_color=[GAIN if d > 0 else LOSS for d in lk.diff_pt],
        hovertemplate="%{y}<br>%{x:+.2f} pt<extra></extra>",
    )  # fmt: skip
    fig.update_xaxes(
        title="score of the run, minus the score of the run it builds on (points)"
    )
    return style(fig, max(200, 34 * len(lk) + 100))


def fig_architectures(df: pd.DataFrame) -> go.Figure:
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
            text=part.student + " · " + part["name"], hovertemplate="%{text}<br>%{x:,.0f} params<br>%{y:.4f}<extra></extra>",
        )  # fmt: skip
    fig.update_xaxes(title=f"parameters (log), runs on {dataset}", type="log")
    fig.update_yaxes(title=METRIC)
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


CSS = f"""
body {{ font-family: {FONT}; color: {INK}; margin: 0; background: #f6f7f9; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 28px 24px 80px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; }}
h2 {{ font-size: 20px; margin: 0 0 2px; }}
section {{ background: white; border-radius: 10px; padding: 20px 22px 14px; margin: 18px 0;
          box-shadow: 0 1px 3px rgba(20,30,50,.08); }}
.cap {{ margin: 0 0 6px; color: {MUTED}; font-size: 14px; }}
.sub {{ color: {MUTED}; margin: 0 0 18px; }}
.tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 6px; }}
.tile {{ background: white; border-radius: 10px; padding: 12px 18px; min-width: 120px;
        box-shadow: 0 1px 3px rgba(20,30,50,.08); }}
.tile b {{ display: block; font-size: 26px; }}
.tile span {{ color: {MUTED}; font-size: 13px; }}
code {{ background: #eef0f3; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
.space {{ font-size: 12.5px; color: {MUTED}; margin: 4px 0; }}
.wide {{ overflow-x: auto; }}
"""


def spaces_note(cmp: pd.DataFrame) -> str:
    """The search space of every study, one line each, under the studies picture."""
    lines = []
    for r in rows(cmp):
        if isinstance(r.search_space, str):
            lines.append(
                f'<div class="space"><b>{html.escape(r.student)} · {html.escape(str(r.study))}</b>'
                f" <code>{html.escape(r.search_space)}</code></div>"
            )
    return "".join(lines)


def ideas_note(ids: pd.DataFrame) -> str:
    """The graph of the ideas, scrolling sideways inside its section when it is wider
    than the page; one line when no run carries an idea."""
    if ids.empty:
        return '<p class="cap">No submitted run carries an idea yet.</p>'
    fig = fig_ideas(ids).to_html(
        full_html=False, include_plotlyjs=False, config={"displaylogo": False}
    )
    return f'<div class="wide">{fig}</div>'


def build_page(df: pd.DataFrame, lesson: int, source: str = "") -> str:
    cmp = comparisons(df)
    trials = handed_out_trials(df)
    base = handed_out_baselines(df)
    base_median = float(base.score.median()) if len(base) else None
    lk = links(df)
    ids = ideas(df, cmp)

    students = df.student.nunique()
    tiles = "".join(
        [
            tile(str(students), "students"),
            tile(str(len(df)), "runs"),
            tile(str(df.study.nunique()), "own studies"),
            tile(str(len(ids)), "ideas"),
            tile(str(len(trials)), "trials of the handed-out search"),
            tile(str(int(df.design.sum())), "runs with a design script"),
        ]
    )
    parts = [
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
            fig_baselines(df),
        ),
        section(
            "Baseline against final, per study",
            "Each student's own study. Grey line: a gain inside the spread between runs "
            "(under two sigma). Green: a gain that clears it.",
            fig_baseline_vs_final(cmp),
        ),
        section(
            "The handed-out search, pooled over the class",
            f"{len(trials)} trials from {trials.student.nunique() if len(trials) else 0} students, every one an "
            "independent draw from the same space: one random search many times the "
            "size. One line is one trial; drag along an axis to keep a range.",
            parcoords(trials, AXES) if len(trials) else None,
        ),
        section(
            "Axis by axis",
            "Accuracy against each searched axis. Faint dots: every trial. Line or "
            "ringed dots: the median per bin or value. Dotted: the class's median baseline.",
            fig_axes(trials, base_median) if len(trials) else None,
        ),
        section(
            "Each student's own study",
            "Trials in the order they ran (crosses: stopped early by the pruner). Grey "
            "band: the baseline ± two standard errors of its runs. Star: the final, rerun a few times.",
            fig_studies(df, cmp),
            spaces_note(cmp),
        ),
        section(
            "The ideas of the class",
            "One box per idea: its name, its author and its words; an arrow runs from "
            "an idea to each idea that builds on it. Colour: whose idea it is (grey: the "
            "course's). Border: green ▲ better than its baseline by more than two sigma, "
            "red ▼ worse, grey ≈ inside the spread between runs, thin: not measured. "
            "Dashed: a parent that is not among the submissions. To build on an idea, "
            "pass its name as builds_on to idea(...), as in "
            "idea('...', builds_on='blissful-bee-18').",
            None,
            ideas_note(ids),
        ),
        section(
            "Who built on whom",
            "A run that names a classmate's run in builds_on, and what it scored against that run.",
            fig_links(lk) if len(lk) else None,
            ""
            if len(lk)
            else '<p class="cap">Nobody has built on a classmate\'s run yet.</p>',
        ),
        section(
            "What a larger head costs and buys",
            "Every scored run of the most common dataset: accuracy against parameters.",
            fig_architectures(df),
        ),
    ]
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Lesson {lesson}: what the class did</title><style>{CSS}</style>"
        f"<script>{get_plotlyjs()}</script></head><body><main>"
        f"<h1>Lesson {lesson}: what the class did</h1>"
        f"<p class='sub'>{html.escape(source)}</p><div class='tiles'>{tiles}</div>"
        + "".join(parts)
        + "</main></body></html>"
    )


# ---- reading the class's runs ----------------------------------------------------------


def read_class(repo: Path, lesson: int, fetch: bool) -> tuple[list[Submission], str]:
    """The valid submissions of `lesson` on the class repository's main, and where they
    came from; invalid files are reported and skipped."""
    from collect_runs import class_ref, read_submission, submission_files

    ref = class_ref(repo, fetch=fetch)
    subs = []
    for rel, mode in submission_files(repo, ref, [lesson]):
        try:
            subs.append(read_submission(repo, ref, rel, mode))
        except ValueError as e:
            print(f"skipped {rel}: " + "; ".join(explain(e)))
    return subs, f"{len(subs)} runs in {repo} at {ref.removeprefix('refs/remotes/')}"


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
    args = parser.parse_args()

    if args.demo:
        from class_demo import demo_submissions

        subs, source = (
            demo_submissions(args.lesson),
            "made-up data: eight invented students",
        )
    else:
        repo = args.runs_repo.resolve()
        subs, source = read_class(repo, args.lesson, fetch=not args.no_pull)
    if not subs:
        sys.exit(f"no runs of lesson {args.lesson} to show")
    df = runs_frame(subs)
    out = (
        args.out
        or COURSE_ROOT
        / "artefacts"
        / "class-analysis"
        / f"lesson{args.lesson}"
        / "index.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(df, args.lesson, source))
    by_role = Counter(df.role)
    print(f"{len(df)} runs of {df.student.nunique()} students ({dict(by_role)})")
    print(f"page: {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
