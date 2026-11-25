"""The class view: one page that shows what the whole class did in a lesson.

    uv run scripts/class_analysis.py --lesson 1          # fetch main, write the page, print its path
    uv run scripts/class_analysis.py --lesson 1 --open   # ... and open it in the browser
    uv run scripts/class_analysis.py --demo              # the page on made-up data
    uv run scripts/class_analysis.py --plotly cdn        # plotly.js from cdnjs: a small page

The class's runs are the `lesson<N>/<student>/<run_id>.json` files on the class
repository's `main`; they are read with git exactly as `collect_runs.py` reads them. The
result is one self-contained HTML page, `artefacts/class-analysis/lesson<N>/index.html`,
that fills one screen: a header with the yardstick every comparison is read against, and
one view per tab. It opens on what the class investigated: per learning goal of the
lesson (`THEMES`), one picture of the class's evidence beside the students whose ideas
asked about it, each idea named by its tags (`TAGS`) and its final; a click on a name
lights up that student's trials. The other tabs: who built on whom, one row per student;
the handed-out baseline run again by everyone, whose spread is the noise; the class's
trials above the baseline in parallel coordinates; pairs of head choices crossed per
encoder; accuracy against the compute of training the head, where a click sets a trial
beside the others of about the same compute; and how much each student tried. A click on
an idea opens it in a drawer.

The teacher is the owner of the class repository (`--teacher` names another login). The
teacher's runs are drawn and named as the teacher's, and the teacher counts as no
student.

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
import base64
import hashlib
import html
import json
import math
import os
import re
import sys
import webbrowser
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_submission import COURSE, Submission, explain, idea_parents, is_number

COURSE_ROOT = Path(__file__).resolve().parents[1]
METRIC = "accuracy"


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
    characters that would open a tag or a link are escaped. Quotes stay as they are:
    plotly would show an escaped quote as its entity."""
    return html.escape(str(text), quote=False)


TEACHER = " · teacher"  # beside the teacher's name wherever a student's would stand


def as_teacher(df: pd.DataFrame, teacher: str | None) -> pd.DataFrame:
    """`df` with the teacher's runs, and the ideas the teacher wrote, under the teacher's
    name with `TEACHER` beside it: every picture that names a student then names the
    teacher as the teacher."""
    if not teacher or df.empty:
        return df
    out = df.copy()
    label = teacher + TEACHER
    out.loc[out.student == teacher, "student"] = label
    if "idea_author" in out:
        out.loc[out.idea_author == teacher, "idea_author"] = label
    return out


def is_teacher(name: object) -> bool:
    return isinstance(name, str) and name.endswith(TEACHER)


def students(df: pd.DataFrame) -> list[str]:
    """The students of `df`, the teacher left out."""
    if df.empty:
        return []
    return sorted(s for s in df.student.unique() if not is_teacher(s))


def people(names: Iterable[str]) -> str:
    """`names` counted in words: "4 students", "4 students and the teacher"."""
    names = set(names)
    n = sum(1 for s in names if not is_teacher(s))
    text = f"{n} student" + ("" if n == 1 else "s")
    return text + (" and the teacher" if any(map(is_teacher, names)) else "")


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


def loss_curves(attrs: Mapping[str, object]) -> tuple[list[float], list[float]] | None:
    """A run's train and test loss after every epoch, as it logged them; None for a run
    that logs neither."""
    train, test = attrs.get("train_loss"), attrs.get("test_loss")
    if not (isinstance(train, list) and isinstance(test, list) and train and test):
        return None
    if not all(is_number(v) for v in [*train, *test]):
        return None
    return [float(v) for v in train], [float(v) for v in test]


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
            "losses": loss_curves(attrs),
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


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline against final, per student and study: both scores with their spread
    between runs, the gain in points, and how much of the best single trial
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
    """The handed-out baseline of every student, one each: the latest one they ran."""
    sel = df[(df.origin == "notebook") & (df.role == "baseline") & df.score.notna()]
    sel = sel.sort_values("start").groupby("student").tail(1)
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


# ---- the noise: one model, run again ----------------------------------------------------


class Noise(NamedTuple):
    """How far one model's score moves when it is run again: the runs of the handed-out
    baseline's config, by everyone, each a mean over its own seeds. `center` is their
    median score, `spread` the standard deviation between them, `n` how many there are,
    `dataset` where they ran."""

    center: float
    spread: float
    n: int
    dataset: str

    @property
    def between(self) -> float:
        """The standard deviation of the difference between two runs of one model: the
        spread beside a difference of two single runs."""
        return math.sqrt(2) * self.spread


def config_of(row: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """A run's config as one value: every param it carries, as text."""
    return tuple(
        sorted(
            (k, str(v))
            for k, v in row.items()
            if k.startswith("p_")
            and v is not None
            and not (isinstance(v, float) and math.isnan(v))
        )
    )


def reruns(df: pd.DataFrame) -> pd.DataFrame:
    """Every baseline run, handed out or in a study, with the config of the handed-out
    baseline (the config most handed-out baselines have; of all baselines when none is
    handed out): the same model run again and again."""
    base = df[(df.role == "baseline") & df.score.notna()]
    if base.empty:
        return base
    configs = base.apply(config_of, axis=1)
    handed = configs[base.origin == "notebook"]
    common = (handed if len(handed) else configs).value_counts().index[0]
    return base[configs == common].sort_values("score").reset_index(drop=True)


def class_noise(df: pd.DataFrame) -> Noise | None:
    """The noise of `reruns`: None with fewer than three runs, too few to say how far
    a score moves, or when they all score the same, which says nothing either."""
    runs = reruns(df)
    spread = float(runs.score.std(ddof=1)) if len(runs) >= 3 else math.nan
    if math.isnan(spread) or spread < 1e-9:
        return None
    return Noise(
        center=float(runs.score.median()),
        spread=spread,
        n=len(runs),
        dataset=str(runs.dataset.iloc[0]),
    )


IDEA_COLUMNS = [
    "name", "idea", "author", "parents", "missing", "students", "n_runs", "student",
    "study", "dataset", "objective", "base", "base_std", "base_n", "final", "final_std",
    "final_n", "gain_pt", "expects", "budget", "axes", "studies",
    "round", "start",
]  # fmt: skip


def number(value: object) -> float:
    """`value` as a float; NaN for anything that is not a finite number."""
    try:
        out = float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def standard_error(*runs: tuple[object, object]) -> float:
    """The pooled standard error of the means of `runs`, each (spread, repeats): the
    spread beside a difference of two means. A run logs its spread as the standard
    deviation over its n repeats (ddof=0); the standard error of its mean is the sample
    standard deviation over √n, so s²/(n−1). NaN when one of them has a single run, no
    spread or no count."""
    total = 0.0
    for std, n in runs:
        s, k = number(std), number(n)
        if math.isnan(s) or math.isnan(k) or k <= 1:
            return float("nan")
        total += s**2 / (k - 1)
    return math.sqrt(total) if total > 0 else float("nan")


# What an idea's row takes from the study that stands for it, when it has a final.
RESULT = (
    "base",
    "base_std",
    "base_n",
    "final",
    "final_std",
    "final_n",
    "gain_pt",
)


def first_of(values: Iterable[object], kind: type) -> object | None:
    """The first of `values` that is a non-empty `kind`."""
    return next((v for v in values if isinstance(v, kind) and v), None)


def ideas(df: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """One row per idea of the class: its name, words, author, parents (the names it
    builds on; `missing` those not among the submissions), the students who ran it and
    their number of runs, its round (the column of the graph, `idea_layout`), its
    `studies` (student, study, trials; the one that stands for it first), and the
    study that stands for it: the one `comparisons` has with the most trials, or else
    the one with the most trials. Of that study: the winner against its own baseline, in
    points, both with their spread and repeats, its dataset
    and objective, what its author `expects`, its `budget` and the axes it searched.

    An idea is its name, so the handed-out idea of author `course` is one row for the
    whole class; it has no final of its own, the class's baselines and pooled search
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
        own = group[group.study.map(lambda s: isinstance(s, str))]
        studies = [
            (s, st, int((g.role == "trial").sum()))
            for (s, st), g in own.groupby(["student", "study"])
        ]
        studies.sort(key=lambda x: (x[:2] != (student, study), -x[2], x[0], x[1]))
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
                "expects": first_of(ran.expects, str) or "",
                "budget": first_of(ran.budget, dict),
                "axes": str(axes).split(",") if axes else [],
                "studies": studies,
                "start": ran.start.min() if len(ran) else group.start.min(),
            }
        )
    out.sort(key=lambda r: (r["author"] != COURSE, r["author"], r["name"]))
    frame = pd.DataFrame(out, columns=IDEA_COLUMNS)
    depth = idea_depths(frame)
    frame["round"] = [depth[n] for n in frame.name]
    return frame


EDGE_COLUMNS = [
    "parent", "child", "parent_author", "child_author", "diff_pt", "spread_pt", "number",
    "why",
]  # fmt: skip


def arrow_between(
    parent: SimpleNamespace, child: SimpleNamespace, noise: Noise | None = None
) -> tuple[float, float]:
    """What an arrow compares, as (difference, spread): the child's final minus the
    parent's final, both on the page's metric, so what building on the parent gave.
    The spread is that of a difference between two runs: from the class's reruns of one
    model (`Noise.between`) when they ran on the same dataset, else the standard error
    from the seeds of the two finals, NaN when a final has a single run. Every number an
    arrow carries comes from here."""
    diff = child.final - parent.final
    if noise is not None and noise.dataset == child.dataset:
        return diff, noise.between
    se = standard_error(
        (child.final_std, child.final_n), (parent.final_std, parent.final_n)
    )
    return diff, se


def edges(ids: pd.DataFrame, noise: Noise | None = None) -> pd.DataFrame:
    """One row per link of the graph, from a parent to the idea that builds on it. The
    number on it, `diff_pt`, is the child's final minus the parent's final in points
    (`arrow_between`), and `spread_pt` the spread of that difference in points, NaN
    where a final has a single run. A link has no number (`number` False, `why` says
    why) from a parent that is not submitted, from the course's idea (the child's own
    card holds its score against the handed-out baseline), from or to an idea without a
    final, and between ideas of different datasets."""
    by = {r.name: r for r in rows(ids)}
    out = []
    for child in rows(ids):
        for p in child.parents:
            parent = by.get(p)
            row = {
                "parent": p, "child": child.name, "child_author": child.author,
                "parent_author": None if parent is None else parent.author,
                "diff_pt": np.nan, "spread_pt": np.nan, "number": False, "why": "",
            }  # fmt: skip
            if parent is None:
                row["why"] = "the parent is not among the submissions"
            elif parent.author == COURSE:
                row["why"] = (
                    "the handed-out idea: the card holds its score on the baseline"
                )
            elif math.isnan(number(parent.final)) or math.isnan(number(child.final)):
                row["why"] = "no final at both ends"
            elif parent.dataset != child.dataset:
                row["why"] = f"different datasets ({parent.dataset}, {child.dataset})"
            else:
                diff, spread = arrow_between(parent, child, noise)
                row.update(diff_pt=100 * diff, spread_pt=100 * spread, number=True)
            out.append(row)
    return pd.DataFrame(out, columns=EDGE_COLUMNS)


def rows(df: pd.DataFrame) -> list[SimpleNamespace]:
    """The rows of `df` as objects with the columns as attributes."""
    return [SimpleNamespace(**r) for r in df.to_dict("records")]


# ---- words and numbers -----------------------------------------------------------------


def signed(value: float, digits: int = 1) -> str:
    """`value` with its sign, a minus sign rather than a hyphen; no sign on a value that
    rounds to zero."""
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        return f"{0:.{digits}f}"
    return text.replace("-", "−")


def tick(value: float) -> str:
    """A number at the end of an axis, short: two digits below 0.01, three above."""
    return f"{value:.2g}" if abs(value) < 0.01 else f"{value:.3g}"


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


OPTION = re.compile(r"""@option\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")
ON = (True, "true", "on", "yes", 1)
OFF = (False, "false", "off", "no", 0)


def chip_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.0f}" if value.is_integer() else f"{value:.3g}"
    return str(value)


def esc(text: object) -> str:
    return html.escape(str(text))


WHEN_TZ = "Europe/Amsterdam"


def moment(start: object) -> pd.Timestamp | None:
    """A run's start (milliseconds since the epoch) in the course's time zone."""
    ms = number(start)
    if math.isnan(ms):
        return None
    when = pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert(WHEN_TZ)
    return when if isinstance(when, pd.Timestamp) else None


def idea_depths(ids: pd.DataFrame) -> dict[str, int]:
    """How many links lie between every idea, and every parent that is not submitted,
    and a root: the longest path, a root at 0, a parent not submitted one left of its
    first child."""
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
    return depth


# ---- the ideas of one study ------------------------------------------------------------


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


def with_curve(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials of `trials` that logged a curve."""
    if trials.empty or "curve" not in trials:
        return trials.iloc[:0]
    return trials[
        trials.curve.map(lambda c: isinstance(c, list) and len(c) > 0).astype(bool)
    ]


def finished(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials of `trials` that ran to the end and have a score."""
    return trials[(trials.status == "COMPLETE") & trials.score.notna()]


# Rows in the train split of a dataset's cache, where the page knows them.
TRAIN_ROWS = {"dtd": 3760}


def train_flops(trials: pd.DataFrame) -> pd.Series:
    """The compute of training each trial's head on the cached vectors: about six
    FLOPs per parameter per training row per epoch (a forward and a backward pass),
    over `n_train` rows when the trial took a subset, else the dataset's train split
    (`TRAIN_ROWS`). NaN where the parameters, the epochs or the rows are unknown. The
    encoder ran once, when the cache was made, and is not in it."""
    params = column(trials, "n_params")
    epochs = column(trials, "epochs_run")
    if "p_epochs" in trials:
        epochs = epochs.fillna(pd.to_numeric(trials.p_epochs, errors="coerce"))
    rows_ = pd.to_numeric(trials.dataset.map(TRAIN_ROWS), errors="coerce")
    if "p_n_train" in trials:
        rows_ = pd.to_numeric(trials.p_n_train, errors="coerce").fillna(rows_)
    return 6 * params * epochs * rows_


def all_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Every finished trial of the class, handed-out search and own studies alike, with
    `flops`, the compute of training its head (`train_flops`)."""
    trials = finished(df[df.role == "trial"]).copy()
    trials["flops"] = train_flops(trials)
    return trials.reset_index(drop=True)


# ---- the pictures ----------------------------------------------------------------------

PAPER = "#f2f3ef"
INK = "#111111"
MUTED = "#5a544d"
RULE = "#d6d9d1"
SURFACE = "#e8eae3"
BLUE = "#1d5a73"
BLUE_WASH = "#c9dde4"
ORANGE = "#c43a14"
GREEN = "#1c6b48"
FONT = "Outfit, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
# A score above the baseline, from just above it to the best: one hue, light to dark.
SCORE_SCALE = [[0.0, "#d6e6ec"], [0.35, "#7fb0c3"], [0.7, "#2f7491"], [1.0, "#0c3445"]]


# Okabe and Ito's colours, distinct for colour-blind readers, one per encoder.
ENCODER_COLOURS = [
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#E69F00",
    "#000000",
]

AXES = [
    "encoder", "n_hidden_layers", "hidden_dim", "dropout", "activation", "norm",
    "skip", "gate", "sigma", "lr", "epochs",
]  # fmt: skip
CATEGORICAL = {"activation", "norm", "skip", "encoder", "dataset", "gate", "head"}
LOG_AXES = {"lr", "hidden_dim"}


def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    layout = {
        "margin": {"l": 64, "r": 24, "t": 24, "b": 52},
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        **layout,
    }
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 14, "color": INK},
        hoverlabel={"font": {"family": FONT, "size": 13}},
        **layout,
    )
    fig.update_xaxes(showline=True, linecolor=RULE, gridcolor="#e2e5dd", zeroline=False,
                     tickfont={"family": MONO, "size": 13}, ticks="outside", tickcolor=RULE)  # fmt: skip
    fig.update_yaxes(showline=True, linecolor=RULE, gridcolor="#e2e5dd", zeroline=False,
                     tickfont={"family": MONO, "size": 13}, ticks="outside", tickcolor=RULE)  # fmt: skip
    return fig


def empty_note(fig: go.Figure, text: str) -> None:
    """`text` in the middle of a picture that has nothing to draw, without its axes."""
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, text=plain(text),
        font={"size": 14, "color": MUTED},
    )  # fmt: skip


def score_range(scores: Iterable[float], noise: Noise | None) -> list[float]:
    """The score axis of a picture: from under the baseline's band (or the lowest score
    when there is no baseline) to just over the best score. The pictures show what the
    class found above the baseline, so what lies under the band gets no room."""
    values = [float(s) for s in scores if not math.isnan(number(s))]
    if not values:
        return [0, 1]
    high = max(values)
    low = noise.center - 2 * noise.spread if noise is not None else min(values)
    low = min(low, high - 0.01)
    pad = (high - low) * 0.06
    return [low, high + pad]


def axis_values(col: pd.Series, axis: str) -> tuple[pd.Series, dict]:
    """A param's values for a parallel axis, and the axis's ticks: a choice spread over
    its values, a number on a log scale for `LOG_AXES` when all are positive."""
    if axis in CATEGORICAL or col.dtype == object or col.dtype == bool:
        cats = sorted(col.dropna().astype(str).unique())
        if axis == "encoder":
            cats = list(encoder_order(cats))  # smallest to largest
        codes = col.astype(str).map({c: i for i, c in enumerate(cats)})
        return codes, {"tickvals": list(range(len(cats))), "ticktext": cats,
                       "range": [-0.3, max(len(cats) - 0.7, 0.7)]}  # fmt: skip
    values = pd.to_numeric(col, errors="coerce")
    positive = values[values > 0]
    if axis in LOG_AXES and len(positive):
        # a zero (no hidden layer, so no width) sits a step under the smallest value
        low = float(positive.min()) / 2
        logs = np.log10(values.where(values > 0, low))
        if axis == "hidden_dim":
            ticks = [v for v in (16, 32, 64, 128, 256, 512, 1024, 2048)
                     if positive.min() <= v <= positive.max()]  # fmt: skip
        else:
            ticks = [10.0**e for e in range(math.ceil(math.log10(positive.min())),
                                            math.floor(math.log10(positive.max())) + 1)]  # fmt: skip
        vals = [math.log10(t) for t in ticks]
        text = [f"{t:g}" if t >= 1 else tick(t) for t in ticks]
        if (values <= 0).any():
            vals, text = [math.log10(low), *vals], ["none", *text]
        return logs, {"tickvals": vals, "ticktext": text}
    whole = values.dropna()
    if len(whole) and (whole == whole.round()).all() and whole.max() - whole.min() <= 8:
        steps = list(range(int(whole.min()), int(whole.max()) + 1))
        return values, {"tickvals": steps, "ticktext": [str(v) for v in steps]}
    return values, {}


def fig_lines(
    trials: pd.DataFrame,
    axes: list[str],
    noise: Noise | None,
    metric: str,
    height: int = 420,
) -> tuple[go.Figure, int]:
    """Parallel coordinates of the finished `trials` that beat the baseline, over the
    `axes` that vary among them, coloured by score from the baseline up. Also the number
    of finished trials left out because they scored under the baseline (all are drawn
    when there is no baseline)."""
    done = finished(trials)
    floor = noise.center if noise is not None else -math.inf
    shown = done[done.score > floor].sort_values("score")
    below = len(done) - len(shown)
    fig = go.Figure()
    used = [a for a in axes if f"p_{a}" in shown and shown[f"p_{a}"].nunique() > 1]
    if shown.empty or not used:
        empty_note(fig, "no finished trial above the baseline" if used or shown.empty
                   else "the trials above the baseline share every value")  # fmt: skip
        return style(fig, 200), below
    dims = []
    for axis in used:
        values, ticks = axis_values(shown[f"p_{axis}"], axis)
        label = "hidden width" if axis == "hidden_dim" else (
            "hidden layers" if axis == "n_hidden_layers" else axis)  # fmt: skip
        dims.append({"label": label, "values": values, **ticks})
    low = noise.center if noise is not None else float(shown.score.min())
    top = float(shown.score.max())
    score = {"label": metric, "values": shown.score, "range": [low, top]}
    if noise is not None:
        # points above the baseline, the only reading of a score this page makes
        gains = range(math.floor(100 * (top - low)) + 1)
        step = max(1, math.ceil(len(gains) / 6))
        score["label"] = "pt above baseline"
        score["tickvals"] = [low + g / 100 for g in gains[::step]]
        score["ticktext"] = [f"+{g}" for g in gains[::step]]
    dims.append(score)
    by_score = {
        "color": shown.score.tolist(),
        "colorscale": SCORE_SCALE,
        "cmin": low,
        "cmax": top,
        "showscale": False,
    }
    encoders = (
        list(encoder_order(shown.p_encoder.fillna("unknown").astype(str)))
        if "p_encoder" in shown
        else []
    )
    codes = [
        encoders.index(e) if e in encoders else 0
        for e in shown.get("p_encoder", pd.Series("unknown", index=shown.index))
        .fillna("unknown")
        .astype(str)
    ]
    steps = max(len(encoders), 1)
    by_encoder = {
        "color": codes,
        "cmin": -0.5,
        "cmax": steps - 0.5,
        "showscale": False,
        "colorscale": [
            [edge, ENCODER_COLOURS[i % len(ENCODER_COLOURS)]]
            for i in range(steps)
            for edge in (i / steps, (i + 1) / steps)
        ],
    }
    fig.update_layout(
        meta={
            "by_score": by_score,
            "by_encoder": by_encoder,
            "encoders": [
                [e, ENCODER_COLOURS[i % len(ENCODER_COLOURS)]]
                for i, e in enumerate(encoders)
            ],
            "encoder_axis": used.index("encoder") if "encoder" in used else None,
        }
    )
    fig.add_trace(
        go.Parcoords(
            line=by_score,
            dimensions=dims,
            labelfont={"family": FONT, "size": 15, "color": INK},
            tickfont={"family": MONO, "size": 13, "color": MUTED},
            unselected={"line": {"color": "#b9b4aa", "opacity": 0.18}},
            rangefont={"size": 1, "color": "rgba(0,0,0,0)"},
        )
    )  # fmt: skip
    return style(fig, height, margin={"l": 110, "r": 90, "t": 54, "b": 24}), below


def fig_reruns(runs: pd.DataFrame, noise: Noise, metric: str) -> go.Figure:
    """Every rerun of the handed-out baseline as a dot with its seeds' spread, the
    median as a line and median ± the standard deviation between runs as a band."""
    fig = go.Figure()
    labels = [
        plain(f"{r.student} · {r.study}" if isinstance(r.study, str) else f"{r.student} · notebook")
        for r in rows(runs)
    ]  # fmt: skip
    fig.add_vrect(x0=noise.center - noise.spread, x1=noise.center + noise.spread,
                  fillcolor=BLUE_WASH, opacity=0.55, line_width=0, layer="below")  # fmt: skip
    fig.add_vline(x=noise.center, line_color=BLUE, line_width=1.5, line_dash="dot")
    fig.add_scatter(
        x=runs.score, y=labels, mode="markers",
        marker={"size": 11, "color": BLUE, "line": {"color": PAPER, "width": 1.5}},
        error_x={"type": "data", "array": runs["std"].fillna(0), "color": BLUE,
                 "thickness": 1.5, "width": 0},
        hovertemplate="%{y}<br>%{x:.4f} (bar: one standard deviation over its seeds)<extra></extra>",
        showlegend=False,
    )  # fmt: skip
    fig.update_xaxes(title=f"{metric} of the handed-out baseline", tickformat=".3f")
    fig.update_yaxes(tickfont={"family": MONO, "size": 11}, showgrid=False)
    return style(fig, 26 * len(runs) + 90, margin={"l": 330, "r": 24, "t": 12, "b": 56})


BIN_ORDER = {
    "hidden_dim": ["< 64", "64–127", "128–255", "256–511", "≥ 512"],
    "dropout": ["0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "≥ 0.4"],
    "flops": ["< 30 G", "30–100 G", "100–300 G", "300 G–1 T", "≥ 1 T"],
    "form": [
        "linear",
        "hidden layers",
        "+ skip",
        "+ gate",
        "+ skip + gate",
        "two encoders fused",
    ],
}


def bins_of(values: pd.Series, axis: str) -> pd.Series:
    """A param's values as the rows or columns of an interaction grid: hidden widths in
    powers of two, dropout in tenths, layers and choices as they are."""
    if axis == "hidden_dim":
        edges_ = [0, 64, 128, 256, 512, 1e9]
        names = BIN_ORDER["hidden_dim"]
        return pd.cut(pd.to_numeric(values), edges_, right=False, labels=names).astype(
            str
        )
    if axis == "dropout":
        edges_ = [0, 0.1, 0.2, 0.3, 0.4, 1.01]
        names = BIN_ORDER["dropout"]
        return pd.cut(pd.to_numeric(values), edges_, right=False, labels=names).astype(
            str
        )
    if axis == "flops":
        edges_ = [0, 3e10, 1e11, 3e11, 1e12, math.inf]
        return pd.cut(
            pd.to_numeric(values), edges_, right=False, labels=BIN_ORDER["flops"]
        ).astype(str)
    if axis == "n_hidden_layers":
        return pd.to_numeric(values).astype("Int64").astype(str)
    return values.astype(str)


PAIR_NAMES = {"n_hidden_layers": "hidden layers", "hidden_dim": "hidden width",
              "dropout": "dropout", "flops": "FLOPs", "form": "head"}  # fmt: skip
PAIRS = [("n_hidden_layers", "hidden_dim"), ("hidden_dim", "dropout"),
         ("n_hidden_layers", "dropout")]  # fmt: skip
# Grids against the compute of training the head, over every trial of an encoder, a
# linear head included.
FLOPS_PAIRS = [("form", "flops"), ("dropout", "flops")]
MIN_CELL = 3  # trials in a cell before its median is shown
MIN_FACET = 30  # finished trials of an encoder before it gets a row of grids


MIN_HEADS = (
    4  # heads (layers × width) with MIN_CELL trials before an encoder gets grids
)


def head_cells(trials: pd.DataFrame) -> int:
    """How many heads, hidden layers × width bin, have at least `MIN_CELL` trials."""
    keys = pd.Series(
        list(
            zip(
                bins_of(trials.p_n_hidden_layers, "n_hidden_layers"),
                bins_of(trials.p_hidden_dim, "hidden_dim"),
                strict=True,
            )
        )
    )
    return int((keys.value_counts() >= MIN_CELL).sum())


def interaction_facets(trials: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The encoders that have enough finished trials over the head's axes for a row of
    interaction grids, the most trials first, each with its trials."""
    need = [f"p_{a}" for a in ("n_hidden_layers", "hidden_dim", "dropout")]
    if any(c not in trials for c in need):
        return []
    usable = trials.dropna(subset=need)
    if "p_encoder" not in usable:
        return [("all", usable)] if len(usable) >= MIN_FACET else []
    out = [
        (str(enc), g)
        for enc, g in usable.groupby("p_encoder")
        if len(g) >= MIN_FACET and head_cells(g) >= MIN_HEADS
    ]
    return sorted(out, key=lambda x: -len(x[1]))


def facets_left_out(trials: pd.DataFrame) -> list[str]:
    """The encoders with finished trials over the head's axes that get no row of grids,
    each with why."""
    need = [f"p_{a}" for a in ("n_hidden_layers", "hidden_dim", "dropout")]
    if any(c not in trials for c in need) or "p_encoder" not in trials:
        return []
    out = []
    for enc, g in trials.dropna(subset=need).groupby("p_encoder"):
        if len(g) < MIN_FACET:
            out.append(
                f"{enc} ({plural(len(g), 'finished trial')}, fewer than {MIN_FACET})"
            )
        elif head_cells(g) < MIN_HEADS:
            out.append(f"{enc} (its trials tried {plural(head_cells(g), 'head')} with "
                       f"{MIN_CELL} trials or more, fewer than {MIN_HEADS})")  # fmt: skip
    return out


ROW_GAP = 150  # pixels between two rows of grids


def fig_interactions(
    trials: pd.DataFrame, noise: Noise, metric: str
) -> go.Figure | None:
    """Per encoder, grids of two head choices against each other, and of the shape of
    the head and dropout against the FLOPs of training it: a cell is the median score
    of the finished trials that fell in it, in points above the baseline; a cell under the baseline, or with fewer than `MIN_CELL` trials, gets no
    colour. Each encoder has its own colour scale, from its weakest coloured cell to its
    best, and its best cell is set in bold. Widths count only where the head has a hidden
    layer."""
    facets = interaction_facets(trials)
    if not facets:
        return None
    pairs = PAIRS + (FLOPS_PAIRS if "flops" in trials else [])
    titles = [
        plain(f"{enc} · {PAIR_NAMES[a]} × {PAIR_NAMES[b]}")
        for enc, _ in facets for a, b in pairs
    ]  # fmt: skip
    # a row's slanted tick labels and axis title above the next row's titles, in pixels
    height = 300 * len(facets) + ROW_GAP * (len(facets) - 1) + 90
    fig = make_subplots(rows=len(facets), cols=len(pairs), subplot_titles=titles,
                        horizontal_spacing=0.065, vertical_spacing=ROW_GAP / (height - 90) if len(facets) > 1 else 0.1)  # fmt: skip
    for i, (enc, g) in enumerate(facets):
        axis_name = "coloraxis" if i == 0 else f"coloraxis{i + 1}"
        grids = []
        every = trials if enc == "all" else trials[trials.p_encoder.astype(str) == enc]
        every = every.assign(
            p_flops=every.flops, p_form=[head_form(r) for r in rows(every)]
        )
        for a, b in pairs:
            part = every if (a, b) in FLOPS_PAIRS else g
            part = part.dropna(subset=[f"p_{a}", f"p_{b}"])
            if "hidden_dim" in (a, b):
                part = part[pd.to_numeric(part.p_n_hidden_layers) > 0]
            ya, xb = bins_of(part[f"p_{a}"], a), bins_of(part[f"p_{b}"], b)
            grid = part.assign(_y=ya.values, _x=xb.values).groupby(["_y", "_x"]).score
            med, cnt = grid.median().unstack(), grid.size().unstack().fillna(0)
            ys = sorted(med.index, key=lambda v, a=a: order_key(v, a))
            xs = sorted(med.columns, key=lambda v, b=b: order_key(v, b))
            med = med.reindex(index=ys, columns=xs)
            cnt = cnt.reindex(index=ys, columns=xs).fillna(0)
            gain = 100 * (med - noise.center)
            grids.append(
                (a, b, xs, ys, gain, cnt, gain.where((cnt >= MIN_CELL) & (gain > 0)))
            )
        # one colour scale per encoder, from its weakest coloured cell to its best, which
        # is set in bold
        shown = [z.values[~np.isnan(z.values)] for *_, z in grids]
        values = np.concatenate(shown) if shown else np.array([])
        low, best = (
            (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
        )
        for j, (a, b, xs, ys, gain, cnt, z) in enumerate(grids):
            text = [
                [cell_text(gain.iloc[r, c], int(cnt.iloc[r, c]), top=best)
                 for c in range(len(xs))]
                for r in range(len(ys))
            ]  # fmt: skip
            fig.add_heatmap(
                z=z.values, x=xs, y=ys, text=text, texttemplate="%{text}",
                textfont={"family": MONO, "size": 13},
                coloraxis=axis_name, xgap=2, ygap=2, hoverinfo="skip",
                row=i + 1, col=j + 1,
            )  # fmt: skip
            fig.update_xaxes(
                title_text=PAIR_NAMES[b], type="category", row=i + 1, col=j + 1
            )
            # the subplot's title names both choices; a y title would crowd the labels
            short = [
                str(y).replace("hidden layers", "hidden").replace(" + ", "+")
                for y in ys
            ]
            fig.update_yaxes(
                type="category", tickvals=ys, ticktext=short, row=i + 1, col=j + 1
            )
        fig.update_layout({axis_name: {"colorscale": SCORE_SCALE, "cmin": low,
                                       "cmax": max(best, low + 1e-6), "showscale": False}})  # fmt: skip
    fig.update_annotations(font={"size": 14, "family": FONT})
    style(fig, height, margin={"l": 70, "r": 24, "t": 40, "b": 50})
    fig.update_xaxes(showgrid=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, showline=False, ticks="")
    fig.update_layout(plot_bgcolor=SURFACE)
    return fig


def order_key(label: object, axis: str) -> tuple[float, str]:
    """A bin's place in its axis: as `BIN_ORDER` lists it, else by its number."""
    text = str(label)
    if text in BIN_ORDER.get(axis, []):
        return (BIN_ORDER[axis].index(text), text)
    m = re.search(r"\d+(\.\d+)?", str(label))
    return (float(m.group()) if m else math.inf, str(label))


def cell_text(gain: float, n: int, top: float = math.inf) -> str:
    """A cell's words: its points above the baseline (bold from `top` up) and its
    number of trials; `below` under the baseline, only the count under `MIN_CELL`."""
    if n == 0:
        return ""
    if n < MIN_CELL:
        return f"<span style='color:{MUTED}'>n {n}</span>"
    if math.isnan(gain) or gain <= 0:
        return f"<span style='color:{MUTED}'>below<br>n {n}</span>"
    value = f"<b>{signed(gain)}</b>" if gain >= top else signed(gain)
    return f"{value}<br><span style='font-size:12px'>n {n}</span>"


def encoder_order(encoders: Iterable[str]) -> list[str]:
    """The encoders from small to large: by the size word in the name (small, base,
    large) and then by name."""
    size = {"tiny": 0, "small": 1, "base": 2, "large": 3, "giant": 4}

    def key(e: str) -> tuple[int, str]:
        return next((v for k, v in size.items() if k in e), 9), e

    return sorted(set(encoders), key=key)


def pareto(points: pd.DataFrame, cost: str = "flops") -> pd.DataFrame:
    """The points no other point beats on both `cost` and score: the cheapest way to
    each score."""
    best = -math.inf
    keep = []
    for i, r in points.sort_values([cost, "score"], ascending=[True, False]).iterrows():
        if r.score > best:
            keep.append(i)
            best = r.score
    return points.loc[keep]


def flops_text(v: float) -> str:
    """A count of FLOPs with its SI prefix: 1e9 as 1 G."""
    for div, unit in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g} {unit}"
    return f"{v:g}"


ENCODER_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "x", "star"]


def fig_cost(
    trials: pd.DataFrame,
    noise: Noise,
    metric: str,
    tags: Mapping[str, list[str]] | None = None,
) -> tuple[go.Figure, int]:
    """Every finished trial that beat the baseline: its score against the compute of
    training its head (`train_flops`), a colour and a marker shape per encoder, and the
    cheapest way to each score as a line whose points name their idea. The figure
    carries every point's FLOPs, score, encoder, student, idea, tags and head, so a
    click can set a trial beside the others of about the same compute. Also the number
    left out under the baseline."""
    has = trials[trials.flops.notna() & (trials.flops > 0)]
    shown = has[has.score > noise.center].reset_index(drop=True)
    below = len(has) - len(shown)
    fig = go.Figure()
    if shown.empty:
        empty_note(fig, "no trial with known parameters and epochs beat the baseline")
        return style(fig, 200), below
    tags = tags or {}
    enc = (
        shown.p_encoder.fillna("unknown").astype(str)
        if "p_encoder" in shown
        else pd.Series("all", index=shown.index)
    )
    names = shown.idea_name.fillna("").astype(str)
    points = [
        [float(r.flops), float(r.score), e, plain(r.student), n,
         ", ".join(tags.get(n, [])) or "handed-out search", head_form(r)]
        for r, e, n in zip(rows(shown), enc, names, strict=True)
    ]  # fmt: skip
    for k, e in enumerate(encoder_order(enc)):
        part = shown[enc == e]
        fig.add_scatter(
            x=part.flops, y=part.score, mode="markers", name=plain(e),
            marker={"size": 8, "symbol": ENCODER_SYMBOLS[k % len(ENCODER_SYMBOLS)],
                    "color": ENCODER_COLOURS[k % len(ENCODER_COLOURS)], "opacity": 0.6,
                    "line": {"color": PAPER, "width": 0.6}},
            customdata=list(part.index),
            text=[plain(f"{r.student} · {r.idea_name or 'handed-out search'}") for r in rows(part)],
            hovertemplate="%{text}<br>%{y:.3f} at %{x:.2s} FLOPs<extra></extra>",
        )  # fmt: skip
    front = pareto(shown)
    # each idea named once on the front, at its first point, alternately above and below
    names_on = front.idea_name.fillna("").astype(str).tolist()
    labels, side, k = [], [], 0
    for i, n in enumerate(names_on):
        first = n and (i == 0 or names_on[i - 1] != n) and n not in names_on[:i]
        labels.append(plain(n) if first else "")
        side.append(
            "top right" if i == 0 else "top left" if k % 2 == 0 else "bottom right"
        )
        k += bool(first)
    fig.add_scatter(
        x=front.flops, y=front.score, mode="lines+markers+text", name="cheapest way to each score",
        line={"color": INK, "width": 1.5, "shape": "hv"}, marker={"size": 9, "color": INK},
        customdata=list(front.index), text=labels, textposition=side,
        textfont={"family": MONO, "size": 12, "color": INK},
        hovertemplate="%{text}<br>%{y:.3f} at %{x:.2s} FLOPs<extra></extra>",
    )  # fmt: skip
    band_shape(fig, noise, accuracy=True)
    ticks, words = decade_ticks(shown.flops)
    fig.update_xaxes(
        type="log", tickvals=ticks, ticktext=words,
        title="compute of training the head (FLOPs, about 6 × parameters × rows × epochs)",
    )  # fmt: skip
    fig.update_yaxes(title=metric)
    fit_range(fig, [shown.score])
    fig.update_layout(meta={"points": points, "near": NEAR})
    return style(fig, 460), below


# How far a trial's compute may lie from a clicked one to count as about the same.
NEAR = 1.6


def fig_trials(
    trials: pd.DataFrame,
    r: SimpleNamespace,
    parent: SimpleNamespace | None,
    noise: Noise | None,
    metric: str,
) -> go.Figure:
    """One study's trials in the order they ran (pruned ones open), the baseline and the
    parent's final as lines, the final as a diamond with its seeds' spread. The axis
    starts just under the baseline; trials further down sit on its floor."""
    fig = go.Figure()
    scores = [*trials.score.dropna(), number(r.final)]
    yr = score_range(scores, noise)
    clip = trials.score.clip(lower=yr[0] + (yr[1] - yr[0]) * 0.01)
    done = trials.status == "COMPLETE"
    if noise is not None:
        fig.add_hline(y=noise.center, line_color=BLUE, line_dash="dot", line_width=1.2,
                      annotation_text="handed-out baseline", annotation_position="bottom right",
                      annotation_font={"size": 11, "color": BLUE})  # fmt: skip
    if parent is not None and not math.isnan(number(parent.final)):
        fig.add_hline(y=parent.final, line_color=ORANGE, line_dash="dash", line_width=1.2,
                      annotation_text=plain(f"final of {parent.name}"), annotation_position="top left",
                      annotation_font={"size": 11, "color": ORANGE})  # fmt: skip
    fig.add_scatter(x=trials.index[~done] + 1, y=clip[~done], mode="markers", name="stopped early",
                    marker={"size": 7, "color": PAPER, "line": {"color": MUTED, "width": 1.2}},
                    hovertemplate="trial %{x}, stopped early: %{y:.4f}<extra></extra>")  # fmt: skip
    fig.add_scatter(x=trials.index[done] + 1, y=clip[done], mode="markers", name="finished",
                    marker={"size": 8, "color": BLUE, "opacity": 0.75},
                    hovertemplate="trial %{x}: %{y:.4f}<extra></extra>")  # fmt: skip
    if not math.isnan(number(r.final)):
        fig.add_scatter(x=[len(trials) + 2], y=[r.final], mode="markers", name="final, rerun",
                        marker={"size": 13, "color": INK, "symbol": "diamond"},
                        error_y={"type": "data", "array": [number(r.final_std) if not math.isnan(number(r.final_std)) else 0],
                                 "color": INK, "thickness": 1.5, "width": 4},
                        hovertemplate="final: %{y:.4f}<extra></extra>")  # fmt: skip
    if trials.empty and math.isnan(number(r.final)):
        empty_note(fig, "no trial and no final submitted")
    fig.update_xaxes(title="trial, in the order it ran")
    fig.update_yaxes(title=metric, range=yr, tickformat=".3f")
    return style(fig, 300, margin={"l": 64, "r": 16, "t": 30, "b": 48})


def fig_curves(
    trials: pd.DataFrame, noise: Noise | None, metric: str, top: int = 5
) -> go.Figure:
    """The metric per epoch of the `top` best finished trials, the winner dark; the
    axis starts just under the baseline."""
    fig = go.Figure()
    has = finished(with_curve(trials)).sort_values("score", ascending=False).head(top)
    if has.empty:
        empty_note(fig, "no finished trial logged its curve")
        return style(fig, 200)
    for k, r in enumerate(rows(has)):
        fig.add_scatter(
            x=list(range(1, len(r.curve) + 1)), y=r.curve, mode="lines",
            line={"color": INK if k == 0 else "#8fb8c8", "width": 2.5 if k == 0 else 1.5},
            name="the winner" if k == 0 else f"number {k + 1}", showlegend=k < 2,
            hovertemplate="epoch %{x}: %{y:.4f}<extra></extra>",
        )  # fmt: skip
    if noise is not None:
        fig.add_hline(y=noise.center, line_color=BLUE, line_dash="dot", line_width=1.2)
    tops = [max(c) for c in has.curve]
    fig.update_yaxes(title=metric, range=score_range(tops, noise), tickformat=".3f")
    fig.update_xaxes(title="epoch")
    return style(fig, 280, margin={"l": 64, "r": 16, "t": 30, "b": 48})


# ---- the page --------------------------------------------------------------------------


# ---- what the class investigated, by learning goal --------------------------------------


class Theme(NamedTuple):
    """A learning goal of the lesson as the class could investigate it: its number in
    the lesson's goals, the goal said short, and the axes whose search or change puts
    an idea under it."""

    key: str
    goal: str
    title: str
    axes: tuple[str, ...]


# Per lesson, the goals a study can investigate, each with the axes that touch it. A
# lesson without an entry gets one theme per searched axis (`themes_of`).
THEMES: dict[int, tuple[Theme, ...]] = {
    1: (
        Theme("encoder", "1.5", "The encoder can move the score more than the head", ("encoder",)),
        Theme("capacity", "1.6", "Capacity of the head: depth and width",
              ("n_hidden_layers", "hidden_dim", "activation")),
        Theme("overfitting", "1.9", "Overfitting: train and test loss over the epochs, and dropout",
              ("dropout", "sigma", "norm", "epochs")),
        Theme("simpler", "1.10", "A more complex head against a simpler one, for what it costs",
              ("skip", "gate", "head")),
    ),
}  # fmt: skip
# Axes nearly every study tunes along the way, which make no theme of their own.
ALONG_THE_WAY = {"lr", "dataset", "seed", "n_train"}
OTHER = Theme("other", "", "Ideas that changed nothing the goals above name", ())


def themes_of(lesson: int, ids: pd.DataFrame) -> tuple[Theme, ...]:
    """The themes of `lesson`: its entry in `THEMES`, or one per axis some idea
    searched."""
    if lesson in THEMES:
        return THEMES[lesson]
    axes = sorted({a for r in rows(ids) for a in r.axes if a not in ALONG_THE_WAY})
    return tuple(Theme(a, "", PAIR_NAMES.get(a, a), (a,)) for a in axes)


def idea_axes(df: pd.DataFrame, r: SimpleNamespace) -> list[str]:
    """What an idea investigated: the axes its study searched, or, for one that searched
    none, the params in which its final (else its runs) differs from the handed-out
    baseline."""
    searched = [a for a in r.axes if a and a not in ALONG_THE_WAY]
    if searched:
        return searched
    base = reruns(df)
    if base.empty:
        return []
    handed = dict(config_of(base.iloc[0]))
    mine = df[(df.idea_name == r.name) & (df.student == r.student)]
    run = mine[mine.role == "final"]
    run = run if len(run) else mine
    if run.empty:
        return []
    own = dict(config_of(run.iloc[0]))
    return sorted(
        k.removeprefix("p_") for k in set(own) | set(handed)
        if own.get(k) != handed.get(k) and k.removeprefix("p_") not in ALONG_THE_WAY
    )  # fmt: skip


def ideas_by_theme(
    df: pd.DataFrame, ids: pd.DataFrame, themes: tuple[Theme, ...]
) -> dict[str, list[SimpleNamespace]]:
    """The students' ideas under every theme whose axes they investigated
    (`idea_axes`); an idea under none stands under `OTHER`."""
    out: dict[str, list[SimpleNamespace]] = {t.key: [] for t in (*themes, OTHER)}
    for r in rows(ids[ids.author != COURSE]):
        axes = set(idea_axes(df, r))
        hit = [t.key for t in themes if axes & set(t.axes)]
        for key in hit or [OTHER.key]:
            out[key].append(r)
    for items in out.values():
        items.sort(key=lambda r: number(r.start))
    return out


# The words an idea can be tagged with: a tag, the axes a study searches or changes when
# it investigates that (none: the question alone decides), and the words of a question
# that name it, in Dutch or English.
TAGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("two encoders", ("encoder", "head", "embed_dim"), r"fusi|fuse|two encoders|tweede encoder|second encoder"),
    ("encoder", ("encoder", "embed_dim"), r"encoder"),
    ("linear head", ("skip", "gate", "head", "n_hidden_layers"), r"linea\w* head|\blinear\b"),
    ("depth", ("n_hidden_layers",), r"diep|depth|deep|\blagen\b|\blayers\b|\blaag\b"),
    ("width", ("hidden_dim",), r"bred|breed|\bwid|width|smaller|smallere|kleiner|narrow|hidden dim|units"),
    ("dropout", ("dropout",), r"dropout"),
    ("norm", ("norm",), r"\bnorm"),
    ("activation", ("activation",), r"gelu|relu|activ"),
    ("skip", ("skip",), r"skip"),
    ("gate", ("gate",), r"\bgate"),
    ("noise", ("sigma",), r"\bnoise|\bruis|sigma"),
    ("epochs", ("epochs",), r"epoch"),
    ("pruner", (), r"prun"),
)  # fmt: skip
AXIS_TAGS = {"n_hidden_layers": "depth", "hidden_dim": "width", "sigma": "noise",
             "head": "head form", "embed_dim": "encoder"}  # fmt: skip
WHOLE_HEAD = (
    5  # searched axes from which an idea is tagged as a search of the whole head
)


def idea_tags(df: pd.DataFrame, r: SimpleNamespace) -> list[str]:
    """What an idea is about, in the words of `TAGS`: those its question names, where its
    study searched or changed that axis (`idea_axes`). A question that names none is
    tagged by its axes, or as `whole head` when it searched `WHOLE_HEAD` or more."""
    if r.author == COURSE:
        return ["handed-out search"]
    axes = set(idea_axes(df, r))
    text = str(r.idea).lower()
    tags = [
        t
        for t, on, words in TAGS
        if re.search(words, text) and (not on or axes & set(on))
    ]
    if "two encoders" in tags and "encoder" in tags:
        tags.remove("encoder")
    if tags:
        return tags
    if len(axes) >= WHOLE_HEAD:
        return ["whole head"]
    return sorted({AXIS_TAGS.get(a, a) for a in axes})


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column of `frame`, all NaN when it has none of that name."""
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def gain_of(trials: pd.DataFrame, noise: Noise) -> pd.Series:
    """Scores as points above the handed-out baseline."""
    return 100 * (trials.score - noise.center)


def head_form(row: SimpleNamespace) -> str:
    """The shape of a trial's head, in words: a linear head, hidden layers, hidden
    layers with a skip, a gate or both, or two encoders fused."""
    enc = str(row.p_encoder) if isinstance(getattr(row, "p_encoder", None), str) else ""
    if "+" in enc or str(getattr(row, "p_head", "")).startswith("concat"):
        return "two encoders fused"
    layers = number(getattr(row, "p_n_hidden_layers", math.nan))
    if layers == 0 or getattr(row, "p_head", None) == "linear":
        return "linear"
    skip = getattr(row, "p_skip", None) in ON
    gate = getattr(row, "p_gate", None) in ON
    return "+ skip + gate" if skip and gate else (
        "+ skip" if skip else "+ gate" if gate else "hidden layers")  # fmt: skip


HEAD_FORMS = [
    "linear",
    "hidden layers",
    "+ skip",
    "+ gate",
    "+ skip + gate",
    "two encoders fused",
]
DROPOUT_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.51)]


def main_encoders(trials: pd.DataFrame) -> list[str]:
    """The encoders with enough finished trials for a column of their own, smallest
    first."""
    counts = (
        trials.p_encoder.value_counts()
        if "p_encoder" in trials
        else pd.Series(dtype=int)
    )
    return [e for e in encoder_order(counts.index) if counts[e] >= MIN_FACET]


def band_shape(
    fig: go.Figure,
    noise: Noise,
    row: int | None = None,
    col: int | None = None,
    accuracy: bool = False,
) -> None:
    """The handed-out baseline with the spread of its reruns as a band: at zero on an
    axis of points above it, at its score on an axis of scores (`accuracy`)."""
    kw = {} if row is None else {"row": row, "col": col}
    at, half = (noise.center, noise.spread) if accuracy else (0.0, 100 * noise.spread)
    fig.add_hrect(y0=at - half, y1=at + half, fillcolor=BLUE_WASH, opacity=0.5,
                  line_width=0, layer="below", **kw)  # fmt: skip
    fig.add_hline(y=at, line_color=BLUE, line_width=1.2, line_dash="dot", **kw)


def box(name: str, y: pd.Series, color: str = INK) -> go.Box:
    """A box with every trial as a point beside it; the page dims it while it lights up
    one student's trials."""
    return go.Box(
        y=y, name=name, boxpoints="all", jitter=0.45, pointpos=0, marker={"size": 4, "color": "rgba(90,84,77,0.35)"},
        line={"color": color, "width": 1.6}, fillcolor="rgba(0,0,0,0)", showlegend=False,
        hovertemplate="%{y:.3f}<extra></extra>", meta={"base": True},
    )  # fmt: skip


# A student's own trials, when the page lights them up.
PICK_MARKER = {"size": 10, "color": ORANGE, "line": {"color": "#ffffff", "width": 1.2}}


def class_trials(runs: pd.DataFrame) -> pd.DataFrame:
    """The trials among `runs`, the class's evidence; the finals ride along only to be
    lit up with their student."""
    return runs[runs.role != "final"] if "role" in runs else runs


def own_trials(
    trials: pd.DataFrame, student: str, names: Iterable[str]
) -> pd.DataFrame:
    """The runs of `student`'s ideas `names` among `trials`: trials and finals."""
    if trials.empty or "idea_name" not in trials:
        return trials.iloc[:0]
    return trials[(trials.student == student) & trials.idea_name.isin(list(names))]


def listed(values: Iterable[object]) -> list[float | None]:
    """Numbers for a picture's data, a missing one as None."""
    return [None if math.isnan(number(v)) else float(number(v)) for v in values]


def pick_trace(
    student: str, part: pd.DataFrame, x: Iterable[object], y: Iterable[object], **meta
) -> go.Scatter:
    """`student`'s own runs as orange points, a final as a star, hidden until the page
    lights that student up."""
    finals = (part.role == "final").tolist() if "role" in part else [False] * len(part)
    marker = {
        **PICK_MARKER,
        "symbol": ["star" if f else "circle" for f in finals],
        "size": [15 if f else PICK_MARKER["size"] for f in finals],
    }
    return go.Scatter(
        x=list(x), y=list(y), mode="markers", visible=False, showlegend=False, marker=marker,
        meta={"student": student, **meta}, text=[plain(f"{student} · {n}") for n in part.idea_name],
        hovertemplate="%{text}<br>%{y:.3f}<extra></extra>",
    )  # fmt: skip


def encoder_runs(df: pd.DataFrame) -> pd.DataFrame:
    """The runs of the handed-out notebook that swap the encoder under one head (named
    `encoder-<name>`)."""
    if df.empty or "name" not in df:
        return df.iloc[:0]
    names = df.name.astype(str)
    return df[
        (df.role == "ablation") & names.str.startswith("encoder-") & df.score.notna()
    ]


Picks = Mapping[str, Iterable[str]]


def fig_encoder(
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    extra: pd.DataFrame | None = None,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """1.5: the score of every finished trial and of every run of the handed-out
    notebook that swapped the encoder (`extra`), per encoder, smallest first. The width
    of a box is what the head choices did with that encoder; the distance between the
    boxes is what the encoder did."""
    fig = go.Figure()
    runs, trials = trials, class_trials(trials)
    # a final on two encoders fused stands as its own encoder: no trial searched it
    fused = runs[(runs.role == "final") & runs.p_encoder.astype(str).str.contains("+", regex=False)] if (
        "role" in runs and "p_encoder" in runs) else runs.iloc[:0]  # fmt: skip
    both = (
        pd.concat([f for f in (trials, extra, fused) if f is not None and len(f)])
        if len(trials)
        else trials
    )
    done = both[both.p_encoder.notna()] if "p_encoder" in both else both.iloc[:0]
    if done.empty:
        empty_note(fig, "no finished trial names its encoder")
        return style(fig, 200)
    names = {}
    for e in encoder_order(done.p_encoder.astype(str)):
        part = done[done.p_encoder == e]
        names[e] = f"{e}<br><span style='font-size:11px'>{len(part)} runs</span>"
        fig.add_trace(box(names[e], part.score))
    for s, idea_names in (picks or {}).items():
        part = own_trials(runs, s, idea_names)
        part = (
            part[part.p_encoder.isin(list(names))]
            if "p_encoder" in part
            else part.iloc[:0]
        )
        if len(part):
            fig.add_trace(pick_trace(s, part, part.p_encoder.map(names), part.score))
    band_shape(fig, noise, accuracy=True)
    fig.update_yaxes(title=metric, zeroline=False)
    fit_range(fig, [done.score])
    return style(fig, height, margin={"l": 64, "r": 16, "t": 16, "b": 64})


def fit_range(
    fig: go.Figure, columns: list[pd.Series], best: bool = True, **where
) -> None:
    """A y axis that holds the bulk of `columns`: from a little under their 2nd
    percentile to their best (`best`), or to their 98th percentile, so a handful of
    runs far out does not squash the rest. How many lie outside is said in the
    picture, above it on the right."""
    values = pd.concat(columns).dropna() if columns else pd.Series(dtype=float)
    if len(values) < 10:
        return
    low = float(values.quantile(0.02))
    high = float(values.max() if best else values.quantile(0.98))
    pad = (high - low) * 0.06 or 1.0
    low, high = low - pad, high + pad
    out = int(((values < low) | (values > high)).sum())
    fig.update_yaxes(range=[low, high], **where)
    if out and not where:
        fig.add_annotation(text=f"{out} outside the axis", xref="paper", yref="paper", x=1, y=1,
                           xanchor="right", yanchor="bottom", showarrow=False,
                           font={"family": FONT, "size": 12, "color": MUTED})  # fmt: skip


def facet_figure(
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    group: str,
    order: list[str],
    title: str,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """Boxes of the scores of `trials` per value of `group` (in `order`), one column per
    encoder with enough trials."""
    runs, trials = trials, class_trials(trials)
    encoders = main_encoders(trials)
    fig = make_subplots(rows=1, cols=max(len(encoders), 1), shared_yaxes=True,
                        horizontal_spacing=0.04, subplot_titles=encoders or None)  # fmt: skip
    if not encoders:
        empty_note(fig, f"no encoder has {MIN_FACET} finished trials")
        return style(fig, 200)
    for c, e in enumerate(encoders, start=1):
        part = trials[trials.p_encoder == e]
        for value in order:
            y = part.score[part[group] == value].dropna()
            if len(y):
                fig.add_trace(box(value, y), row=1, col=c)
        for s, idea_names in (picks or {}).items():
            mine = own_trials(runs[runs.p_encoder == e], s, idea_names)
            mine = mine[mine[group].isin(order)]
            if len(mine):
                fig.add_trace(
                    pick_trace(s, mine, mine[group], mine.score), row=1, col=c
                )
        band_shape(fig, noise, 1, c, accuracy=True)
    fig.update_yaxes(title_text=metric, row=1, col=1)
    fit_range(fig, [trials.score[trials.p_encoder.isin(encoders)]])
    fig.update_xaxes(categoryorder="array", categoryarray=order, title_text=title)
    fig.update_annotations(font={"family": MONO, "size": 13, "color": MUTED})
    return style(fig, height, margin={"l": 64, "r": 16, "t": 36, "b": 56})


def head_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """`trials` of one encoder's head, with its depth (a linear head has none), its
    width (only where there is a hidden layer) and its form; fused encoders left out."""
    t = trials.copy()
    t["form"] = [head_form(r) for r in rows(t)] if len(t) else []
    t = t[t.form != "two encoders fused"].copy()
    depth = column(t, "p_n_hidden_layers")
    t["depth"] = depth.where(t.form != "linear", 0.0)
    t["width"] = column(t, "p_hidden_dim").where(t.depth > 0)
    t["dropout_rate"] = column(t, "p_dropout").where(t.depth > 0)
    return t[t.depth.notna()]


def decade_ticks(values: pd.Series) -> tuple[list[float], list[str]]:
    """Tick values and words for a log axis of FLOPs: every power of ten it spans."""
    v = values.dropna()
    v = v[v > 0]
    if v.empty:
        return [], []
    low, high = math.floor(math.log10(v.min())), math.ceil(math.log10(v.max()))
    ticks = [10.0**e for e in range(low, high + 1)]
    return ticks, [flops_text(t) for t in ticks]


def capacity_axes(t: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """What the capacity picture can put along x (depth, width, FLOPs) and in colour
    (width, depth, FLOPs, dropout): per choice the values of `t` and how its axis or
    colour bar reads them."""
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(t))
    xs = {"depth": t.depth + jitter, "width": t.width, "flops": t.flops}
    depths = sorted({int(v) for v in t.depth.dropna()})
    widths = [w for w in (16, 32, 64, 128, 256, 512, 1024, 2048)
              if not t.width.dropna().empty and t.width.min() / 2 <= w <= t.width.max() * 2]  # fmt: skip
    ft, fw = decade_ticks(t.flops)
    x_axes = {
        "depth": {"type": "linear", "title": "hidden layers", "tickvals": depths, "ticktext": [str(d) for d in depths]},
        "width": {"type": "log", "title": "hidden width", "tickvals": widths,
                  "ticktext": [str(w) for w in widths]},
        "flops": {"type": "log", "title": "FLOPs of training the head", "tickvals": ft, "ticktext": fw},
    }  # fmt: skip
    cs = {
        "width": np.log2(t.width),
        "depth": t.depth,
        "flops": np.log10(t.flops.where(t.flops > 0)),
        "dropout": t.dropout_rate,
    }
    colours: dict[str, dict[str, object]] = {
        "width": {"title": "width", "tickvals": [math.log2(w) for w in widths], "ticktext": [str(w) for w in widths]},
        "depth": {"title": "layers", "tickvals": depths, "ticktext": [str(d) for d in depths]},
        "flops": {"title": "FLOPs", "tickvals": [math.log10(v) for v in ft], "ticktext": fw},
        "dropout": {"title": "dropout", "tickvals": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "ticktext": ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]},
    }  # fmt: skip
    for k, c in colours.items():
        vals = cs[k].dropna()
        low = float(vals.min()) if len(vals) else 0.0
        c["cmin"] = low
        c["cmax"] = float(vals.max()) if len(vals) and vals.max() > low else low + 1
    return xs, x_axes, cs, colours


CAPACITY_X = ("depth", "width", "flops")
CAPACITY_COLOUR = ("width", "depth", "flops", "dropout")


def fig_capacity(
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """1.6: every finished trial's score against the depth of its head, coloured by its
    width, one column per encoder. The page switches what lies along x (depth, width,
    FLOPs) and what the colour says (width, depth, FLOPs, dropout) from the lists this
    figure carries."""
    encoders = main_encoders(class_trials(trials))
    t = (
        head_trials(trials[trials.p_encoder.isin(encoders)])
        if encoders
        else trials.iloc[:0]
    )
    trial = (
        (t.role != "final").to_numpy() if "role" in t else np.ones(len(t), dtype=bool)
    )
    fig = make_subplots(rows=1, cols=max(len(encoders), 1), horizontal_spacing=0.05,
                        subplot_titles=encoders or None)  # fmt: skip
    if t.empty:
        empty_note(fig, "no finished trial with a known head")
        return style(fig, 200)
    xs, x_axes, cs, colours = capacity_axes(t)
    for c, e in enumerate(encoders, start=1):
        cell = (t.p_encoder == e).to_numpy()
        mask = cell & trial
        part = t[mask]
        fig.add_trace(go.Scatter(
            x=listed(xs["depth"][mask]), y=part.score, mode="markers", showlegend=False,
            marker={"size": 7, "color": cs["width"][mask].to_numpy(dtype=float), "coloraxis": "coloraxis", "opacity": 0.8,
                    "line": {"color": PAPER, "width": 0.4}},
            meta={"base": True, "xs": {k: listed(v[mask]) for k, v in xs.items()},
                  "cs": {k: listed(v[mask]) for k, v in cs.items()}},
            text=[plain(f"{r.student} · {r.idea_name or 'handed-out search'} · {int(r.depth)} layers"
                        + (f" × {int(r.width)}" if not math.isnan(number(r.width)) else "")) for r in rows(part)],
            hovertemplate="%{text}<br>%{y:.3f}<extra></extra>",
        ), row=1, col=c)  # fmt: skip
        for s, idea_names in (picks or {}).items():
            mine = own_trials(t[cell], s, idea_names)
            if mine.empty:
                continue
            sub = t.index.isin(mine.index)
            fig.add_trace(pick_trace(s, mine, listed(xs["depth"][sub]), mine.score,
                                     xs={k: listed(v[sub]) for k, v in xs.items()}), row=1, col=c)  # fmt: skip
        band_shape(fig, noise, 1, c, accuracy=True)
        fit_range(fig, [part.score], row=1, col=c)
    first = colours["width"]
    fig.update_layout(
        coloraxis={"colorscale": "Viridis", "cmin": first["cmin"], "cmax": first["cmax"],
                   "colorbar": {"title": {"text": first["title"]}, "tickvals": first["tickvals"],
                                "ticktext": first["ticktext"], "thickness": 12, "len": 0.8}},
        meta={"x": x_axes, "colour": colours},
    )  # fmt: skip
    a = x_axes["depth"]
    fig.update_xaxes(
        type=a["type"],
        tickvals=a["tickvals"],
        ticktext=a["ticktext"],
        title_text=a["title"],
        matches="x",
    )
    fig.update_yaxes(title_text=metric, row=1, col=1)
    fig.update_annotations(font={"family": MONO, "size": 13, "color": MUTED})
    return style(fig, height, margin={"l": 64, "r": 16, "t": 36, "b": 56})


def capacity_switch(fig_id: str) -> str:
    """The buttons that set what the capacity picture puts along x and in colour."""

    def group(kind: str, words: str, keys: Iterable[str], on: str) -> str:
        return f"<span>{words}</span>" + "".join(
            f"<button type='button' data-kind='{kind}' data-key='{k}' aria-pressed='{str(k == on).lower()}'>"
            f"{'FLOPs' if k == 'flops' else k}</button>" for k in keys
        )  # fmt: skip

    return (
        f"<div class='switch axis-switch' data-fig='{fig_id}' role='group' aria-label='What the picture shows'>"
        + group("x", "along x", CAPACITY_X, "depth")
        + "<span class='gap'></span>"
        + group("colour", "colour", CAPACITY_COLOUR, "width")
        + "</div>"
    )


# The size of a head in parameters, for the curves of overfitting.
HEAD_SIZES = (
    (0, 1e5, "under 100k params"),
    (1e5, 4e5, "100k–400k params"),
    (4e5, math.inf, "over 400k params"),
)
# Dropout in three groups, each its own hue from Okabe and Ito.
DROPOUT_GROUPS = (
    (0.0, 0.05, "no dropout", "#0072B2"),
    (0.05, 0.25, "dropout 0.05–0.25", "#009E73"),
    (0.25, 1.01, "dropout 0.25 and more", "#CC79A7"),
)
MIN_CURVES = 5  # curves that reach an epoch before its median is drawn


def median_curve(curves: list[list[float]]) -> list[float]:
    """The median of `curves` after every epoch, as far as `MIN_CURVES` of them reach."""
    out = []
    for e in range(max((len(c) for c in curves), default=0)):
        values = [c[e] for c in curves if len(c) > e]
        if len(values) < MIN_CURVES:
            break
        out.append(float(np.median(values)))
    return out


def size_title(lo: float, hi: float, words: str, rows: int | None) -> str:
    """A head size in parameters, and, with the training rows known, in parameters per
    training image and in FLOPs per epoch (6 × parameters × rows): FLOPs are parameters
    times the data seen, so within one epoch the two order heads alike."""
    if not rows:
        return words

    def per_image(v: float) -> str:
        return f"{v / rows:.0f}"

    def per_epoch(v: float) -> str:
        return flops_text(float(f"{6 * v * rows:.2g}"))

    if lo <= 0:
        image, epoch = f"under {per_image(hi)}", f"under {per_epoch(hi)}"
    elif not math.isfinite(hi):
        image, epoch = f"over {per_image(lo)}", f"over {per_epoch(lo)}"
    else:
        image = f"{per_image(lo)}–{per_image(hi)}"
        epoch = f"{per_epoch(lo).removesuffix(' G')}–{per_epoch(hi)}"
    return f"{words}<br>{image} per training image<br>{epoch} FLOPs per epoch"


def fig_overfitting(
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """1.9: the median train loss (dashed) and test loss (solid) after every epoch, per
    dropout group, one row per encoder and one column per size of the head; a test
    loss that turns up while the train loss keeps falling is the head learning its
    training images by heart."""
    has = (
        trials[trials.losses.map(lambda v: isinstance(v, (list, tuple)))]
        if "losses" in trials
        else trials.iloc[:0]
    )
    encoders = main_encoders(class_trials(has))
    if not encoders:
        fig = go.Figure()
        empty_note(fig, "no encoder has enough trials that logged their losses")
        return style(fig, 200)
    sizes = column(has, "n_params")
    drop = column(has, "p_dropout").fillna(0.0)
    titles = [
        size_title(lo, hi, words, TRAIN_ROWS.get(noise.dataset))
        for lo, hi, words in HEAD_SIZES
    ]
    titles += [""] * (len(HEAD_SIZES) * (len(encoders) - 1))
    fig = make_subplots(rows=len(encoders), cols=len(HEAD_SIZES), shared_xaxes=True, shared_yaxes="rows",
                        horizontal_spacing=0.04, vertical_spacing=0.1, subplot_titles=titles)  # fmt: skip
    shown = set()
    for r_, e in enumerate(encoders, start=1):
        top = 0.0
        for c, (lo, hi, _) in enumerate(HEAD_SIZES, start=1):
            cell = has[(has.p_encoder == e) & (sizes >= lo) & (sizes < hi)]
            for dlo, dhi, words, colour in DROPOUT_GROUPS:
                part = class_trials(
                    cell[(drop[cell.index] >= dlo) & (drop[cell.index] < dhi)]
                )
                test = median_curve([list(v[1]) for v in part.losses])
                train = median_curve([list(v[0]) for v in part.losses])
                if not test:
                    continue
                top = max(top, max(test))
                epochs = list(range(1, len(test) + 1))
                fig.add_scatter(x=epochs, y=test, mode="lines", name=words, legendgroup=words,
                                showlegend=words not in shown, line={"color": colour, "width": 2.4},
                                meta={"base": True}, row=r_, col=c,
                                hovertemplate=f"{words} · {len(part)} trials<br>epoch %{{x}}: test loss %{{y:.2f}}<extra></extra>")  # fmt: skip
                fig.add_scatter(x=list(range(1, len(train) + 1)), y=train, mode="lines", legendgroup=words,
                                showlegend=False, line={"color": colour, "width": 1.4, "dash": "dash"},
                                meta={"base": True}, row=r_, col=c,
                                hovertemplate=f"{words}<br>epoch %{{x}}: train loss %{{y:.2f}}<extra></extra>")  # fmt: skip
                shown.add(words)
            for s, idea_names in (picks or {}).items():
                mine = own_trials(cell, s, idea_names)
                xs: list[float | None] = []
                ys: list[float | None] = []
                for v in mine.losses:
                    xs += [*range(1, len(v[1]) + 1), None]
                    ys += [*v[1], None]
                if xs:
                    fig.add_scatter(x=xs, y=ys, mode="lines", visible=False, showlegend=False,
                                    line={"color": ORANGE, "width": 1.3}, meta={"student": s},
                                    hovertemplate=f"{plain(s)}<br>epoch %{{x}}: test loss %{{y:.2f}}<extra></extra>",
                                    row=r_, col=c)  # fmt: skip
        fig.update_yaxes(range=[0, max(top * 1.15, 0.5)], row=r_)
        fig.update_yaxes(title_text=f"{plain(e)}<br>loss", row=r_, col=1)
    fig.add_scatter(x=[None], y=[None], mode="lines", name="train loss (dashed)",
                    line={"color": MUTED, "width": 1.4, "dash": "dash"}, meta={"base": True})  # fmt: skip
    fig.update_xaxes(title_text="epoch", row=len(encoders))
    fig.update_annotations(font={"family": MONO, "size": 12, "color": MUTED})
    return style(fig, height, margin={"l": 84, "r": 16, "t": 112, "b": 48},
                 legend={"orientation": "h", "y": 1.0, "yanchor": "top", "x": 0, "yref": "container"})  # fmt: skip


# A colour per shape of the head, from Okabe and Ito.
HEAD_COLOURS = {
    "linear": "#000000",
    "hidden layers": "#0072B2",
    "+ skip": "#009E73",
    "+ gate": "#E69F00",
    "+ skip + gate": "#CC79A7",
    "two encoders fused": "#D55E00",
}


def fig_simpler(
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """1.10: every finished trial's score against the FLOPs of training its head,
    coloured by the shape of the head, one column per encoder: what a more complex head
    costs, and whether the score pays for it."""
    encoders = main_encoders(class_trials(trials))
    t = (
        trials[trials.p_encoder.isin(encoders) & trials.flops.notna()].copy()
        if encoders
        else trials.iloc[:0]
    )
    fig = make_subplots(rows=1, cols=max(len(encoders), 1), horizontal_spacing=0.05,
                        subplot_titles=encoders or None)  # fmt: skip
    if t.empty:
        empty_note(fig, "no finished trial with known FLOPs")
        return style(fig, 200)
    t["form"] = [head_form(r) for r in rows(t)]
    shown = set()
    for c, e in enumerate(encoders, start=1):
        part = t[t.p_encoder == e]
        for form, colour in HEAD_COLOURS.items():
            p = class_trials(part[part.form == form])
            if p.empty:
                continue
            fig.add_scatter(
                x=p.flops, y=p.score, mode="markers", name=form, legendgroup=form, showlegend=form not in shown,
                marker={"size": 7, "color": colour, "opacity": 0.75, "line": {"color": PAPER, "width": 0.4}},
                meta={"base": True}, text=[plain(f"{r.student} · {r.idea_name or 'handed-out search'}") for r in rows(p)],
                hovertemplate="%{text}<br>%{y:.3f} at %{x:.2s} FLOPs<extra></extra>", row=1, col=c,
            )  # fmt: skip
            shown.add(form)
        for s, idea_names in (picks or {}).items():
            mine = own_trials(part, s, idea_names)
            if len(mine):
                fig.add_trace(pick_trace(s, mine, mine.flops, mine.score), row=1, col=c)
        band_shape(fig, noise, 1, c, accuracy=True)
        fit_range(fig, [part.score], row=1, col=c)
    ticks, words = decade_ticks(t.flops)
    fig.update_xaxes(
        type="log",
        tickvals=ticks,
        ticktext=words,
        title_text="FLOPs of training the head",
        matches="x",
    )
    fig.update_yaxes(title_text=metric, row=1, col=1)
    fig.update_annotations(font={"family": MONO, "size": 13, "color": MUTED})
    return style(fig, height, margin={"l": 64, "r": 16, "t": 76, "b": 56},
                 legend={"orientation": "h", "y": 1.0, "yanchor": "top", "x": 0, "yref": "container"})  # fmt: skip


THEME_FIGURES = {
    "capacity": fig_capacity,
    "overfitting": fig_overfitting,
    "simpler": fig_simpler,
}


def theme_figure(
    theme: Theme,
    trials: pd.DataFrame,
    noise: Noise,
    height: int,
    df: pd.DataFrame,
    picks: Picks | None = None,
    metric: str = METRIC,
) -> go.Figure:
    """The class's evidence on a theme: its own picture, or, for a theme without one,
    the score against each of its axes. `picks` names, per student, the ideas whose
    trials the page can light up."""
    if theme.key == "encoder":
        return fig_encoder(trials, noise, height, encoder_runs(df), picks, metric)
    if theme.key in THEME_FIGURES:
        return THEME_FIGURES[theme.key](trials, noise, height, picks, metric)
    axis = f"p_{theme.axes[0]}" if theme.axes else ""
    if axis not in trials:
        fig = go.Figure()
        empty_note(fig, "no trial varies this")
        return style(fig, 200)
    t = trials.copy()
    t["value"] = t[axis].astype(str)
    order = sorted(t.value.unique(), key=lambda v: order_key(v, theme.axes[0]))
    return facet_figure(t, noise, height, "value", order,
                        PAIR_NAMES.get(theme.axes[0], theme.axes[0]), picks, metric)  # fmt: skip


def tag_chips(tags: Iterable[str]) -> str:
    """An idea's tags as chips."""
    return "".join(f"<span class='chip'>{esc(t)}</span>" for t in tags)


def theme_students(items: list[SimpleNamespace], tags: Mapping[str, list[str]]) -> str:
    """The students with an idea under a theme, in the order their first one ran: a
    button that lights up their runs in the picture, and each idea as its name, its
    tags and its final score, a link that opens it."""
    order: dict[str, list[SimpleNamespace]] = {}
    for r in items:
        order.setdefault(lane_of(r), []).append(r)
    out = []
    for s, mine in order.items():
        ideas_ = "".join(
            f"<li><a href='#{esc(r.name)}' title='{esc(r.idea)}'><span class='idea-name'>{esc(r.name)}</span>"
            f"<span class='chips'>{tag_chips(tags.get(r.name, []))}</span>"
            f"<span class='num'>{'' if math.isnan(number(r.final)) else f'{r.final:.3f}'}</span></a></li>"
            for r in mine
        )  # fmt: skip
        who = (f"<button type='button' class='pick' data-student='{esc(s)}' aria-pressed='false'>"
               f"<b>{esc(s)}</b><span>{plural(len(mine), 'idea')}</span></button>")  # fmt: skip
        out.append(f"<div class='tstudent'>{who}<ul>{ideas_}</ul></div>")
    return "".join(out)


def themes_view(
    df: pd.DataFrame,
    ids: pd.DataFrame,
    noise: Noise | None,
    trials: pd.DataFrame,
    lesson: int,
    tags: Mapping[str, list[str]],
    metric: str = METRIC,
) -> str:
    """Per learning goal: the class's evidence as one picture, beside the students who
    investigated it, whose own trials the picture lights up. A rail lists the goals."""
    themes = themes_of(lesson, ids)
    grouped = ideas_by_theme(df, ids, themes)
    shown = [t for t in themes if noise is not None] + (
        [OTHER] if grouped[OTHER.key] else []
    )
    if not shown:
        return missing("No idea of a student submitted yet." if ids[ids.author != COURSE].empty
                       else "Too few runs of the handed-out baseline to read the evidence against.")  # fmt: skip
    finals = finished(df[df.role == "final"]).copy()
    finals["flops"] = train_flops(finals)
    done = pd.concat([finished(trials), finals], ignore_index=True)
    rail, panels = [], []
    for i, t in enumerate(shown):
        mine = grouped[t.key]
        picks: dict[str, list[str]] = {}
        for r in mine:
            picks.setdefault(lane_of(r), []).append(r.name)
        fig_id = f"theme-fig-{t.key}"
        picture, lit = "", None
        if t.key != OTHER.key and noise is not None:
            fig = theme_figure(t, done, noise, 600, df, picks, metric)
            lit = {
                tr.meta["student"]
                for tr in fig.data
                if isinstance(tr.meta, dict) and "student" in tr.meta
            }
            picture = (capacity_switch(fig_id) if t.key == "capacity" else "") + figure(fig, div_id=fig_id)  # fmt: skip
            # a student the picture holds no run of is left off its list
            mine = [r for r in mine if lane_of(r) in lit]
        rail.append(
            f"<button type='button' class='theme-tab' data-theme='{t.key}' aria-selected='{str(i == 0).lower()}'>"
            f"<span class='goal num'>{esc(t.goal)}</span><span class='ttl'>{esc(t.title)}</span>"
            f"<span class='cnt'>{plural(len(mine), 'idea')}</span></button>"
        )
        people_ = theme_students(mine, tags) or missing(
            "No idea of a student investigated this."
        )
        panels.append(
            f"<div class='theme-panel' data-theme='{t.key}'{'' if i == 0 else ' hidden'}>"
            f"<div class='theme-evidence'>{picture}</div>"
            f"<div class='theme-ideas'>{people_}</div></div>"
        )
    return (
        f"<div class='themes'><nav class='theme-rail' aria-label='Learning goals'>{''.join(rail)}</nav>"
        f"{''.join(panels)}</div>"
    )


def lane_of(r: SimpleNamespace) -> str:
    """The student whose lane an idea stands in: the one whose study stands for it."""
    return (
        r.student
        if isinstance(r.student, str)
        else (r.students[0] if r.students else r.author)
    )


def found_line(
    r: SimpleNamespace,
    noise: Noise | None,
    by: Mapping[str, SimpleNamespace],
    eds: pd.DataFrame,
    brief: bool = False,
) -> list[tuple[str, str]]:
    """What an idea found, as lines of (class, html): its final with its seeds, then its
    final against the final of each idea of a student it builds on, with the spread of
    that difference, then its score against the handed-out baseline with the spread of
    its reruns. For an idea that builds on a student's idea the baseline line is
    context, grey, and `brief` (a card) leaves it out."""
    if math.isnan(number(r.final)):
        return [("none", "no final submitted yet")]
    seeds = ""
    if not math.isnan(number(r.final_std)) and number(r.final_n) > 1:
        seeds = f" <span class='pm'>± {100 * r.final_std:.1f} pt · {int(r.final_n)} seeds</span>"
    out = [("final", f"<b class='num'>{r.final:.3f}</b>{seeds}")]
    on_parent = False
    for e in rows(eds[eds.child == r.name]):
        if e.parent_author == COURSE:
            continue
        on_parent = True
        parent = by.get(e.parent)
        who = esc(e.parent) if parent is None or parent.author == r.author else (
            f"{esc(lane_of(parent))}’s {esc(e.parent)}")  # fmt: skip
        if e.number:
            out.append(
                (
                    "diff",
                    f"<span class='num'>{signed(e.diff_pt)}{pm(e.spread_pt)} pt</span> on {who}",
                )
            )
        else:
            out.append(("none", f"on {who}: {esc(e.why)}"))
    if noise is not None and noise.dataset == r.dataset and not (brief and on_parent):
        gain = 100 * (r.final - noise.center)
        text = f"<span class='num'>{signed(gain)}{pm(100 * noise.spread)} pt</span>"
        out.append(
            ("base" if on_parent else "diff", f"{text} on the handed-out baseline")
        )
    return out


def pm(spread_pt: float) -> str:
    """` ± spread` in points, nothing when there is no spread."""
    return "" if math.isnan(number(spread_pt)) else f" ± {spread_pt:.1f}"


def card_found(r: SimpleNamespace, noise: Noise | None, eds: pd.DataFrame) -> str:
    """An idea's result on one line: its final, then how far it lies from the final of
    each student's idea it builds on, or, building on the handed-out idea alone, from
    the handed-out baseline, each with its spread."""
    if math.isnan(number(r.final)):
        return "<span class='none'>no final yet</span>"
    out = [f"<b class='num'>{r.final:.3f}</b>"]
    on_parent = False
    for e in rows(eds[eds.child == r.name]):
        if e.parent_author == COURSE:
            continue
        on_parent = True
        if e.number:
            out.append(f"<span class='d' title='on {esc(e.parent)}'>"
                       f"<span class='num'>{signed(e.diff_pt)}{pm(e.spread_pt)}</span> pt on its parent</span>")  # fmt: skip
    if not on_parent and noise is not None and noise.dataset == r.dataset:
        gain = 100 * (r.final - noise.center)
        out.append(
            f"<span class='d'><span class='num'>{signed(gain)}{pm(100 * noise.spread)}</span> pt on the baseline</span>"
        )
    return " ".join(out)


def card(
    r: SimpleNamespace,
    noise: Noise | None,
    by: Mapping[str, SimpleNamespace],
    eds: pd.DataFrame,
    elsewhere: Mapping[str, int],
    metric: str,
    tags: Mapping[str, list[str]],
) -> str:
    """One idea in its lane: its name and time, what it investigated as tags, and what
    it found; the question is the card's title, and in full in the drawer."""
    when = moment(r.start)
    notes = []
    for p in r.parents:
        parent = by.get(p)
        if parent is None:
            lesson = elsewhere.get(p)
            notes.append(f"<span class='tag'>on {esc(p)}"
                         f"{f' (lesson {lesson})' if lesson is not None else ', not submitted'}</span>")  # fmt: skip
        elif parent.author != COURSE and lane_of(parent) != lane_of(r):
            notes.append(
                f"<span class='tag cross'>on {esc(lane_of(parent))}’s idea</span>"
            )
    if isinstance(r.objective, str) and r.objective != metric:
        notes.append(f"<span class='tag'>searched on {esc(r.objective)}</span>")
    return (
        f"<a class='idea' href='#{esc(r.name)}' id='card-{esc(r.name)}' data-name='{esc(r.name)}' "
        f"data-parents='{esc(json.dumps([p for p in r.parents if p in by and by[p].author != COURSE]))}' "
        f"title='{esc(r.idea)}'>"
        f"<span class='idea-head'><span class='idea-name'>{esc(r.name)}</span>"
        f"<span class='idea-time'>{when.strftime('%a %H:%M') if when is not None else ''}</span></span>"
        f"<span class='chips'>{tag_chips(tags.get(r.name, []))}{''.join(notes)}</span>"
        f"<span class='res'>{card_found(r, noise, eds)}</span></a>"
    )


def lanes(
    ids: pd.DataFrame,
    eds: pd.DataFrame,
    noise: Noise | None,
    elsewhere: Mapping[str, int],
    metric: str = METRIC,
    tags: Mapping[str, list[str]] | None = None,
) -> str:
    """The ideas of the class: one lane per student, in the order their first idea ran,
    and in a lane the student's ideas in the order they ran, each card naming its day.
    The links between cards are drawn by the page's script from `data-parents`."""
    own = ids[ids.author != COURSE]
    if own.empty:
        return missing("No submitted run carries an idea of a student yet.")
    by = {r.name: r for r in rows(ids)}

    def start(r: SimpleNamespace) -> float:
        return number(r.start) if not math.isnan(number(r.start)) else math.inf

    first: dict[str, float] = {}
    for r in rows(own):
        first[lane_of(r)] = min(first.get(lane_of(r), math.inf), start(r))
    body = []
    for lane in sorted(first, key=lambda s: (first[s], s)):
        mine = sorted((r for r in rows(own) if lane_of(r) == lane), key=start)
        cards = "".join(
            card(r, noise, by, eds, elsewhere, metric, tags or {}) for r in mine
        )
        body.append(
            f"<div class='lane-label'><b>{esc(lane)}</b><span>{plural(len(mine), 'idea')}</span></div>"
            f"<div class='cell'>{cards}</div>"
        )
    return (
        "<div class='lanes-wrap'><div class='lanes'>"
        f"<svg class='links' aria-hidden='true'></svg>{''.join(body)}</div></div>"
    )


def ideas_list(
    ids: pd.DataFrame, eds: pd.DataFrame, noise: Noise | None, metric: str = METRIC
) -> str:
    """Every student's idea in their own words: what they asked, what they expected,
    and what they found, lane by lane."""
    own = ids[ids.author != COURSE]
    if own.empty:
        return ""
    by = {r.name: r for r in rows(ids)}
    first: dict[str, float] = {}
    for r in rows(own):
        first[lane_of(r)] = min(first.get(lane_of(r), math.inf), number(r.start))
    items = sorted(
        rows(own), key=lambda r: (first[lane_of(r)], lane_of(r), number(r.start))
    )
    out = []
    for r in items:
        when = moment(r.start)
        found = "".join(
            f"<li class='{k}'>{v}</li>" for k, v in found_line(r, noise, by, eds)
        )
        out.append(
            f"<article class='said' id='said-{esc(r.name)}'><header><a href='#{esc(r.name)}'>{esc(r.name)}</a>"
            f"<span>{esc(lane_of(r))}{' · ' + when.strftime('%a %-d %b') if when is not None else ''}</span></header>"
            f"<dl><dt>Asked</dt><dd>{esc(r.idea) or '<i>not written</i>'}</dd>"
            f"<dt>Expected</dt><dd>{esc(r.expects) or '<i>not said</i>'}</dd>"
            f"<dt>Found</dt><dd><ul class='found'>{found}</ul></dd></dl></article>"
        )
    return f"<div class='said-list'>{''.join(out)}</div>"


def effort_table(df: pd.DataFrame, ids: pd.DataFrame) -> str:
    """Per student: ideas, finished trials, trials the pruner stopped, runs of the
    handed-out notebook, and minutes of training on their own machine."""
    rows_ = []
    own = ids[ids.author != COURSE]
    lanes_ = Counter(lane_of(r) for r in rows(own))
    for student, g in df.groupby("student"):
        trials = g[g.role == "trial"]
        rows_.append((
            str(student), lanes_.get(student, 0),
            int(((trials.status == "COMPLETE") & (trials.origin == "study")).sum()),
            int(((trials.status == "PRUNED") & (trials.origin == "study")).sum()),
            int((g.origin == "notebook").sum()),
            float(g.train_seconds.fillna(0).sum()) / 60,
        ))  # fmt: skip
    rows_.sort(key=lambda x: (is_teacher(x[0]), -x[1], -x[2], x[0]))
    body = "".join(
        f"<tr{' class=teacher' if is_teacher(s) else ''}><th scope='row'>{esc(s)}"
        f"{', for reference' if is_teacher(s) else ''}</th>"
        f"<td>{i}</td><td>{f}</td><td>{p}</td><td>{n}</td><td>{m:.0f}</td></tr>"
        for s, i, f, p, n, m in rows_
    )
    return (
        "<div class='table-wrap'><table class='effort'><thead><tr><th scope='col'>student</th>"
        "<th scope='col'>ideas</th><th scope='col'>trials finished</th>"
        "<th scope='col'>stopped early by the pruner</th><th scope='col'>runs of the handed-out notebook</th>"
        f"<th scope='col'>minutes of training</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def missing(text: str) -> str:
    """A section's line in place of a picture that has nothing to draw."""
    return f'<p class="missing">{html.escape(text)}</p>'


def figure(fig: go.Figure | None, div_id: str | None = None) -> str:
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "displayModeBar": False, "responsive": True})  # fmt: skip


def section(sid: str, title: str, lead: str, body: str) -> str:
    return (
        f"<section id='{sid}'><h2>{esc(title)}</h2>"
        f"<p class='lead'>{lead}</p>{body}</section>"
    )


def study_details(df: pd.DataFrame, r: SimpleNamespace) -> str:
    """An idea's study: the space it searched, the winner's config (the searched axes
    marked), the options of the design on or off in the winner, its budget and its
    design script."""
    group = df[(df.student == r.student) & (df.study == r.study)]
    finals = group[group.role == "final"].sort_values("start")
    if finals.empty:
        return ""
    final = finals.iloc[-1]
    params = {
        k[2:]: v for k, v in final.items()
        if k.startswith("p_") and v is not None and not (isinstance(v, float) and math.isnan(v))
    }  # fmt: skip
    space = first_of(group.search_space, str)
    axes = str(first_of(group.searched_axes, str) or "").split(",")
    order = [a for a in axes if a in params] + [a for a in params if a not in axes]
    winner = "".join(
        f"<span{' class=on' if a in axes else ''}>{esc(a)} = {esc(chip_value(params[a]))}</span>"
        for a in order
    )
    script = final.design_script or first_of(group.design_script, str)
    options = list(dict.fromkeys(OPTION.findall(str(script or ""))))
    chips = []
    for o in options:
        v = params.get(o)
        state = "on" if v in ON else ("off" if v in OFF else "")
        chips.append(f"<span class='{state}'>{esc(o)} {state}</span>" if state
                     else f"<span>{esc(o)}: not in the winner’s config</span>")  # fmt: skip
    budget = first_of(group.budget, dict)
    spent = budget_text(budget, int((group.role == "trial").sum()))
    design = (
        f"<details><summary>design.py · {len(str(script).splitlines())} lines</summary>"
        f"<pre>{esc(script)}</pre></details>"
        if script
        else "<i>not submitted</i>"
    )
    return (
        "<dl class='kv'>"
        f"<dt>space</dt><dd><code>{esc(space or 'not submitted')}</code></dd>"
        f"<dt>winner</dt><dd class='chips'>{winner or '—'}</dd>"
        + (f"<dt>options</dt><dd class='chips'>{''.join(chips)}</dd>" if chips else "")
        + f"<dt>budget</dt><dd>{esc(spent)}</dd><dt>design</dt><dd>{design}</dd></dl>"
    )


def drawer_data(
    df: pd.DataFrame,
    ids: pd.DataFrame,
    eds: pd.DataFrame,
    noise: Noise | None,
    metric: str,
) -> str:
    """Per student's idea, what the drawer shows: its words and result as HTML, and its
    pictures (trials in order, parallel coordinates, learning curves) as plotly JSON;
    safe inside a script tag."""
    by = {r.name: r for r in rows(ids)}
    children: dict[str, list[str]] = {}
    for e in rows(eds):
        children.setdefault(e.parent, []).append(e.child)
    parts = {}
    for r in rows(ids[ids.author != COURSE]):
        trials = idea_trials(df, ids, r.name)
        parent = next(
            (by[p] for p in r.parents if p in by and by[p].author != COURSE), None
        )
        lines, below = fig_lines(trials, r.axes or AXES, noise, metric, height=340)
        when = moment(r.start)
        links = [
            f"<a href='#{esc(p)}'>{esc(p)}</a>" if p in by and by[p].author != COURSE
            else ("the handed-out baseline" if p in by else esc(p))
            for p in r.parents
        ]  # fmt: skip
        kids = [
            f"<a href='#{esc(c)}'>{esc(c)}</a> ({esc(lane_of(by[c]))})"
            for c in children.get(r.name, [])
        ]
        found = "".join(
            f"<li class='{k}'>{v}</li>" for k, v in found_line(r, noise, by, eds)
        )
        n_done = int((trials.status == "COMPLETE").sum())
        studies = ""
        if len(r.studies) > 1:
            rest = "; ".join(
                f"{esc(st)} ({plural(n, 'trial')})" for _, st, n in r.studies[1:]
            )
            studies = (
                f"<p class='note'>{len(r.studies)} studies of this idea: shown "
                f"{esc(r.studies[0][1])} ({plural(r.studies[0][2], 'trial')}); also {rest}</p>"
            )
        head = (
            f"<h2 id='drawer-title'>{esc(r.name)}</h2>"
            f"<p class='meta'>{esc(lane_of(r))}{' · ' + when.strftime('%a %-d %b, %H:%M') if when is not None else ''}"
            f" · {plural(len(trials), 'trial')}, {n_done} finished</p>{studies}"
            f"<dl class='words'><dt>Asked</dt><dd>{esc(r.idea) or '<i>not written</i>'}</dd>"
            f"<dt>Expected</dt><dd>{esc(r.expects) or '<i>not said</i>'}</dd>"
            f"<dt>Found</dt><dd><ul class='found'>{found}</ul></dd>"
            f"<dt>Builds on</dt><dd>{', '.join(links) or 'nothing named'}</dd>"
            + (f"<dt>Built on by</dt><dd>{', '.join(kids)}</dd>" if kids else "")
            + "</dl>"
        )
        lines_note = (f"{plural(below, 'finished trial')} under the baseline left out"
                      if below and noise is not None else "")  # fmt: skip
        parts[r.name] = {
            "head": head,
            "details": study_details(df, r) if isinstance(r.study, str) else "",
            "lines_note": lines_note,
            "figs": {
                "trials": json.loads(
                    fig_trials(trials, r, parent, noise, metric).to_json()
                ),
                "lines": json.loads(lines.to_json()),
                "curves": json.loads(fig_curves(trials, noise, metric).to_json()),
            },
        }
    return json.dumps(parts).replace("</", "<\\/")


DRAWER = """
<aside class="drawer" id="drawer" aria-labelledby="drawer-title" tabindex="-1" hidden>
<div class="drawer-bar"><a class="close" href="#" id="drawer-close">Close</a></div>
<div class="drawer-body"><div id="drawer-head"></div>
<h3>Trials in the order they ran</h3><div id="fig-trials"></div>
<h3>What its trials above the baseline tried</h3><p class="note" id="lines-note"></p><div id="fig-lines"></div>
<h3>Learning curves of its five best trials</h3><div id="fig-curves"></div>
<h3>The study</h3><div id="drawer-details"></div></div></aside>
"""

SCRIPT = r"""
(function () {
  const data = document.getElementById('drawer-data');
  const D = data ? JSON.parse(data.textContent) : {};
  const CONFIG = {displaylogo: false, displayModeBar: false, responsive: true};
  const lanes = document.querySelector('.lanes');
  const svg = lanes ? lanes.querySelector('svg.links') : null;
  const drawer = document.getElementById('drawer');
  const NS = 'http://www.w3.org/2000/svg';

  function drawLinks() {
    if (!svg) return;
    const box = lanes.getBoundingClientRect();
    svg.setAttribute('width', lanes.scrollWidth);
    svg.setAttribute('height', lanes.scrollHeight);
    while (svg.lastChild) svg.removeChild(svg.lastChild);
    svg.insertAdjacentHTML('afterbegin',
      '<defs><marker id="head-own" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" class="head-own"/></marker>' +
      '<marker id="head-cross" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" class="head-cross"/></marker></defs>');
    const used = {};
    lanes.querySelectorAll('a.idea').forEach(function (child) {
      const parents = JSON.parse(child.dataset.parents || '[]');
      parents.forEach(function (p) {
        const parent = document.getElementById('card-' + p);
        if (!parent) return;
        const a = parent.getBoundingClientRect(), b = child.getBoundingClientRect();
        const cross = laneOf(parent) !== laneOf(child);
        let d;
        if (b.left > a.right - 4) {
          // out of the parent's right edge, down the gutter beside its column, into the child
          const x1 = a.right - box.left, y1 = a.top + 30 - box.top;
          const x2 = b.left - box.left - 2, y2 = b.top + 30 - box.top;
          const k = used[p] = (used[p] || 0) + 1;
          if (cross) {
            // down the parent's gutter to its lane's lower edge, along that edge (the cell's
            // padding, where no card stands) to the gutter left of the child, down into it
            const ga = x1 + 9, gb = x2 - 9 - ((k - 1) % 2) * 5;
            const edge = parent.closest('.cell').getBoundingClientRect().bottom - box.top - 6;
            d = orth([[x1, y1], [ga, y1], [ga, edge], [gb, edge], [gb, y2], [x2, y2]]);
          } else {
            const gx = x1 + 12 + ((k - 1) % 3) * 5;
            d = Math.abs(y2 - y1) < 2 ? 'M' + x1 + ' ' + y1 + ' H' + x2 :
              orth([[x1, y1], [gx, y1], [gx, y2], [x2, y2]]);
          }
        } else {
          const x1 = a.left - box.left, y1 = a.top + 28 - box.top;
          const x2 = b.left - box.left - 2, y2 = b.top + 28 - box.top;
          const bend = x1 - 22;
          d = 'M' + x1 + ' ' + y1 + ' C' + bend + ' ' + y1 + ' ' + bend + ' ' + y2 + ' ' + x2 + ' ' + y2;
        }
        const path = document.createElementNS(NS, 'path');
        path.setAttribute('d', d);
        path.setAttribute('class', cross ? 'cross' : 'own');
        path.setAttribute('marker-end', cross ? 'url(#head-cross)' : 'url(#head-own)');
        path.dataset.parent = p; path.dataset.child = child.dataset.name;
        svg.appendChild(path);
      });
    });
  }
  // a path through `pts`, running across or down, its corners rounded
  function orth(pts) {
    let d = 'M' + pts[0][0] + ' ' + pts[0][1];
    for (let i = 1; i < pts.length - 1; i++) {
      const [px, py] = pts[i - 1], [x, y] = pts[i], [nx, ny] = pts[i + 1];
      const r = Math.min(9, Math.hypot(x - px, y - py) / 2, Math.hypot(nx - x, ny - y) / 2);
      const ax = x - Math.sign(x - px) * r, ay = y - Math.sign(y - py) * r;
      const bx = x + Math.sign(nx - x) * r, by = y + Math.sign(ny - y) * r;
      d += ' L' + ax + ' ' + ay + ' Q' + x + ' ' + y + ' ' + bx + ' ' + by;
    }
    const last = pts[pts.length - 1];
    return d + ' L' + last[0] + ' ' + last[1];
  }
  function laneOf(card) {
    let el = card.closest('.cell');
    while (el && !el.classList.contains('lane-label')) el = el.previousElementSibling;
    return el ? el.textContent : '';
  }

  let opener = null;
  function show(name) {
    document.querySelectorAll('a.idea.on').forEach(function (c) { c.classList.remove('on'); });
    document.querySelectorAll('svg.links path.on').forEach(function (c) { c.classList.remove('on'); });
    if (!drawer) return;
    if (!(name in D)) {
      const was = !drawer.hidden;
      drawer.hidden = true; document.body.classList.remove('drawer-open');
      if (was && opener && document.contains(opener)) opener.focus({preventScroll: true});
      return;
    }
    if (drawer.hidden) opener = document.activeElement;
    const p = D[name];
    document.getElementById('drawer-head').innerHTML = p.head;
    document.getElementById('drawer-details').innerHTML = p.details || '<p class="note">The handed-out search: no study of its own.</p>';
    document.getElementById('lines-note').textContent = p.lines_note;
    drawer.hidden = false;
    document.body.classList.add('drawer-open');
    // each figure gets its height before it is drawn, starts from an empty div, and is
    // sized again once the open drawer has its width
    ['trials', 'lines', 'curves'].forEach(function (k) {
      const f = p.figs[k], div = document.getElementById('fig-' + k);
      Plotly.purge(div);
      div.style.height = (f.layout.height || 320) + 'px';
      Plotly.newPlot(div, f.data, Object.assign({}, f.layout, {autosize: true, width: null}), CONFIG);
    });
    requestAnimationFrame(function () {
      ['trials', 'lines', 'curves'].forEach(function (k) {
        Plotly.Plots.resize(document.getElementById('fig-' + k));
      });
    });
    drawer.querySelector('.drawer-body').scrollTop = 0;
    drawer.focus({preventScroll: true});
    const card = document.getElementById('card-' + name);
    if (card) {
      card.classList.add('on');
      card.scrollIntoView({block: 'nearest', inline: 'nearest'});
    }
    document.querySelectorAll('svg.links path').forEach(function (l) {
      if (l.dataset.parent === name || l.dataset.child === name) l.classList.add('on');
    });
  }
  function fromHash() {
    let name = '';
    try { name = decodeURIComponent(location.hash.slice(1)); } catch (e) { name = ''; }
    show(name);
  }
  // views: one tab open at a time; its figures are sized once they can be seen
  // a figure that stands alone in a view fills the height left under it
  function resizeIn(el) {
    if (!window.Plotly) return;
    el.querySelectorAll('.js-plotly-plot').forEach(function (g) {
      if (g.offsetParent === null) return;
      const box = g.closest('.theme-evidence, .panel-body');
      const size = {width: Math.floor(g.parentElement.clientWidth)};
      if (box && !box.classList.contains('scroll')) {
        const h = Math.floor(box.getBoundingClientRect().bottom - g.getBoundingClientRect().top - 12);
        if (h > 240) size.height = h;
      }
      if (size.width > 0) Plotly.relayout(g, size);
    });
  }
  function openTab(id) {
    document.querySelectorAll('.tab').forEach(function (t) {
      const on = t.id === 'tab-' + id;
      t.setAttribute('aria-selected', String(on));
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    const panel = document.getElementById('panel-' + id);
    if (panel) { resizeIn(panel); if (id === 'built') drawLinks(); }
    try { localStorage.setItem('class-tab', id); } catch (e) {}
  }
  document.querySelectorAll('.tab').forEach(function (t) {
    t.addEventListener('click', function () { openTab(t.id.slice(4)); });
    t.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const all = Array.from(document.querySelectorAll('.tab'));
      const next = all[(all.indexOf(t) + (e.key === 'ArrowRight' ? 1 : all.length - 1)) % all.length];
      next.focus(); openTab(next.id.slice(4));
    });
  });
  document.querySelectorAll('.theme-tab').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('.theme-tab').forEach(function (x) {
        x.setAttribute('aria-selected', String(x === b));
      });
      document.querySelectorAll('.theme-panel').forEach(function (p) {
        p.hidden = p.dataset.theme !== b.dataset.theme;
        if (!p.hidden) resizeIn(p);
      });
    });
  });
    const lines = document.getElementById('lines-fig');
  function press(b, sel) {
    b.parentElement.querySelectorAll(sel).forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
  }
  // what the lines show: colour by score or by encoder, every encoder or one; a picked
  // encoder's lines alone are drawn
  const view = {colour: 'score', code: -1};
  let whole = null;
  function snapshot() {
    const d = lines.data[0], f = lines._fullData[0];
    whole = d.dimensions.map(function (dim, k) {
      const values = Array.from(f.dimensions[k].values);
      const real = values.filter(function (v) { return typeof v === 'number' && isFinite(v); });
      const range = dim.range || [Math.min.apply(null, real), Math.max.apply(null, real)];
      return Object.assign({}, dim, {values: values, range: range});
    });
  }
  function draw() {
    if (!lines || !lines.layout || !lines.layout.meta) return;
    if (!whole) snapshot();
    const m = lines.layout.meta, by = view.colour === 'encoder' ? m.by_encoder : m.by_score;
    const codes = Array.from(m.by_encoder.color);
    const keep = codes.map(function (c, k) { return view.code < 0 || c === view.code ? k : -1; })
                      .filter(function (k) { return k >= 0; });
    const dims = whole.map(function (dim) {
      const out = Object.assign({}, dim, {values: keep.map(function (r) { return dim.values[r]; })});
      delete out.constraintrange;
      return out;
    });
    const colours = Array.from(by.color);
    Plotly.restyle(lines, {dimensions: [dims], 'line.color': [keep.map(function (r) { return colours[r]; })],
                           'line.colorscale': [by.colorscale], 'line.cmin': by.cmin, 'line.cmax': by.cmax}, [0])
      .then(function () { resizeIn(lines.parentElement); });
  }
  document.querySelectorAll('.switch button[data-colour]').forEach(function (b) {
    b.addEventListener('click', function () {
      view.colour = b.dataset.colour; draw(); press(b, 'button[data-colour]');
    });
  });
  document.querySelectorAll('.switch button.enc').forEach(function (b) {
    b.addEventListener('click', function () {
      view.code = Number(b.dataset.code); draw(); press(b, 'button.enc');
    });
  });

  // a goal's picture: one student's own trials lit up, the rest of the class dimmed
  function pick(g, who) {
    if (!g || !g.data) return;
    const idx = [], vis = [], op = [];
    g.data.forEach(function (t, i) {
      const m = t.meta || {};
      if (m.student !== undefined) { idx.push(i); vis.push(m.student === who); op.push(1); }
      else if (m.base) { idx.push(i); vis.push(true); op.push(who ? 0.22 : 1); }
    });
    if (idx.length) Plotly.restyle(g, {visible: vis, opacity: op}, idx);
  }
  document.querySelectorAll('.theme-panel').forEach(function (p) {
    const g = p.querySelector('.theme-evidence .js-plotly-plot');
    p.querySelectorAll('button.pick').forEach(function (b) {
      b.addEventListener('click', function () {
        const on = b.getAttribute('aria-pressed') !== 'true';
        p.querySelectorAll('button.pick').forEach(function (x) {
          x.setAttribute('aria-pressed', String(on && x === b));
          x.parentElement.classList.toggle('on', on && x === b);
        });
        pick(g, on ? b.dataset.student : null);
      });
    });
  });
  // the capacity picture: what lies along x, and what the colour says
  document.querySelectorAll('.axis-switch').forEach(function (sw) {
    const g = document.getElementById(sw.dataset.fig);
    sw.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () {
        if (!g || !g.data || !g.layout.meta) return;
        const M = g.layout.meta, key = b.dataset.key, idx = [], vals = [], lay = {};
        press(b, 'button[data-kind="' + b.dataset.kind + '"]');
        if (b.dataset.kind === 'x') {
          g.data.forEach(function (t, i) { if (t.meta && t.meta.xs) { idx.push(i); vals.push(t.meta.xs[key]); } });
          Plotly.restyle(g, {x: vals}, idx);
          const a = M.x[key];
          Object.keys(g.layout).filter(function (k) { return /^xaxis\d*$/.test(k); }).forEach(function (k) {
            lay[k + '.type'] = a.type; lay[k + '.tickvals'] = a.tickvals; lay[k + '.ticktext'] = a.ticktext;
            lay[k + '.title.text'] = a.title; lay[k + '.autorange'] = true;
          });
        } else {
          g.data.forEach(function (t, i) { if (t.meta && t.meta.cs) { idx.push(i); vals.push(t.meta.cs[key]); } });
          Plotly.restyle(g, {'marker.color': vals}, idx);
          const c = M.colour[key];
          Object.assign(lay, {'coloraxis.cmin': c.cmin, 'coloraxis.cmax': c.cmax,
            'coloraxis.colorbar.tickvals': c.tickvals, 'coloraxis.colorbar.ticktext': c.ticktext,
            'coloraxis.colorbar.title.text': c.title});
        }
        Plotly.relayout(g, lay);
      });
    });
  });
  window.addEventListener('resize', function () {
    const open = document.querySelector('.tab-panel:not([hidden])');
    if (open) resizeIn(open);
  });

  document.addEventListener('keydown', function (e) {
    if (!drawer) return;
    if (e.key === 'Escape' && !drawer.hidden) {
      try { history.replaceState(null, '', location.pathname + location.search); } catch (err) {}
      show('');
    }
  });
  if (drawer) document.getElementById('drawer-close').addEventListener('click', function (e) {
    e.preventDefault();
    try { history.replaceState(null, '', location.pathname + location.search); } catch (err) {}
    show('');
  });
  // cost: a clicked trial beside the trials of about the same compute
  const costFig = document.getElementById('cost-fig');
  const same = document.getElementById('same-flops');
  function median(v) {
    const s = v.slice().sort(function (a, b) { return a - b; }), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function esc(t) {
    return String(t).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }
  function groupBy(items, k) {
    const out = {};
    items.forEach(function (q) { (out[q[k]] = out[q[k]] || []).push(q); });
    return Object.keys(out).map(function (key) {
      const sc = out[key].map(function (q) { return q[1]; });
      return {key: key, n: sc.length, med: median(sc), best: Math.max.apply(null, sc), items: out[key]};
    }).sort(function (a, b) { return b.med - a.med; });
  }
  function fl(v) {
    const u = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
    for (const [d, s] of u) if (v >= d) return (v / d).toPrecision(2).replace(/\.0$/, '') + ' ' + s;
    return String(v);
  }
  let costShapes = null;
  function showSame(i) {
    const M = costFig.layout.meta, P = M.points, me = P[i];
    if (!me) return;
    const lo = me[0] / M.near, hi = me[0] * M.near;
    const near = P.filter(function (q) { return q[0] >= lo && q[0] <= hi; });
    if (costShapes === null) costShapes = (costFig.layout.shapes || []).slice();
    Plotly.relayout(costFig, {shapes: costShapes.concat([{type: 'rect', xref: 'x', yref: 'paper', x0: lo, x1: hi,
      y0: 0, y1: 1, fillcolor: 'rgba(196,58,20,0.08)', line: {width: 0}, layer: 'below'}]),
      annotations: (costFig.layout.annotations || []).filter(function (a) { return !a.name; }).concat([{
        name: 'me', x: Math.log10(me[0]), y: me[1], xref: 'x', yref: 'y', text: '', showarrow: true,
        arrowhead: 0, arrowcolor: '#c43a14', arrowwidth: 2, ax: 0, ay: -28}])});
    const rowsOf = groupBy(near, 2).map(function (g) {
      const subs = groupBy(g.items, 5).slice(0, 6).map(function (t) {
        return '<tr class="sub"><th>' + esc(t.key) + '</th><td>' + t.n + '</td><td>' + t.med.toFixed(3) +
          '</td><td>' + t.best.toFixed(3) + '</td></tr>';
      }).join('');
      return '<tr class="enc' + (g.key === me[2] ? ' mine' : '') + '"><th>' + esc(g.key) + '</th><td>' + g.n +
        '</td><td>' + g.med.toFixed(3) + '</td><td>' + g.best.toFixed(3) + '</td></tr>' + subs;
    }).join('');
    const link = me[4] in D ? '<a href="#' + encodeURIComponent(me[4]) + '">' + esc(me[4]) + '</a>' : esc(me[4] || 'handed-out search');
    same.innerHTML = '<p class="me"><b>' + esc(me[3]) + '</b> · ' + link + '<br><span class="num">' +
      me[1].toFixed(3) + '</span> at <span class="num">' + fl(me[0]) + '</span> FLOPs · ' + esc(me[2]) +
      ' · ' + esc(me[6]) + '<br><span class="chip">' + esc(me[5]).split(', ').join('</span><span class="chip">') +
      '</span></p><h4>' + near.length + ' trials of about the same compute (' + fl(lo) + '–' + fl(hi) + ')</h4>' +
      '<table><thead><tr><th>encoder, then what the ideas investigated</th><th>n</th><th>median</th><th>best</th>' +
      '</tr></thead><tbody>' + rowsOf + '</tbody></table>';
  }
  if (costFig && costFig.on && same) {
    costFig.on('plotly_click', function (ev) {
      const p = ev.points && ev.points[0];
      if (p && typeof p.customdata === 'number') showSame(p.customdata);
    });
  }
  window.addEventListener('hashchange', fromHash);
  window.addEventListener('resize', drawLinks);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawLinks);
  let start = 'investigated';
  try { start = localStorage.getItem('class-tab') || start; } catch (e) {}
  if (!document.getElementById('tab-' + start)) start = 'investigated';
  openTab(start);
  drawLinks();
  fromHash();
})();
"""


def font_faces() -> str:
    """The course's faces, inside the page so it opens from disk as the course pages
    look; nothing when the font files are not there."""
    fonts = COURSE_ROOT / "presentations" / "fonts"
    out = []
    for family, weight, file in (("Outfit", 400, "Outfit-400.woff2"),
                                 ("Outfit", 500, "Outfit-500.woff2"),
                                 ("JetBrains Mono", 400, "JetBrainsMono-400.woff2")):  # fmt: skip
        path = fonts / file
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            out.append(f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
                       f"font-display:swap;src:url(data:font/woff2;base64,{data}) format('woff2')}}")  # fmt: skip
    return "".join(out)


CSS = """
:root{--paper:#f2f3ef;--ink:#111;--muted:#5a544d;--rule:#d6d9d1;--surface:#e8eae3;
--blue:#1d5a73;--blue-wash:#c9dde4;--orange:#c43a14;--orange-wash:#f3cdbf;--green:#1c6b48;
--sans:Outfit,'Segoe UI',Helvetica,Arial,sans-serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
color-scheme:light}
*{box-sizing:border-box}
html{scroll-padding-top:24px}
body{margin:0;background:var(--paper);color:var(--ink);font:400 17px/1.5 var(--sans);
-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
::selection{background:var(--blue-wash);color:var(--ink)}
a{color:var(--blue);text-underline-offset:3px}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:3px;border-radius:4px}
html,body{height:100%}
body{display:flex;flex-direction:column;overflow:hidden}
header.top{flex:none;padding:16px 28px 0;border-bottom:1px solid var(--rule)}
.title-row{display:flex;align-items:baseline;justify-content:space-between;gap:24px;flex-wrap:wrap}
h1{font-weight:500;font-size:30px;line-height:1.15;letter-spacing:-0.015em;margin:0;text-wrap:balance}
.yardstick{margin:0;font-size:16px;color:var(--muted);max-width:62ch}
.yardstick b{color:var(--ink);font-weight:500}
nav.tabs{display:flex;gap:4px;margin-top:10px;overflow-x:auto;scrollbar-width:thin}
.tab{font:500 16px/1.2 var(--sans);color:var(--muted);background:none;border:0;border-bottom:3px solid transparent;
padding:10px 14px 9px;cursor:pointer;white-space:nowrap;display:flex;gap:8px;align-items:baseline}
.tab .goal{font:400 13px var(--mono);color:var(--muted)}
.tab:hover{color:var(--ink)}
.tab[aria-selected=true]{color:var(--ink);border-bottom-color:var(--blue)}
main{flex:1;min-height:0;display:flex;flex-direction:column}
.tab-panel{flex:1;min-height:0;display:flex;flex-direction:column;padding:14px 28px 0}
.tab-panel[hidden]{display:none}
.panel-head{flex:none;display:flex;align-items:baseline;gap:24px;flex-wrap:wrap;margin-bottom:8px}
.caption{margin:0;font-size:17px;max-width:96ch}
.caption b{font-weight:500}
details.how{font-size:15px;color:var(--muted)}
details.how[open]{flex-basis:100%}
details.how p{max-width:96ch;margin:6px 0 0}
.panel-body{flex:1;min-height:0;position:relative}
.panel-body.scroll{overflow:auto;padding-bottom:24px}
footer{flex:none;border-top:1px solid var(--rule);padding:6px 28px;font-size:14px;color:var(--muted)}
footer summary{color:var(--muted)}
footer .source{margin:6px 0 4px}

/* what the class investigated: a rail of goals, the evidence, the ideas */
.themes{display:grid;grid-template-columns:250px minmax(0,1fr) minmax(320px,30%);gap:0 24px;height:100%}
.theme-rail{display:flex;flex-direction:column;gap:2px;overflow-y:auto;border-right:1px solid var(--rule);padding-right:12px}
.theme-tab{all:unset;cursor:pointer;display:grid;grid-template-columns:42px 1fr;gap:2px 8px;padding:12px 10px;border-radius:6px}
.theme-tab .goal{grid-row:span 2;font-size:15px;color:var(--blue);padding-top:2px}
.theme-tab .ttl{font-size:16px;line-height:1.3}
.theme-tab .cnt{font-size:14px;color:var(--muted)}
.theme-tab:hover{background:var(--surface)}
.theme-tab[aria-selected=true]{background:var(--blue-wash)}
.theme-tab:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.theme-panel{display:contents}
.theme-panel[hidden]{display:none}
.theme-evidence{min-width:0;min-height:0}
.theme-ideas{overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding:0 4px 24px 0}
.tstudent{border:1px solid var(--rule);border-radius:8px;background:#fbfbf8;transition:border-color .15s,box-shadow .15s}
.tstudent.on{border-color:var(--orange);box-shadow:inset 4px 0 0 var(--orange)}
.tstudent .pick{all:unset;box-sizing:border-box;cursor:pointer;display:flex;justify-content:space-between;align-items:baseline;
gap:10px;width:100%;padding:9px 12px 4px}
.tstudent .pick b{font-weight:500;font-size:16px}
.tstudent .pick span{font-size:14px;color:var(--muted)}
.tstudent .pick:hover b{color:var(--orange)}
.tstudent .pick:focus-visible{outline:2px solid var(--blue);outline-offset:-2px;border-radius:8px}
.tstudent ul{list-style:none;margin:0;padding:0 6px 6px}
.tstudent li a{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 10px;align-items:baseline;padding:5px 6px;
border-radius:6px;color:var(--ink);text-decoration:none}
.tstudent li a:hover{background:var(--surface)}
.tstudent li .idea-name{font-size:13px}
.tstudent li .num{grid-row:span 2;font-size:15px}
.tstudent li .chips{grid-column:1}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{display:inline-block;font-size:13px;line-height:1.5;border-radius:4px;padding:0 7px;background:var(--blue-wash);color:#0c3445}
.switch{display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-size:15px;color:var(--muted);margin:0 0 4px}
.switch button{font:15px var(--sans);border:1px solid var(--rule);background:#fbfbf8;color:var(--ink);border-radius:6px;
padding:3px 10px;cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.switch button[aria-pressed=true]{background:var(--blue-wash);border-color:var(--blue)}
.switch button i{width:14px;height:4px;border-radius:2px;display:inline-block}
.switch .gap{width:18px}
.theme-evidence{display:flex;flex-direction:column}
.theme-evidence .js-plotly-plot{flex:none}
h2{font-weight:500;font-size:28px;line-height:1.2;letter-spacing:-0.01em;margin:0 0 8px;text-wrap:balance}
h3{font-weight:500;font-size:17px;margin:28px 0 6px}
.lede{font-size:17px;line-height:1.4;margin:4px 0 0;color:var(--muted)}
.lede b{font-weight:500}
.source{color:var(--muted);font-size:15px;max-width:80ch;margin:0}
.source code,.lead code,.note code{font-family:var(--mono);font-size:.86em;color:var(--blue)}
.lead{color:var(--muted);max-width:76ch;margin:0 0 20px}
.lead b{color:var(--ink);font-weight:500}
.missing{color:var(--muted);background:var(--surface);border-radius:6px;padding:12px 16px;margin:8px 0}
.num,.idea-name,.pm{font-family:var(--mono)}
.key{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:15px;color:var(--muted);margin:0 0 16px;padding:0;list-style:none}
.key li{display:flex;align-items:center;gap:8px}
.key svg{flex:none}

/* lanes: one row per student, one column per day */
.lanes-wrap{padding:4px 0 12px}
.lanes{position:relative;display:grid;grid-template-columns:max-content minmax(0,1fr);column-gap:32px}
.lanes svg.links{position:absolute;inset:0;pointer-events:none;overflow:visible;z-index:2}
.lane-label{padding:10px 0;border-bottom:1px solid var(--rule);display:flex;flex-direction:column;gap:2px}
.lane-label b{font-weight:500;font-size:16px;white-space:nowrap}
.lane-label span{color:var(--muted);font-size:14px}
.cell{padding:8px 0;border-bottom:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:8px 40px}
.cell a.idea{width:250px}
a.idea{position:relative;z-index:3;display:flex;flex-direction:column;gap:4px;background:#fbfbf8;
border:1px solid var(--rule);border-radius:8px;padding:7px 10px;color:var(--ink);text-decoration:none;
box-shadow:0 1px 2px rgba(17,17,17,.05);transition:border-color .15s,box-shadow .15s}
a.idea:hover{border-color:var(--blue);box-shadow:0 4px 14px rgba(17,17,17,.08)}
a.idea.on{border-color:var(--blue);box-shadow:0 0 0 2px var(--blue)}
.idea-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.idea-name{font-size:13px;color:var(--muted)}
.idea-time{font-size:13px;color:var(--muted);white-space:nowrap}
.res{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:0 8px;align-items:baseline}
.res b.num{font-size:16px;font-weight:500;color:var(--ink)}
.q{font-size:16px;line-height:1.35}
.tag{align-self:flex-start;font-size:12px;line-height:1.5;border-radius:4px;padding:1px 7px;background:var(--surface);color:var(--muted)}
.tag.cross{background:var(--orange-wash);color:#7a2410;font-weight:500}
ul.found{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px;font-size:15px;line-height:1.35;color:var(--muted)}
ul.found li.final{color:var(--ink);font-size:15px}
ul.found li.final b{font-weight:500;font-size:20px}
ul.found .pm{font-size:14px;color:var(--muted)}
ul.found li.diff{color:var(--ink)}
ul.found li.diff .num{font-weight:500}
svg.links path{fill:none;stroke-linecap:round}
svg.links path.own{stroke:#9a958d;stroke-width:1.6}
svg.links path.cross{stroke:var(--orange);stroke-width:3}
svg.links path.on{stroke-width:3.4}
svg.links path.own.on{stroke:var(--blue)}
.head-own{fill:#9a958d}.head-cross{fill:var(--orange)}

.drawer dl.words{display:grid;grid-template-columns:84px minmax(0,1fr);gap:6px 14px;margin:0}
.drawer dl.words dt{color:var(--muted);font-size:14px;padding-top:2px}
.drawer dl.words dd{margin:0;max-width:68ch}

.figure-note{color:var(--muted);font-size:14px;margin:4px 0 0}
.table-wrap{overflow-x:auto}
table.effort{border-collapse:collapse;font-size:16px;min-width:640px}
table.effort th,table.effort td{padding:9px 18px 9px 0;border-bottom:1px solid var(--rule);text-align:right}
table.effort thead th{font-weight:500;color:var(--muted);font-size:14px;vertical-align:bottom}
table.effort th[scope=row],table.effort thead th:first-child{text-align:left;font-weight:500}
table.effort td{font-family:var(--mono);font-size:15px}
table.effort tr.teacher th,table.effort tr.teacher td{color:var(--muted);font-weight:400;border-top:1px dashed var(--rule);padding-top:14px}
.cost-wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,26%);gap:0 24px;height:100%}
.cost-plot{min-width:0;min-height:0}
.same{overflow-y:auto;font-size:15px;padding:0 4px 24px 0}
.same .me{margin:0 0 10px;line-height:1.6}
.same .me b{font-weight:500}
.same h4{font-weight:500;font-size:15px;margin:12px 0 6px}
.same table{border-collapse:collapse;width:100%;font-size:14px}
.same th,.same td{text-align:right;padding:4px 0 4px 10px;border-bottom:1px solid var(--rule);font-weight:400}
.same thead th{color:var(--muted);font-size:13px;vertical-align:bottom}
.same th:first-child{text-align:left;padding-left:0}
.same td{font-family:var(--mono)}
.same tr.enc th{font-weight:500;padding-top:10px}
.same tr.mine th{color:var(--orange)}
.same tr.sub th{padding-left:14px;color:var(--muted)}
p,li,dd,.caption{text-wrap:pretty}

/* the drawer: one idea */
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(640px,100vw);background:var(--paper);
border-left:1px solid var(--rule);box-shadow:-12px 0 32px rgba(17,17,17,.12);z-index:10;display:flex;flex-direction:column}
.drawer[hidden]{display:none}
#fig-trials,#fig-lines,#fig-curves{position:relative;width:100%;overflow:hidden}
.drawer-bar{display:flex;justify-content:flex-end;padding:12px 20px 0}
.close{font-size:15px;color:var(--ink);border:1px solid var(--rule);border-radius:6px;padding:4px 12px;text-decoration:none;background:#fbfbf8}
.close:hover{border-color:var(--ink)}
.drawer-body{overflow-y:auto;padding:4px 28px 40px}
.drawer h2{font-family:var(--mono);font-size:22px;font-weight:400;margin:0}
.drawer .meta{color:var(--muted);margin:2px 0 16px;font-size:15px}
.drawer .note{color:var(--muted);font-size:14px;margin:0}
dl.kv{display:grid;grid-template-columns:84px minmax(0,1fr);gap:8px 14px;font-size:15px;margin:0}
dl.kv dt{color:var(--muted);font-size:14px}
dl.kv dd{margin:0;min-width:0}
dl.kv code{font-family:var(--mono);font-size:13px;overflow-wrap:anywhere;color:var(--blue)}
.chips span{display:inline-block;background:var(--surface);border-radius:4px;padding:1px 7px;margin:2px 4px 2px 0;font:13px var(--mono)}
.chips .on{background:var(--blue-wash);color:#0c3445}
.chips .off{color:var(--muted);text-decoration:line-through}
details pre{background:var(--surface);border-radius:6px;padding:10px 12px;font:12px/1.5 var(--mono);overflow:auto;max-height:360px}
summary{cursor:pointer;color:var(--blue)}
@media (max-width:1100px){
 .cost-wrap{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}.same{max-height:36vh}
 .themes{grid-template-columns:200px minmax(0,1fr);grid-template-rows:minmax(0,1fr) auto}
 .theme-ideas{grid-column:1/-1;max-height:40vh}
}
@media (max-width:760px){
 body{overflow:auto;display:block}html,body{height:auto}
 header.top,.tab-panel,footer{padding-left:16px;padding-right:16px}h1{font-size:24px}
 .tab-panel{min-height:80vh}
 .themes{display:flex;flex-direction:column}.theme-rail{flex-direction:row;overflow-x:auto;border:0}
 .theme-evidence{height:460px}
 .lanes{grid-template-columns:1fr}.lane-label{border-bottom:0;padding-bottom:0}
 .drawer-body{padding:4px 16px 32px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def key_line() -> str:
    """What the links between cards mean."""
    own = ("<svg width='34' height='10' aria-hidden='true'><path d='M1 5H31' stroke='#9a958d' stroke-width='1.6'/>"
           "<path d='M27 1L33 5L27 9z' fill='#9a958d'/></svg>")  # fmt: skip
    cross = ("<svg width='34' height='10' aria-hidden='true'><path d='M1 5H31' stroke='#c43a14' stroke-width='3'/>"
             "<path d='M27 1L33 5L27 9z' fill='#c43a14'/></svg>")  # fmt: skip
    return (
        "<ul class='key'>"
        f"<li>{own}a student builds on their own idea</li>"
        f"<li>{cross}a student builds on a classmate’s idea</li></ul>"
    )


def class_summary(df: pd.DataFrame, ids: pd.DataFrame, eds: pd.DataFrame) -> str:
    """The page's opening sentence: students, ideas, days, building on a classmate,
    and trials."""
    own = ids[ids.author != COURSE]
    by = {r.name: r for r in rows(ids)}
    on_classmate = sum(
        1 for e in rows(eds)
        if e.parent in by and by[e.parent].author != COURSE and not is_teacher(by[e.parent].author)
        and lane_of(by[e.parent]) != lane_of(by[e.child])
    )  # fmt: skip
    days = {w.date() for w in (moment(r.start) for r in rows(own)) if w is not None}
    trials = df[df.role == "trial"]
    n_students = len(students(df))
    text = (
        f"<b>{plural(n_students, 'student')}</b> handed in <b>{plural(len(own), 'idea')}</b>"
        + (f" over {plural(len(days), 'day')}" if days else "")
        + f"; <b>{on_classmate}</b> {'builds' if on_classmate == 1 else 'build'} on a classmate’s idea."
        f" They ran {plural(len(trials), 'trial')}, {int((trials.status == 'COMPLETE').sum())} to the end."
    )
    return f"<p class='lede'>{text}</p>"


def plotly_script(how: str) -> str:
    """The page's plotly.js: inside it (`inline`), or (`cdn`) loaded from cdnjs in the
    exact version plotly.py bundles, checked against the bundled file's hash, which keeps
    the page small enough to commit."""
    js = get_plotlyjs()
    if how == "inline":
        return f"<script>{js}</script>"
    if how != "cdn":
        raise ValueError(f"plotly is inline or cdn, not {how!r}")
    digest = base64.b64encode(hashlib.sha384(js.encode()).digest()).decode()
    url = f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{get_plotlyjs_version()}/plotly.min.js"
    return (
        f'<script src="{url}" integrity="sha384-{digest}" '
        'crossorigin="anonymous"></script>'
    )


def source_line(source: str, submitted: int | None, read: int) -> str:
    """Where the runs came from and how many the page reads, one number: the submitted
    runs, and the earlier executions of a handed-out notebook among them that the page
    leaves out."""
    if submitted is None:
        return source
    text = f"{plural(submitted, 'run')} submitted" + (f" to {source}" if source else "")
    left = submitted - read
    if left > 0:
        text += (
            f". {left} of them are earlier executions of the handed-out notebook by "
            f"someone who ran it again; only the latest one counts, so the page reads "
            f"{plural(read, 'run')}"
        )
    return text + "."


COURSE_LINK_CSS = (
    ".course-link { margin: 0 0 14px; font-size: 14px; }"
    f" .course-link a {{ color: {MUTED}; text-decoration: none; }}"
    " .course-link a:hover { text-decoration: underline; }"
)


def course_link(course: str) -> str:
    """A link back to the course's start page at `course`, for a class page published
    among the course pages; nothing when `course` is empty."""
    if not course:
        return ""
    return (
        f"<style>{COURSE_LINK_CSS}</style><p class='course-link'>"
        f"<a href='{html.escape(course, quote=True)}'>← MADS-ML start</a></p>"
    )


def build_page(
    df: pd.DataFrame,
    lesson: int,
    source: str = "",
    metric: str = METRIC,
    others: Iterable[str] = (),
    elsewhere: Mapping[str, int] | None = None,
    teacher: str | None = None,
    submitted: int | None = None,
    plotly: str = "inline",
    course: str = "",
) -> str:
    """The page of `lesson` from `df` (`runs_frame`). `source` names where the runs came
    from, `submitted` how many submission files `df` was read from: `runs_frame` keeps
    every one but a handed-out notebook's earlier executions, so the difference is
    named as those. The `teacher`'s runs are drawn as the teacher's (`as_teacher`).
    `plotly` is `inline` (plotly.js inside the page, which then opens from disk with no
    network) or `cdn` (`plotly_script`). `course`, when given, is the address of the
    course's start page, linked above the title (`course_link`)."""
    elsewhere = elsewhere or {}
    df = as_teacher(df, teacher)
    cmp = comparisons(df)
    noise = class_noise(df)
    ids = ideas(df, cmp)
    eds = edges(ids, noise)
    runs = reruns(df)
    pooled = all_trials(df)
    tags = {r.name: idea_tags(df, r) for r in rows(ids)}

    other = [m for m in others if m != metric and "/" not in m]
    per_class: dict[str, int] = {}
    for m in others:
        head, _, _ = m.partition("/")
        if "/" in m:
            per_class[head] = per_class.get(head, 0) + 1
    other += [
        f"{head} per class ({n} classes, as {head}/<class>)"
        for head, n in per_class.items()
    ]
    reads = (
        f" The runs also log {esc(', '.join(other))}; read one with "
        f"<code>make dashboard LESSON={lesson} METRIC=&lt;name&gt;</code>."
        if other
        else ""
    )
    tabs: list[tuple[str, str, str, str]] = []  # (id, label, goal, html)
    tabs.append((
        "investigated", "What they investigated", "",
        panel("The class’s evidence per learning goal. Click a student to light up the runs of "
              "their ideas on that goal, a final as a star; click an idea to open it. The dotted line is the handed-out "
              "baseline, its band the spread of its reruns.",
              themes_view(df, ids, noise, pooled, lesson, tags, metric)),
    ))  # fmt: skip
    tabs.append((
        "built", "Built on", "1.15",
        panel("One lane per student, their ideas in the order they ran, tagged with what they "
              "investigated; an orange link builds on a classmate’s idea. Click a card to open the idea.",
              key_line() + lanes(ids, eds, noise, elsewhere, metric, tags), scroll=True),
    ))  # fmt: skip
    if noise is not None:
        noise_body = figure(fig_reruns(runs, noise, metric))
        noise_how = (
            f"Everyone ran the handed-out baseline with the same config: <b>{noise.n} runs</b> of one "
            f"model. Each averages its own seeds on its own machine, and what is left between them is "
            f"noise, a spread of <b class='num'>{100 * noise.spread:.2f} pt</b> (one standard deviation). "
            f"The band is the median ± that spread. A bar is one run’s spread over its seeds, smaller "
            f"than the spread between runs, so a difference on the page carries the spread between runs."
        )
    else:
        noise_body = missing(
            "Fewer than three runs of the handed-out baseline were submitted."
        )
        noise_how = ""
    tabs.append((
        "noise", "Run again", "1.10",
        panel("The handed-out baseline, run again by everyone: how far one model’s score moves.",
              noise_body, how=noise_how),
    ))  # fmt: skip
    lines, below = fig_lines(pooled, AXES, noise, metric)
    searched = (
        f"Every finished trial above the handed-out baseline, <b>{len(pooled) - below}</b> of {len(pooled)}"
        + (f"; the {below} under it are left out" if below and noise else "")
        + ". Pick an encoder to draw only its trials. The darker a line, the higher its score."
    )
    tabs.append((
        "searched", "Search space", "1.14",
        panel(searched, (colour_switch(lines.layout.meta) if len(pooled) and noise is not None else "")
              + (figure(lines, div_id="lines-fig") if len(pooled) else missing("No finished trial submitted.")),
              how="One line is one trial. Drag along an axis to keep a range; drag again elsewhere to "
                  "clear it."),
    ))  # fmt: skip
    inter = fig_interactions(pooled, noise, metric) if noise is not None else None
    tabs.append((
        "interactions", "Two choices", "",
        panel("Each grid crosses two choices of the head; a cell is the median of its trials in points "
              "above the baseline, a bold number its row’s best."
              + (f" No grid for {esc('; '.join(left))}." if (left := facets_left_out(pooled)) else ""),
              figure(inter) if inter is not None else missing(
                  "Not enough finished trials over hidden layers, width and dropout for a grid."),
              how=f"Grey: under the baseline, or fewer than {MIN_CELL} trials. One row per encoder, "
                  "because a larger encoder lifts every cell and would hide the head; each row has its own "
                  "colour scale, from its weakest coloured cell to its best. The trials come from "
                  "different studies with their own learning rates and epochs, so read a cell as what the "
                  "class happened to find there.", scroll=True),
    ))  # fmt: skip
    if noise is not None:
        cost, below_c = fig_cost(pooled, noise, metric, tags)
        tabs.append((
            "cost", "Cost", "1.4",
            panel("Score against the compute of training the head, the same on every machine; the line "
                  "is the cheapest way the class found to each score. Click a point to set it beside the "
                  "trials of about the same compute."
                  + (f" {plural(below_c, 'trial')} under the baseline left out." if below_c else ""),
                  "<div class='cost-wrap'><div class='cost-plot'>" + figure(cost, div_id="cost-fig")
                  + "</div><aside class='same' id='same-flops' aria-live='polite'><p class='note'>Click a "
                  "point: the trials within ×1.6 of its compute stand here, by encoder and by what their "
                  "ideas investigated.</p></aside></div>",
                  how="Compute is about 6 FLOPs per parameter per training row per epoch, from the "
                      "parameters and epochs every trial logs. The encoder ran once, when the cache was "
                      "made, and is not in it."),
        ))  # fmt: skip
    tabs.append((
        "students", "Effort", "",
        panel("Ideas and trials of their own studies, the runs of the handed-out notebook, and the "
              "minutes their machine spent training.", effort_table(df, ids), scroll=True),
    ))  # fmt: skip
    drawer = ""
    if (ids.author != COURSE).any():
        drawer = (
            DRAWER
            + f"<script type='application/json' id='drawer-data'>{drawer_data(df, ids, eds, noise, metric)}</script>"
        )
    bar = "".join(
        f"<button type='button' role='tab' class='tab' id='tab-{t}' aria-controls='panel-{t}' "
        f"aria-selected='{str(i == 0).lower()}'>{esc(label)}"
        f"{f'<span class=goal>{esc(goal)}</span>' if goal else ''}</button>"
        for i, (t, label, goal, _) in enumerate(tabs)
    )
    panels = "".join(
        f"<section role='tabpanel' class='tab-panel' id='panel-{t}' aria-labelledby='tab-{t}'"
        f"{'' if i == 0 else ' hidden'}>{body}</section>"
        for i, (t, _, _, body) in enumerate(tabs)
    )
    yardstick = (
        f"<p class='yardstick'>The handed-out baseline scores <span class='num'>{noise.center:.3f}</span> "
        f"and moves <b class='num'>± {100 * noise.spread:.1f} pt</b> when run again "
        f"(the standard deviation over {noise.n} runs).</p>"
        if noise is not None
        else "<p class='yardstick'>Too few runs of the handed-out baseline to say how far a score moves.</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Lesson {lesson}: what the class did</title>"
        f"<style>{font_faces()}{CSS}</style>{plotly_script(plotly)}</head><body>"
        f"<header class='top'>{course_link(course)}<div class='title-row'>"
        f"<h1>Lesson {lesson}: what the class did</h1>{yardstick}</div>"
        f"{class_summary(df, ids, eds)}"
        f"<nav class='tabs' role='tablist' aria-label='Views'>{bar}</nav></header>"
        f"<main>{panels}</main>"
        f"<footer><details><summary>Where these runs come from</summary>"
        f"<p class='source'>{esc(source_line(source, submitted, len(df)))} Every picture reads "
        f"<b>{esc(metric)}</b>.{reads}</p></details></footer>"
        f"{drawer}<script>{SCRIPT}</script></body></html>"
    )


def panel(caption: str, body: str, how: str = "", scroll: bool = False) -> str:
    """A view of the page: one sentence on what it shows, the view, and the longer
    reading behind a disclosure."""
    more = (
        f"<details class='how'><summary>How to read this</summary><p>{how}</p></details>"
        if how
        else ""
    )
    return (
        f"<div class='panel-head'><p class='caption'>{caption}</p>{more}</div>"
        f"<div class='panel-body{' scroll' if scroll else ''}'>{body}</div>"
    )


def colour_switch(meta: Mapping[str, object] | None = None) -> str:
    """The switch between colouring the lines by score and by encoder, and, when the
    lines have an encoder axis, buttons that keep one encoder's lines and fade the
    rest."""
    keep = ""
    listed = (meta or {}).get("encoders")
    encoders = list(listed) if isinstance(listed, list) else []
    if (meta or {}).get("encoder_axis") is not None and len(encoders) > 1:
        keep = (
            "<span class='gap'></span><span>keep</span>"
            "<button type='button' class='enc' data-code='-1' aria-pressed='true'>every encoder</button>"
            + "".join(
                f"<button type='button' class='enc' data-code='{i}' aria-pressed='false'>"
                f"<i style='background:{c}'></i>{esc(e)}</button>"
                for i, (e, c) in enumerate(encoders)
            )
        )
    return (
        "<div class='switch' role='group' aria-label='Colour and keep the lines'>"
        "<span>colour by</span>"
        "<button type='button' data-colour='score' aria-pressed='true'>score</button>"
        "<button type='button' data-colour='encoder' aria-pressed='false'>encoder</button>"
        f"{keep}</div>"
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


GITHUB = re.compile(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$")


def class_repository(repo: Path) -> tuple[str, str] | None:
    """(owner, name) of the class repository a clone reads from: its `upstream` remote,
    else its `origin`, as `collect_runs.class_ref` picks them. None when neither is a
    GitHub URL."""
    from collect_runs import git

    for remote in ("upstream", "origin"):
        got = git(repo, "config", "--get", f"remote.{remote}.url")
        if got.returncode == 0:
            m = GITHUB.search(got.stdout.decode().strip())
            return (m.group(1), m.group(2)) if m else None
    return None


def read_class(
    repo: Path, lesson: int, fetch: bool
) -> tuple[list[Submission], str, dict[str, int]]:
    """The valid submissions of `lesson` on the class repository's main, where they came
    from (`owner/name at main`), and the ideas of the other lessons
    (`ideas_elsewhere`); invalid files are reported and skipped."""
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
    slug = class_repository(repo)
    branch = ref.rsplit("/", 1)[-1]
    where = f"{slug[0]}/{slug[1]}" if slug else repo.name
    return subs, f"{where} at {branch}", ideas_elsewhere(others, lesson)


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
    parser.add_argument(
        "--plotly",
        choices=["inline", "cdn"],
        default="inline",
        help="plotly.js inside the page (opens from disk) or from cdnjs (a small page)",
    )
    parser.add_argument(
        "--teacher",
        help="the teacher's login (default: the owner of the class repository)",
    )
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
    parser.add_argument(
        "--source",
        help="what the page says its runs were submitted to, in place of the class "
        "repository's name: for a page of a rehearsal",
    )
    parser.add_argument(
        "--course-link",
        default="",
        metavar="URL",
        help="link the course's start page at URL above the title: for a class page "
        "published among the course pages",
    )
    args = parser.parse_args()

    if args.demo:
        from class_demo import DEMO_TEACHER, demo_submissions

        subs = demo_submissions(args.lesson)
        source, elsewhere, teacher = "made-up data of a class", {}, DEMO_TEACHER
    else:
        repo = args.runs_repo.resolve()
        subs, source, elsewhere = read_class(repo, args.lesson, fetch=not args.no_pull)
        owner = class_repository(repo)
        teacher = owner[0] if owner else None
    teacher = args.teacher or teacher
    source = args.source or source
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
    out.write_text(
        build_page(
            df,
            args.lesson,
            source,
            metric,
            known,
            elsewhere,
            teacher,
            len(subs),
            args.plotly,
            args.course_link,
        ),
        encoding="utf-8",
    )
    who = people(as_teacher(df, teacher).student)
    trials, studies = int((df.role == "trial").sum()), int((df.origin == "study").sum())
    print(
        f"{len(df)} of {len(subs)} runs, of {who}: {trials} trials, {studies} runs of own studies"
    )
    print(f"page: {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
