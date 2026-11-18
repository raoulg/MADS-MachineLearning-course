"""A student's page: one page that shows what one student studied in a lesson.

    uv run scripts/study_page.py --lesson 1                      # from your own store
    uv run scripts/study_page.py --lesson 1 --open               # ... and open it
    uv run scripts/study_page.py --lesson 1 --submitted raoulg   # from a login's submitted runs
    make page LESSON=1                                           # the first, opened

The page is the class view (`scripts/class_analysis.py`) read for one student. By
default it reads the student's own store (`hpo.record.Store.default()`): every study of
the lesson, the search and its `-runs` companion, and the handed-out exercise's runs when
they are there. With `--submitted <login>` it reads that login's runs of the lesson on
the class runs repository's `main`, as `class_analysis.py` reads them, so a page can be
built from published runs alone.

It writes one self-contained page, `artefacts/study-page/lesson<N>/index.html`, and
prints its path. The page has three kinds of tab:

- **Overview**: the student's message and picture, the line of their ideas in the order
  they build on each other, three panels for one study (its architecture, its search
  space, its winner next to its baseline), and what surprised and what comes next.
- **Design**: the page atlas draws of the lesson's `exercises/` folder.
- **one per study**: its idea and expectation, every searched axis against the objective
  and against `train_seconds`, the learning curves, and the winner's other metrics.

What is the student's own comes from `notebooks/<lesson>/exercises/page.toml`, or the
file `--page` names: the message, the picture, what surprised and what next. A key left
empty shows as an open slot that asks its question. Once runs of the lesson are found,
the script creates the default file with empty keys when it is missing, and never writes
it otherwise. The file is not tracked: a student commits theirs on a branch of their own.
A page of submitted runs reads no page.toml unless `--page` names one; its own slots say
they are not part of the submitted runs.

Every score is read on its own study's objective and its own dataset: scores of two
studies, or of two datasets, are never put on one axis.
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import webbrowser
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_analysis import (
    OFF,
    ON,
    budget_text,
    chip_value,
    idea_depths,
    number,
    page_metric,
    plural,
    read_class,
    runs_frame,
    sigma,
    signed,
    study_of,
    verdict,
)
from class_analysis import tick as short
from run_submission import (
    COURSE,
    LOGIN,
    Submission,
    explain,
    idea_parents,
)

from hpo.record import Store

COURSE_ROOT = Path(__file__).resolve().parents[1]
PRESENTATIONS = COURSE_ROOT / "presentations"
ATLAS_COMMANDS = ("atlas-code-nightly", "atlas-code")
ATLAS_SETUP = "docs/setup/tooling-setup.md, milestone M7 (Atlas)"

# What `page.toml` holds, and the question an empty key asks on the page.
OWN = {
    "message": "Say in three sentences what you did.",
    "picture": "Draw your idea in a picture of your own: name an .svg, .png or .html "
    "fragment in page.toml under picture.",
    "surprised": "What surprised you, against what you expected?",
    "next": "What would you try next, and why?",
}
MAX_SENTENCES = 3
PAGE_TOML = """\
# Your page of lesson {lesson}: what is yours on it. `make page LESSON={lesson}` reads this
# file and never writes it once it exists. An empty key shows as an open slot on the page.

# Say in at most three sentences what you did: what you investigated, what you found,
# and what it means.
message = ""

# A picture of your own, shown under the message: a path, relative to this folder, to an
# .svg, a .png or an .html fragment. The beelddenker skill helps you find one.
picture = ""

# What surprised you, against what you expected (EXPECTS in study.py)?
surprised = ""

# What would you try next, and why?
next = ""
"""


# ---- what the student wrote: page.toml -------------------------------------------------


def exercises_folder(lesson: int, root: Path = COURSE_ROOT) -> Path:
    """`notebooks/<lesson>_*/exercises` of the course checkout; a `SystemExit` when the
    lesson has none."""
    found = sorted((root / "notebooks").glob(f"{lesson}_*/exercises"))
    if not found:
        sys.exit(f"lesson {lesson} has no notebooks/{lesson}_*/exercises folder")
    return found[0]


def shown(path: Path) -> str:
    """`path` relative to the root of the course repository when it lies inside it,
    absolute otherwise."""
    path = path.resolve()
    return str(
        path.relative_to(COURSE_ROOT) if path.is_relative_to(COURSE_ROOT) else path
    )


def read_own(path: Path, lesson: int, create: bool = False) -> dict[str, str]:
    """The keys of the page.toml at `path` as text, every key of `OWN` present (empty
    when the file leaves it out). With `create`, a missing file is made with empty keys
    and a comment per key first; without, a missing file stops the script."""
    if not path.exists():
        if not create:
            sys.exit(f"no page.toml at {path}")
        path.write_text(PAGE_TOML.format(lesson=lesson))
        print(f"wrote {shown(path)}: fill it in, and build the page again")
    try:
        held = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{path} is not valid TOML: {e}")
    return {key: str(held.get(key) or "").strip() for key in OWN}


def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def open_slot(key: str) -> str:
    """The place of a key the student has not filled in: its question."""
    return (
        f"<div class='slot open'><span class='tag'>page.toml · {key}</span>"
        f"<p>{html.escape(OWN[key])}</p></div>"
    )


def filled(key: str, text: str) -> str:
    body = "".join(
        f"<p>{html.escape(part)}</p>" for part in text.split("\n\n") if part.strip()
    )
    note = ""
    if key == "message" and sentences(text) > MAX_SENTENCES:
        note = (
            f"<p class='warn'>{sentences(text)} sentences: say it in "
            f"{MAX_SENTENCES}.</p>"
        )
    return f"<div class='slot filled' data-key='{key}'>{body}{note}</div>"


NOT_SUBMITTED = "Not part of the submitted runs."


def absent(key: str) -> str:
    """The place of a key on a page built from submitted runs, which carry no page.toml."""
    return (
        f"<div class='slot absent'><span class='tag'>page.toml · {key}</span>"
        f"<p>{NOT_SUBMITTED}</p></div>"
    )


def own_text(own: Mapping[str, str] | None, key: str) -> str:
    if own is None:
        return absent(key)
    return filled(key, own[key]) if own.get(key) else open_slot(key)


def picture(own: Mapping[str, str] | None, folder: Path) -> str:
    """The student's picture inside the page: an svg or an html fragment as it is, a png
    as data; an open slot without one, and a line saying so when the file is not there or
    lies outside `folder` (an absolute path, or one that climbs out with `..`)."""
    if own is None:
        return absent("picture")
    name = own.get("picture", "")
    if not name:
        return open_slot("picture")
    path = (folder / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(folder.resolve()):
        return (
            "<div class='slot open'><span class='tag'>page.toml · picture</span>"
            f"<p>{html.escape(name)} lies outside the folder of page.toml: name a file "
            "inside it.</p></div>"
        )
    if not path.is_file():
        return (
            "<div class='slot open'><span class='tag'>page.toml · picture</span>"
            f"<p>No file {html.escape(name)} next to page.toml.</p></div>"
        )
    suffix = path.suffix.lower()
    if suffix == ".png":
        data = base64.b64encode(path.read_bytes()).decode()
        body = (
            f"<img alt='the picture of this study' src='data:image/png;base64,{data}'>"
        )
    elif suffix in (".svg", ".html", ".htm"):
        body = path.read_text()
    else:
        return (
            "<div class='slot open'><span class='tag'>page.toml · picture</span>"
            f"<p>{html.escape(name)} is not an .svg, .png or .html file.</p></div>"
        )
    return f"<figure class='slot filled picture'>{body}</figure>"


# ---- the runs: the student's store, or what a login submitted --------------------------


def store_login(store: Store, attrs: Iterable[Mapping[str, object]]) -> str:
    """The student whose store it is: the login the store keeps, else the author of an
    idea of their own; empty when neither says. It asks no one and writes nothing."""
    kept = store.root / "login"
    try:
        held = kept.read_text().strip() if kept.is_file() else ""
    except OSError:
        held = ""
    if LOGIN.fullmatch(held):
        return held
    for a in attrs:
        author = a.get("idea_author")
        if isinstance(author, str) and author != COURSE and LOGIN.fullmatch(author):
            return author
    return ""


# The student of a submission made of a store that names no login: only the course's
# handed-out ideas are in it, and they pass with any submitter.
NO_LOGIN = "student"


def read_store(store: Store, lesson: int) -> tuple[list[Submission], str, str]:
    """The runs of `lesson` in `store`, as the submissions `make share` would make of
    them, the login they are of (empty when the store does not say), and where they came
    from."""
    import optuna
    from optuna.artifacts import FileSystemArtifactStore
    from submit_runs import Store as Db
    from submit_runs import default_experiments, own_runs, select_runs, to_submission

    where = store.db
    source = f"your store {shown(where)}"
    if not where.exists():
        return [], "", source
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = Db(where)
    db.artifacts = FileSystemArtifactStore(str(store.artifacts))
    import fnmatch

    patterns = default_experiments(lesson)
    if not any(fnmatch.fnmatchcase(n, p) for n in db.study_names() for p in patterns):
        return [], "", source
    found = own_runs(select_runs(db, lesson, None, None))
    login = store_login(store, [f.study_attrs for f in found])
    code = {"commit": None, "dirty": None, "remote": None}
    now = datetime.now(UTC)
    subs = []
    for f in found:
        try:
            subs.append(to_submission(db, f, lesson, login or NO_LOGIN, code, now))
        except ValueError as e:
            print(f"left out {f.study} #{f.trial.number}: " + "; ".join(explain(e)))
    return subs, login, source


def read_submitted(
    repo: Path, lesson: int, login: str, fetch: bool
) -> tuple[list[Submission], str, str]:
    """The runs of `lesson` that `login` submitted to the class runs repository."""
    subs, where, _ = read_class(repo, lesson, fetch)
    mine = [s for s in subs if s.student.lower() == login.lower()]
    name = mine[0].student if mine else login
    return mine, name, f"the runs {name} submitted to {where}"


# ---- one study: a search and its runs, read on its own objective -----------------------


@dataclass
class Axis:
    """One axis of a search space as `Space.line` writes it."""

    name: str
    kind: str  # "fixed", "choice" or "range"
    value: str = ""
    choices: list[str] = field(default_factory=list)
    low: float = math.nan
    high: float = math.nan
    log: bool = False


SEGMENT = re.compile(
    r"(?P<name>[\w.-]+)\s*(?:=\s*(?P<value>.*)"
    r"|in\s*\{(?P<choices>.*)\}"
    r"|~\s*(?P<kind>\w+)\(\s*(?P<low>[^,]+),\s*(?P<high>[^)]+)\))"
)


def parse_space(line: str | None) -> list[Axis]:
    """The axes of a search space line, in its order; a segment it cannot read is left
    out."""
    out = []
    for part in (line or "").split("|"):
        m = SEGMENT.fullmatch(part.strip())
        if m is None:
            continue
        if m["value"] is not None:
            out.append(Axis(m["name"], "fixed", value=m["value"].strip()))
        elif m["choices"] is not None:
            out.append(
                Axis(
                    m["name"],
                    "choice",
                    choices=[c.strip() for c in m["choices"].split(",")],
                )
            )
        else:
            out.append(
                Axis(
                    m["name"], "range", low=number(m["low"]), high=number(m["high"]),
                    log=m["kind"].startswith("log"),
                )
            )  # fmt: skip
    return out


@dataclass
class Study:
    """A study of the page: its search (`trials`) and its runs (`base`, `final`), all
    read on its own objective."""

    key: str
    label: str
    handed_out: bool
    objective: str
    df: pd.DataFrame
    subs: dict[str, Submission]
    idea: str = ""
    idea_name: str = ""
    idea_author: str = ""
    parents: list[str] = field(default_factory=list)
    expects: str = ""
    budget: dict | None = None
    space: str | None = None
    axes: list[str] = field(default_factory=list)
    start: float = 0.0
    slug: str = ""

    @property
    def trials(self) -> pd.DataFrame:
        """The trials of the search in the order they ran, `winner` True for the best
        finished one, as the search picks it."""
        return self.trials_on(None)

    def trials_on(self, dataset: str | None) -> pd.DataFrame:
        """The trials on `dataset` (all with None), `winner` True for the best finished
        one among them: a score is compared only with scores of its own dataset."""
        sel = self.df[self.df.role == "trial"]
        if dataset is not None:
            sel = sel[sel.dataset == dataset]
        sel = sel.sort_values("start").reset_index(drop=True)
        done = sel[(sel.status == "COMPLETE") & sel.score.notna()]
        best = done.score.idxmax() if len(done) else None
        return sel.assign(winner=sel.index == best)

    def datasets(self) -> list[str]:
        """Every dataset the study's runs are on, the winner's first."""
        first = self.dataset
        return sorted({str(d) for d in self.df.dataset}, key=lambda d: (d != first, d))

    def on_text(self) -> str:
        """The dataset its scores are on, in words: the winner's, and the others the
        search drew."""
        others = self.datasets()[1:]
        if not others:
            return self.dataset
        return f"{self.dataset}, the winner's dataset (the search also drew {', '.join(others)})"

    def latest(self, role: str) -> pd.Series | None:
        sel = self.df[(self.df.role == role) & self.df.score.notna()]
        return None if sel.empty else sel.sort_values("start").iloc[-1]

    @property
    def base(self) -> pd.Series | None:
        """The baseline on the winner's dataset; with no winner rerun yet, the latest
        baseline."""
        f = self.final
        if f is None:
            return self.latest("baseline")
        sel = self.df[
            (self.df.role == "baseline")
            & self.df.score.notna()
            & (self.df.dataset == f.dataset)
        ]
        return None if sel.empty else sel.sort_values("start").iloc[-1]

    @property
    def final(self) -> pd.Series | None:
        return self.latest("final")

    @property
    def winner(self) -> pd.Series | None:
        t = self.trials
        won = t[t.winner]
        return None if won.empty else won.iloc[0]

    @property
    def finished(self) -> int:
        return int(((self.df.role == "trial") & (self.df.status == "COMPLETE")).sum())

    @property
    def pruned(self) -> int:
        return int(((self.df.role == "trial") & (self.df.status == "PRUNED")).sum())

    @property
    def dataset(self) -> str:
        for row in (self.final, self.winner, self.base):
            if row is not None and row.dataset != "unknown":
                return str(row.dataset)
        return "unknown"

    def gain(self) -> tuple[float, float, str]:
        """The winner against its own baseline: points, sigmas and `verdict`'s word; NaN
        and `unmeasured` without both."""
        b, f = self.base, self.final
        if b is None or f is None:
            return math.nan, math.nan, "unmeasured"
        sig = sigma(f, b)
        return 100 * (f.score - b.score), sig, verdict(sig)

    def params(self) -> dict[str, object]:
        """The winner's config: the final's params, else those of the best trial on the
        winner's dataset."""
        on = self.trials_on(self.dataset if self.final is not None else None)
        best = on[on.winner]
        for row in (self.final, None if best.empty else best.iloc[0]):
            if row is None:
                continue
            out = {
                str(k)[2:]: v
                for k, v in row.items()
                if str(k).startswith("p_") and v is not None
                and not (isinstance(v, float) and math.isnan(v))
            }  # fmt: skip
            if out:
                return out
        return {}

    def artifact(self, name: str) -> object | None:
        """An artifact of the final, else of the winner, else of any run of the study."""
        rows = [r for r in (self.final, self.winner) if r is not None]
        refs = [str(r.ref) for r in rows] + list(self.df.ref)
        for ref in refs:
            sub = self.subs.get(ref)
            if sub is not None and name in sub.run.artifacts:
                return sub.run.artifacts[name]
        return None

    def design_script(self) -> str | None:
        script = self.artifact("design.py")
        return script if isinstance(script, str) and script else None


def studies_of(subs: Sequence[Submission]) -> list[Study]:
    """The studies of `subs`, a search and its `-runs` companion as one, in the order of
    their ideas (`idea_depths`: an idea after every idea it builds on), then the order
    they ran."""
    groups: dict[str, list[Submission]] = {}
    for sub in subs:
        groups.setdefault(sub.run.experiment.removesuffix("-runs"), []).append(sub)
    out = []
    for key, group in groups.items():
        objective = page_metric(group)
        df = runs_frame(group, objective)
        if df.empty:
            continue
        attrs: dict[str, object] = {}
        for sub in sorted(group, key=lambda s: not s.run.experiment.endswith("-runs")):
            attrs.update(sub.run.study.attrs)  # the search's attrs win
        handed_out = attrs.get("origin") == "notebook" or bool(
            re.fullmatch(r"lesson\d+-notebook", key)
        )
        label = "handed out" if handed_out else (study_of(key) or key)
        axes = attrs.get("searched_axes")
        budget = attrs.get("budget")
        out.append(
            Study(
                key=key,
                label=label,
                handed_out=handed_out,
                objective=objective,
                df=df,
                subs={s.ref(): s for s in group},
                idea=str(attrs.get("idea") or ""),
                idea_name=str(attrs.get("idea_name") or ""),
                idea_author=str(attrs.get("idea_author") or ""),
                parents=idea_parents(attrs.get("idea_parents") or ""),
                expects=str(attrs.get("expects") or ""),
                budget=budget if isinstance(budget, dict) else None,
                space=str(attrs["search_space"]) if attrs.get("search_space") else None,
                axes=str(axes).split(",") if isinstance(axes, str) and axes else [],
                start=float(pd.to_numeric(df.start, errors="coerce").min()),
            )
        )
    named = {s.idea_name: s.parents for s in out if s.idea_name}
    depth = (
        idea_depths(
            pd.DataFrame({"name": list(named), "parents": list(named.values())})
        )
        if named
        else {}
    )
    out.sort(key=lambda s: (depth.get(s.idea_name, 0), s.start, s.key))
    taken: set[str] = {"overview", "design"}
    for s in out:  # a tab's id, unique on the page
        base = "study-" + re.sub(r"[^a-z0-9]+", "-", s.label.lower()).strip("-")
        s.slug, n = base, 2
        while s.slug in taken:
            s.slug, n = f"{base}-{n}", n + 1
        taken.add(s.slug)
    return out


# ---- pictures: inline svg, coloured by the course's tokens -----------------------------


def svg(width: int, height: int, body: str, label: str) -> str:
    """A picture that fills its box up to 1.6 times its own size, so its text stays
    legible on a wide screen and on a phone."""
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(label)}'"
        f" style='max-width:{round(width * 1.6)}px'"
        f" xmlns='http://www.w3.org/2000/svg'>{body}</svg>"
    )


def tick(value: float) -> str:
    """A number at the end of an axis: a whole number as it is, else `class_analysis`'s
    short form."""
    if math.isfinite(value) and value == round(value) and abs(value) < 1e6:
        return str(round(value))
    return short(value)


def text(x: float, y: float, s: object, cls: str = "t", anchor: str = "start") -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' class='{cls}' text-anchor='{anchor}'>"
        f"{html.escape(str(s))}</text>"
    )


def fraction(value: float, low: float, high: float, log: bool) -> float:
    """Where `value` lies between `low` and `high`, 0..1, on a log scale with `log`."""
    if log and low > 0 and high > 0 and value > 0:
        value, low, high = math.log10(value), math.log10(low), math.log10(high)
    if not high > low:
        return 0.5
    return min(1.0, max(0.0, (value - low) / (high - low)))


def fig_space(study: Study) -> str:
    """One bar per axis of the study's space: a range as a track with its ends, a choice
    as its options side by side, a fixed value as itself; the winner's value marked."""
    axes = parse_space(study.space)
    if not axes:
        return missing("No search space was logged for this study.")
    won = study.params()
    width, row = 320, 34
    left, right = 104, 312
    body = []
    for i, a in enumerate(axes):
        y = 8 + i * row
        searched = a.kind != "fixed" and not (a.kind == "choice" and len(a.choices) < 2)
        body.append(text(0, y + 13, a.name, "t b" if searched else "t m"))
        body.append(text(0, y + 25, "searched" if searched else "fixed", "s m"))
        value = won.get(a.name)
        if a.kind == "range":
            body.append(
                f"<line x1='{left}' x2='{right}' y1='{y + 12}' y2='{y + 12}' class='track'/>"
            )
            body.append(text(left, y + 27, tick(a.low), "s m"))
            body.append(
                text(
                    right,
                    y + 27,
                    tick(a.high) + (" log" if a.log else ""),
                    "s m",
                    "end",
                )
            )
            v = number(value)
            if not math.isnan(v):
                x = left + fraction(v, a.low, a.high, a.log) * (right - left)
                body.append(f"<circle cx='{x:.1f}' cy='{y + 12}' r='5.5' class='win'/>")
                body.append(text(x, y + 4, chip_value(value), "s b", "middle"))
        elif a.kind == "choice":
            n = max(len(a.choices), 1)
            w = (right - left) / n
            for j, c in enumerate(a.choices):
                x = left + j * w
                on = str(value).lower() == c.lower() if value is not None else False
                body.append(
                    f"<rect x='{x + 1:.1f}' y='{y + 2}' width='{w - 2:.1f}' height='20' "
                    f"rx='3' class='{'seg on' if on else 'seg'}'/>"
                )
                short = c if len(c) * 6 < w - 4 else c[: max(1, int((w - 8) / 6))] + "…"
                body.append(
                    text(x + w / 2, y + 16, short, "s on" if on else "s", "middle")
                )
        else:
            body.append(
                f"<rect x='{left}' y='{y + 2}' width='{right - left}' height='20' rx='3' class='fixed'/>"
            )
            body.append(text(left + 8, y + 16, f"= {a.value}", "s m"))
    height = 8 + len(axes) * row
    note = "" if won else missing("No winner yet: no value is marked.")
    return (
        svg(width, height, "".join(body), f"the search space of {study.label}")
        + "<p class='foot'>● the winner's value · bold: searched</p>"
        + note
    )


def spread_text(score: float, std: object, n: object) -> str:
    s = number(std)
    runs = number(n)
    out = f"{score:.4f}"
    if not math.isnan(s):
        out += f" ± {s:.4f}"
    if not math.isnan(runs):
        out += f" · {plural(int(runs), 'run')}"
    return out


def gain_words(
    points: float, sig: float, word: str, base_n: object, final_n: object
) -> str:
    """The gain in points and in sigmas, and what it means, in words."""
    if math.isnan(points):
        return "no winner against a baseline yet"
    if math.isnan(sig):
        single = 1 in (number(base_n), number(final_n))
        return f"{signed(points)} pt · no sigma: " + (
            "a side has a single run" if single else "no spread to measure it by"
        )
    meaning = {
        "better": "more than two sigma above its baseline",
        "worse": "more than two sigma below its baseline",
        "inside": "inside two sigma: no measurable difference from its baseline",
    }[word]
    return f"{signed(points)} pt · {sig:.1f}σ · {meaning}"


def fig_results(study: Study) -> str:
    """The baseline and the winner on the study's objective, each ± one standard
    deviation between its runs."""
    b, f = study.base, study.final
    rows = [(n, r) for n, r in (("baseline", b), ("winner", f)) if r is not None]
    if not rows:
        return missing(
            "No finished trial yet." if study.finished == 0 else
            "No baseline or winner rerun was logged."
        )  # fmt: skip
    lows = [
        r.score - (0 if math.isnan(number(r["std"])) else r["std"]) for _, r in rows
    ]
    highs = [
        r.score + (0 if math.isnan(number(r["std"])) else r["std"]) for _, r in rows
    ]
    low, high = min(lows), max(highs)
    pad = max((high - low) * 0.25, 0.002)
    low, high = low - pad, high + pad
    width, left, right = 320, 70, 308
    body = []
    for i, (name, r) in enumerate(rows):
        y = 22 + i * 40
        x = left + fraction(r.score, low, high, False) * (right - left)
        s = number(r["std"])
        if not math.isnan(s):
            x0 = left + fraction(r.score - s, low, high, False) * (right - left)
            x1 = left + fraction(r.score + s, low, high, False) * (right - left)
            body.append(
                f"<line x1='{x0:.1f}' x2='{x1:.1f}' y1='{y}' y2='{y}' class='err'/>"
            )
        body.append(
            f"<circle cx='{x:.1f}' cy='{y}' r='6' class='{'win' if name == 'winner' else 'base'}'/>"
        )
        body.append(text(0, y + 4, name, "t b"))
        body.append(text(x, y - 10, f"{r.score:.4f}", "s", "middle"))
    y = 22 + len(rows) * 40 - 14
    body.append(f"<line x1='{left}' x2='{right}' y1='{y}' y2='{y}' class='track'/>")
    body.append(text(left, y + 14, tick(low), "s m"))
    body.append(text(right, y + 14, tick(high), "s m", "end"))
    body.append(text((left + right) / 2, y + 14, study.objective, "s m", "middle"))
    points, sig, word = study.gain()
    gain = gain_words(
        points, sig, word, None if b is None else b["n"], None if f is None else f["n"]
    )
    lines = "".join(
        f"<li><b>{n}</b> {html.escape(spread_text(r.score, r['std'], r['n']))}</li>"
        for n, r in rows
    )
    return (
        svg(width, y + 22, "".join(body), f"winner and baseline of {study.label}")
        + f"<p class='gain {word}'>{html.escape(gain)}</p>"
        + f"<ul class='plain'>{lines}</ul>"
        + f"<p class='foot'>{html.escape(study.objective)} on {html.escape(study.on_text())}"
        " · ± one standard deviation between runs</p>"
    )


@dataclass
class Stage:
    name: str
    shape: str
    option: str | None = None


def shape_of(node: ast.expr | None) -> str | None:
    """The shape string of an annotation `Float[Tensor, "batch {self.dim}"]`."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
        last = node.slice.elts[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str):
            return last.value
    return None


def option_of(fn: ast.FunctionDef) -> str | None:
    for d in fn.decorator_list:
        if (
            isinstance(d, ast.Call)
            and getattr(d.func, "id", None) == "option"
            and d.args
            and isinstance(d.args[0], ast.Constant)
        ):
            return str(d.args[0].value)
    return None


def design_stages(script: str) -> tuple[str, list[Stage]] | None:
    """The input shape and the stages of the design in `script`: the typed methods
    `forward` calls, in the order it calls them, each with the shape it gives and the
    axis of its `@option`. The class whose `forward` calls the most stages is the
    design. None when the script holds no typed stage."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return None
    best: tuple[str, list[Stage]] | None = None
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        forward = methods.get("forward")
        if forward is None:
            continue
        typed = {
            n: f
            for n, f in methods.items()
            if n not in ("forward", "__init__") and shape_of(f.returns)
        }
        calls = sorted(
            (
                (c.lineno, c.col_offset, c.func.attr)
                for c in ast.walk(forward)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "self"
                and c.func.attr in typed
            ),
        )
        order = list(dict.fromkeys(name for _, _, name in calls))
        stages = [
            Stage(n, shape_of(typed[n].returns) or "", option_of(typed[n]))
            for n in order
        ]
        args = forward.args.args[1:]
        given = shape_of(args[0].annotation) if args else None
        if stages and (best is None or len(stages) > len(best[1])):
            best = (given or "", stages)
    return best


def resolved(shape: str, values: Mapping[str, object]) -> str:
    """`shape` with every `{self.<name>}` that `values` knows filled in."""
    return re.sub(
        r"\{self\.(\w+)\}",
        lambda m: chip_value(values.get(m.group(1), m.group(1))),
        shape,
    )


def architecture_panel(study: Study) -> str:
    """The stages of the design with the shape after each, its options marked on or off
    in the winner; from the logged layers when no design script was logged."""
    arch = study.artifact("architecture.json")
    arch = arch if isinstance(arch, dict) else {}
    params = study.params()
    first = arch.get("input_shape") or []
    last = arch.get("output_shape") or []
    values: dict[str, object] = dict(params)
    if first and isinstance(first[-1], int):
        values.setdefault("dim", first[-1])
    if last and isinstance(last[-1], int):
        values.setdefault("n_classes", last[-1])
    n_params = arch.get("n_params")
    count = (
        f"<p class='foot'>{int(n_params):,} parameters in the winner</p>"
        if isinstance(n_params, int)
        else ""
    )
    script = study.design_script()
    parsed = design_stages(script) if script else None
    if parsed is not None:
        given, stages = parsed
        items = [
            f"<li class='in'><b>in</b><code>{html.escape(resolved(given, values))}</code></li>"
        ]
        for s in stages:
            state = ""
            cls = "stage"
            if s.option is not None:
                cls += " optional"
                v = params.get(s.option)
                state = (
                    " · on in the winner" if v in ON
                    else " · off in the winner" if v in OFF else ""
                )  # fmt: skip
                cls += " off" if v in OFF else ""
            opt = (
                f"<span class='opt'>@option(&quot;{html.escape(s.option)}&quot;){state}</span>"
                if s.option
                else ""
            )
            items.append(
                f"<li class='{cls}'><b>{html.escape(s.name)}</b>"
                f"<code>{html.escape(resolved(s.shape, values))}</code>{opt}</li>"
            )
        return (
            f"<ol class='stages'>{''.join(items)}</ol>"
            "<p class='foot'>from the design script · dashed: a stage the search may "
            "leave out</p>" + count
        )
    layers = [
        layer
        for layer in arch.get("layers", [])
        if isinstance(layer, dict) and layer.get("leaf")
    ]
    if layers:
        items = [
            f"<li class='stage'><b>{html.escape(str(layer.get('type')))}</b>"
            f"<code>{html.escape(' × '.join(map(str, (layer.get('output_shape') or [])[1:])))}</code></li>"
            for layer in layers
        ]
        return (
            missing(
                "No design script was logged: these are the winner's layers as "
                "logged, without the design's stages or its options."
            )
            + f"<ol class='stages'>{''.join(items)}</ol>"
            + count
        )
    return missing("No architecture was logged for this study.")


def scatter(
    xs: Sequence[object],
    ys: Sequence[float],
    winner: Sequence[bool],
    axis: Axis | None,
    x_label: str,
    y_label: str,
) -> str:
    """One dot per trial, `ys` against `xs`, the winner filled: a range on its own scale
    (log when the space says so), a choice or anything else that is not a number as its
    values side by side."""
    width, height, left, right, top, bottom = 300, 160, 46, 292, 10, 124
    pts = [
        (x, y, w)
        for x, y, w in zip(xs, ys, winner, strict=True)
        if not math.isnan(number(y))
    ]
    if not pts:
        return missing(f"No finished trial logged {y_label}.")
    numeric = (
        axis.kind == "range"
        if axis is not None and axis.kind != "fixed"
        else all(
            not isinstance(x, bool | np.bool_ | str) and not math.isnan(number(x))
            for x, _, _ in pts
        )
    )
    if numeric:
        values = [number(x) for x, _, _ in pts]
        low, high = (
            (axis.low, axis.high)
            if axis and axis.kind == "range"
            else (min(values), max(values))
        )
        log = bool(axis.log) if axis and axis.kind == "range" else (
            min(values) > 0 and max(values) / max(min(values), 1e-12) > 20
        )  # fmt: skip

        def at(x: object) -> float:
            return left + fraction(number(x), low, high, log) * (right - left)

        ticks = [(left, tick(low)), (right, tick(high) + (" log" if log else ""))]
    else:
        cats = (
            list(axis.choices)
            if axis and axis.kind == "choice"
            else sorted({str(x).lower() for x, _, _ in pts})
        )
        cats_lower = [c.lower() for c in cats]
        step = (right - left) / max(len(cats), 1)

        def at(x: object) -> float:
            key = str(x).lower()
            i = cats_lower.index(key) if key in cats_lower else 0
            return left + step * (i + 0.5)

        ticks = [(left + step * (i + 0.5), c) for i, c in enumerate(cats)]
    yv = [float(y) for _, y, _ in pts]
    ylow, yhigh = min(yv), max(yv)
    pad = (yhigh - ylow) * 0.08 or max(abs(yhigh) * 0.01, 1e-3)
    ylow, yhigh = ylow - pad, yhigh + pad
    if min(yv) >= 0:  # seconds and scores do not go below zero
        ylow = max(ylow, 0.0)

    def ay(y: float) -> float:
        return bottom - fraction(y, ylow, yhigh, False) * (bottom - top)

    body = [
        f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' class='track'/>",
        f"<line x1='{left}' x2='{left}' y1='{top}' y2='{bottom}' class='track'/>",
        text(left - 4, top + 8, tick(yhigh), "s m", "end"),
        text(left - 4, bottom, tick(ylow), "s m", "end"),
    ]
    jitter = 0 if numeric else 6
    for i, (x, y, w) in enumerate(sorted(pts, key=lambda p: p[2])):
        dx = 0 if not jitter else ((i * 37) % 13 - 6) / 6 * jitter
        body.append(
            f"<circle cx='{at(x) + dx:.1f}' cy='{ay(float(y)):.1f}' r='{5 if w else 3.2}' "
            f"class='{'win' if w else 'dot'}'/>"
        )
    for x, label in ticks:
        anchor = "middle" if not numeric else ("start" if x == left else "end")
        body.append(text(x, bottom + 14, label, "s m", anchor))
    body.append(text((left + right) / 2, bottom + 30, x_label, "s", "middle"))
    return f"<p class='ylab'>{html.escape(y_label)}</p>" + svg(
        width, height, "".join(body), f"{y_label} against {x_label}"
    )


def fig_curves(
    study: Study, trials: pd.DataFrame, b: pd.Series | None, dataset: str | None
) -> str:
    """The objective after every epoch of `trials` (one dataset's): every trial faint,
    the best one bold, and the baseline `b` of that dataset as its own curve when it
    logged one, else as its score ± one standard deviation. `dataset` names the panel
    when the study drew more than one."""
    has = trials[trials.curve.map(lambda c: isinstance(c, list) and len(c) > 0)]
    base_curve = (
        b.curve
        if b is not None and isinstance(b.curve, list) and len(b.curve)
        else None
    )
    head = f"<p class='ds'>on {html.escape(dataset)}</p>" if dataset else ""
    if has.empty and base_curve is None:
        return head + missing("No run of this study logged its learning curve.")
    curves = list(has.curve) + ([base_curve] if base_curve else [])
    ys = [v for c in curves for v in c]
    if b is not None:
        s = 0.0 if math.isnan(number(b["std"])) else float(b["std"])
        ys += [b.score - s, b.score + s]
    low, high = min(ys), max(ys)
    pad = (high - low) * 0.06 or 0.01
    low, high = low - pad, high + pad
    epochs = max(len(c) for c in curves)
    width, height, left, right, top, bottom = 320, 190, 46, 312, 10, 160

    def ax(i: int) -> float:
        return left + (i / max(epochs - 1, 1)) * (right - left)

    def ay(y: float) -> float:
        return bottom - fraction(y, low, high, False) * (bottom - top)

    def path(c: Sequence[float]) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{ax(i):.1f},{ay(v):.1f}" for i, v in enumerate(c)
        )

    body = [
        f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' class='track'/>",
        f"<line x1='{left}' x2='{left}' y1='{top}' y2='{bottom}' class='track'/>",
        text(left - 4, top + 8, tick(high), "s m", "end"),
        text(left - 4, bottom, tick(low), "s m", "end"),
        text(left, bottom + 14, "1", "s m"),
        text(right, bottom + 14, str(epochs), "s m", "end"),
        text((left + right) / 2, bottom + 26, "epoch", "s", "middle"),
    ]
    note = []
    if b is not None and base_curve is None:
        s = 0.0 if math.isnan(number(b["std"])) else float(b["std"])
        body.append(
            f"<rect x='{left}' y='{ay(b.score + s):.1f}' width='{right - left}' "
            f"height='{max(ay(b.score - s) - ay(b.score + s), 1):.1f}' class='band'/>"
        )
        body.append(
            f"<line x1='{left}' x2='{right}' y1='{ay(b.score):.1f}' y2='{ay(b.score):.1f}' class='baseline'/>"
        )
        note.append(
            "the baseline logs no curve: its band is its score ± one standard deviation"
        )
    elif b is None:
        note.append(
            "no baseline was logged on this dataset"
            if dataset
            else "no baseline was logged"
        )
    for r in has.itertuples():
        if not r.winner:
            body.append(f"<path d='{path(r.curve)}' class='faint'/>")
    if base_curve:
        body.append(f"<path d='{path(base_curve)}' class='basecurve'/>")
    won = has[has.winner]
    for c in won.curve:
        body.append(f"<path d='{path(c)}' class='wincurve'/>")
    if won.empty:
        note.append("the winning trial logged no curve")
    return (
        head
        + f"<p class='ylab'>{html.escape(study.objective)} per epoch</p>"
        + svg(width, height, "".join(body), f"learning curves of {study.label}")
        + "<p class='foot'>bold: the winning trial of the search, not its rerun · "
        "faint: the other trials · "
        "dotted: the baseline" + (" · " + " · ".join(note) if note else "") + "</p>"
    )


def metrics_table(study: Study) -> str:
    """Every metric the winner's rerun logged, with its spread between runs."""
    f = study.final
    if f is None:
        return missing(
            "The winner has not been rerun yet: no metrics of it are logged."
        )
    sub = study.subs.get(str(f.ref))
    attrs = sub.run.attrs if sub is not None else {}
    logged = attrs.get("metrics")
    spread = (
        attrs.get("metrics_std") if isinstance(attrs.get("metrics_std"), dict) else {}
    )
    if not isinstance(logged, dict):
        return missing("The winner logs no metrics beyond its objective.")
    scalars = []
    per_class: dict[str, dict[str, float]] = {}
    for name, v in logged.items():
        if isinstance(v, dict):
            for cls, value in v.items():
                per_class.setdefault(str(cls), {})[name] = value
        elif isinstance(v, int | float):
            s = spread.get(name) if isinstance(spread, dict) else None
            mark = " (objective)" if name == study.objective else ""
            scalars.append(
                f"<tr><th>{html.escape(name)}{mark}</th><td>{v:.4f}</td>"
                f"<td>{'' if not isinstance(s, int | float) else f'± {s:.4f}'}</td></tr>"
            )
    for name, key in (("train_seconds", "train_seconds"), ("parameters", "n_params")):
        v = attrs.get(key)
        if isinstance(v, int | float):
            shown = f"{v:,.0f}" if key == "n_params" else f"{v:.2f}"
            scalars.append(f"<tr><th>{name}</th><td>{shown}</td><td></td></tr>")
    out = f"<table class='metrics'><tbody>{''.join(scalars)}</tbody></table>"
    if per_class:
        heads = sorted({m for v in per_class.values() for m in v})
        rows = "".join(
            f"<tr><th>{html.escape(cls)}</th>"
            + "".join(
                f"<td>{v[m]:.3f}</td>"
                if isinstance(v.get(m), int | float)
                else "<td></td>"
                for m in heads
            )
            + "</tr>"
            for cls, v in sorted(
                per_class.items(), key=lambda kv: kv[1].get("recall", 0)
            )
        )
        out += (
            f"<details><summary>per class, {len(per_class)} classes, the weakest recall "
            "first</summary><table class='metrics per-class'><thead><tr><th>class</th>"
            + "".join(f"<th>{html.escape(m)}</th>" for m in heads)
            + f"</tr></thead><tbody>{rows}</tbody></table></details>"
        )
    return out


def missing(text_: str) -> str:
    return f"<p class='missing'>{html.escape(text_)}</p>"


# ---- the page --------------------------------------------------------------------------


def idea_box(study: Study, known: Mapping[str, Study]) -> str:
    """One box of the line of ideas: the idea in the student's words, what they
    expected, and the gain over its own baseline."""
    points, sig, word = study.gain()
    b, f = study.base, study.final
    if study.finished == 0 and f is None:
        result = "<p class='gain unmeasured'>No finished trial yet.</p>"
    else:
        result = (
            f"<p class='gain {word}'>"
            + html.escape(
                gain_words(
                    points,
                    sig,
                    word,
                    None if b is None else b["n"],
                    None if f is None else f["n"],
                )
            )
            + "</p>"
        )
    words = study.idea or "the study carries no idea"
    expects = (
        f"<p class='expects'><b>expected</b> {html.escape(study.expects)}</p>"
        if study.expects
        else (
            "<p class='expects open'><b>expected</b> not said: the handed-out study "
            "states no expectation</p>"
            if study.handed_out
            else "<p class='expects open'><b>expected</b> not said: EXPECTS in study.py</p>"
        )
    )
    parents = []
    for p in study.parents:
        other = known.get(p)
        parents.append(
            f"{html.escape(p)} ({html.escape(other.label)})" if other else
            f"{html.escape(p)} (not on this page)"
        )  # fmt: skip
    builds = f"<p class='builds'>builds on {', '.join(parents)}</p>" if parents else ""
    return (
        f"<li><button type='button' class='idea' data-study='{study.slug}' "
        f"aria-pressed='false'><span class='label'>{html.escape(study.label)}</span>"
        f"<span class='name'>{html.escape(study.idea_name or study.key)}</span>"
        f"<span class='words'>{html.escape(words)}</span>{expects}{result}{builds}"
        f"<span class='meta'>{html.escape(study.objective)} on {html.escape(study.on_text())}"
        f" · {html.escape(budget_text(study.budget, study.finished + study.pruned))}</span>"
        "</button></li>"
    )


def panels(study: Study, shown: bool) -> str:
    """The three panels of one study on the overview."""
    return (
        f"<div class='three' data-panels='{study.slug}'{'' if shown else ' hidden'}>"
        f"<div class='panel'><h3>Architecture</h3>{architecture_panel(study)}</div>"
        f"<div class='panel'><h3>Search space</h3>{fig_space(study)}</div>"
        f"<div class='panel'><h3>Results</h3>{fig_results(study)}</div></div>"
    )


def overview(
    studies: Sequence[Study], own: Mapping[str, str] | None, folder: Path
) -> str:
    known = {s.idea_name: s for s in studies if s.idea_name}
    if studies:
        line = (
            "<ol class='ideas'>"
            + "".join(idea_box(s, known) for s in studies)
            + "</ol>"
        )
        current = studies[-1]
        three = "".join(panels(s, s is current) for s in studies)
        caption = (
            f"<p class='cap' id='panels-cap'>Showing <b>{html.escape(current.label)}</b>. "
            "Click a box above to show another study.</p>"
        )
    else:
        line, three, caption = (
            missing("No study of this lesson is in the runs."),
            "",
            "",
        )
    return (
        "<section class='own'><h2>What I did</h2>"
        f"{own_text(own, 'message')}{picture(own, folder)}</section>"
        "<section><h2>The line of my ideas</h2><p class='cap'>One box per study, each "
        "after the studies it builds on. The gain is the winner against the study's own "
        "baseline, in points and in sigmas (standard errors of the difference).</p>"
        f"{line}</section>"
        f"<section><h2>One study, in three panels</h2>{caption}{three}</section>"
        "<section class='own'><h2>What surprised me</h2>"
        f"{own_text(own, 'surprised')}<h2>What I would try next</h2>"
        f"{own_text(own, 'next')}</section>"
    )


def study_tab(study: Study) -> str:
    """A study's own tab. When the search drew more than one dataset, every picture is
    drawn once per dataset, and no picture compares scores of two datasets."""
    trials = study.trials
    datasets = study.datasets()
    several = len(datasets) > 1
    space = {a.name: a for a in parse_space(study.space)}
    axes = study.axes or [a for a, v in space.items() if v.kind != "fixed"]
    if several:
        axes = [a for a in axes if a != "dataset"]
    on = {d: study.trials_on(d) for d in datasets}
    done = {d: t[(t.status == "COMPLETE") & t.score.notna()] for d, t in on.items()}
    cards = []
    for axis in axes:
        col = f"p_{axis}"
        parts = []
        for d in datasets:
            head = f"<p class='ds'>on {html.escape(d)}</p>" if several else ""
            got = done[d]
            if col not in got or got.empty:
                parts.append(head + missing("No finished trial carries this axis."))
                continue
            xs, win = list(got[col]), list(got.winner)
            parts.append(
                head
                + scatter(xs, list(got.score), win, space.get(axis), axis, study.objective)
                + scatter(
                    xs, [number(v) for v in got.train_seconds], win, space.get(axis),
                    axis, "train_seconds",
                )
            )  # fmt: skip
        cards.append(
            f"<div class='card'><h4>{html.escape(axis)}</h4>{''.join(parts)}</div>"
        )
    finished = any(len(t) for t in done.values())
    effects = (
        "<div class='cards'>" + "".join(cards) + "</div>"
        if cards
        else missing("This study names no searched axis.")
        if finished
        else missing("No finished trial yet.")
    )
    drew = (
        f"<p class='cap'>The search drew the dataset ({html.escape(', '.join(datasets))}): "
        "every picture is drawn per dataset, and the best trial filled is the best on "
        "that dataset. Scores of two datasets are never compared.</p>"
        if several
        else ""
    )
    base = study.base
    curves = "".join(
        fig_curves(
            study,
            on[d],
            base if base is not None and str(base.dataset) == d else None,
            d if several else None,
        )
        for d in datasets
    )
    expects = (
        f"<p><b>Expected</b> {html.escape(study.expects)}</p>"
        if study.expects
        else "<p class='expects open'><b>Expected</b> not said.</p>"
    )
    counts = f"{plural(study.finished, 'finished trial')}, {plural(study.pruned, 'pruned trial')}"
    return (
        f"<section><h2>{html.escape(study.label)} · "
        f"<span class='name'>{html.escape(study.idea_name or study.key)}</span></h2>"
        f"<p><b>Idea</b> {html.escape(study.idea or 'the study carries no idea')}</p>"
        f"{expects}<p class='cap'>{html.escape(study.key)} · {html.escape(study.objective)}"
        f" on {html.escape(study.on_text())} · {html.escape(counts)} · "
        f"{html.escape(budget_text(study.budget, len(trials)))}</p></section>"
        "<section><h2>Each searched axis</h2><p class='cap'>One dot per finished "
        f"trial, the best filled: {html.escape(study.objective)} above, the seconds "
        "it trained below. Read an axis on its own: the others varied at the same "
        f"time.</p>{drew}{effects}</section>"
        "<div class='two'>"
        f"<section><h2>Learning curves</h2>{curves}</section>"
        f"<section><h2>The winner's other metrics</h2>{metrics_table(study)}</section>"
        "</div>"
    )


ATLAS_SECONDS = 60


def atlas_page(command: str, folder: Path) -> tuple[bytes | None, str]:
    """The page `command view` writes of `folder`, or None and why it gave none."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "atlas.html"
        try:
            ran = subprocess.run(
                [command, "view", str(folder), "--out", str(out), "--no-open"],
                capture_output=True,
                text=True,
                timeout=ATLAS_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"{command} took more than {ATLAS_SECONDS} seconds"
        except OSError as e:
            return None, f"{command} did not start: {e}"
        if ran.returncode != 0 or not out.exists():
            why = (ran.stderr or ran.stdout).strip().splitlines()[-1:] or ["no output"]
            return None, f"{command} failed: {why[0]}"
        return out.read_bytes(), ""


def design_tab(
    folder: Path,
    which: Callable[[str], str | None] | None = None,
    run: Callable[[str, Path], tuple[bytes | None, str]] = atlas_page,
) -> str:
    """The page atlas draws of `folder`, inside the tab: `atlas-code-nightly`, and
    `atlas-code` when that is missing, fails or takes longer than `ATLAS_SECONDS`. One
    line when neither gives a page."""
    find = which or shutil.which
    commands = [c for c in ATLAS_COMMANDS if find(c)]
    if not commands:
        return (
            "<section><h2>Design</h2>"
            f"<p class='missing'>Atlas is not on this machine, so the design of "
            f"{html.escape(shown(folder))} is not drawn here: install it as "
            f"{html.escape(ATLAS_SETUP)} says.</p></section>"
        )
    reasons = []
    for command in commands:
        page, why = run(command, folder)
        if page is not None:
            data = base64.b64encode(page).decode()
            return (
                "<section class='design'><h2>Design</h2><p class='cap'>"
                f"{html.escape(command)} view {html.escape(shown(folder))}: the stages of "
                "the design, the shape on every arrow, and where an option leaves a stage "
                "out.</p><iframe id='atlas' title='the design in atlas'></iframe>"
                f"<script type='application/octet-stream' id='atlas-page'>{data}</script>"
                "</section>"
            )
        reasons.append(why)
    return f"<section><h2>Design</h2>{missing('; '.join(reasons))}</section>"


def font_faces() -> str:
    """The course's fonts, inside the page."""
    faces = []
    for family, weight, name in (
        ("Outfit", 400, "Outfit-400.woff2"),
        ("Outfit", 500, "Outfit-500.woff2"),
        ("JetBrains Mono", 400, "JetBrainsMono-400.woff2"),
    ):
        path = PRESENTATIONS / "fonts" / name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            faces.append(
                f"@font-face{{font-family:'{family}';font-weight:{weight};font-display:swap;"
                f"src:url(data:font/woff2;base64,{data}) format('woff2')}}"
            )
    return "".join(faces)


def course_tokens() -> str:
    """The colour tokens of `presentations/course-ui.css`, light and dark."""
    css = (PRESENTATIONS / "course-ui.css").read_text()
    head, sep, _ = css.partition("\nhtml{")
    return head if sep else css


CSS = """
:root{--slot-ink:#a8310f}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--slot-ink:var(--course-orange)}}
:root[data-theme="dark"]{--slot-ink:var(--course-orange)}
*{box-sizing:border-box}
body{margin:0;background:var(--course-paper);color:var(--course-ink);
  font-family:Outfit,Arial,sans-serif;font-size:16px;line-height:1.45}
main{max-width:1180px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
h1{font-size:26px;font-weight:500;margin:0}
h2{font-size:18px;font-weight:500;margin:0 0 6px;color:var(--course-ink)}
h3{font-size:15px;font-weight:500;margin:0 0 8px}
h4{font-size:14px;font-weight:500;margin:0;font-family:'JetBrains Mono',monospace}
p{margin:4px 0}
code,.name,.metrics td{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--course-blue);
  overflow-wrap:anywhere}
.sub,.cap,.foot,.meta,.ylab{color:var(--course-muted);font-size:13.5px;overflow-wrap:anywhere}
.foot{font-size:12px;margin-top:2px}
.ylab{margin:8px 0 0;font-size:12px}
.ds{margin:10px 0 0;font-size:13px;font-weight:500;color:var(--course-ink)}
section{background:var(--course-surface);border-radius:10px;padding:16px 18px;margin:14px 0;min-width:0}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0 0;border-bottom:1px solid var(--course-rule)}
.tabs button{font:inherit;font-size:14px;background:none;border:0;border-bottom:3px solid transparent;
  color:var(--course-muted);padding:8px 10px;cursor:pointer;max-width:100%;overflow-wrap:anywhere}
.tabs button[aria-selected=true]{color:var(--course-ink);border-bottom-color:var(--course-green)}
.theme{font:inherit;font-size:13px;background:var(--course-surface);color:var(--course-ink);
  border:1px solid var(--course-rule);border-radius:6px;padding:4px 10px;cursor:pointer}
.slot{border-radius:8px;padding:10px 14px;margin:8px 0}
.slot.filled{background:var(--course-paper);border-left:4px solid var(--course-green);font-size:17px}
.slot.open{border:2px dashed var(--slot-ink);color:var(--slot-ink)}
.slot.open p{font-style:italic}
.slot.absent{border:2px dotted var(--course-rule);color:var(--course-muted)}
.slot .tag{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--course-muted)}
.slot.picture{text-align:center}
.slot.picture svg,.slot.picture img{max-width:100%;max-height:360px;height:auto}
.warn{color:var(--slot-ink);font-size:13px}
.missing{color:var(--course-muted);font-size:13.5px;background:var(--course-paper);border-radius:6px;
  padding:8px 12px;margin:6px 0;overflow-wrap:anywhere}
.ideas{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:22px}
.ideas li{position:relative;flex:1 1 240px;min-width:0;max-width:380px}
.ideas li+li::before{content:'→';position:absolute;left:-19px;top:40%;color:var(--course-muted);font-size:16px}
.idea{display:flex;flex-direction:column;gap:3px;text-align:left;width:100%;height:100%;font:inherit;
  color:inherit;background:var(--course-paper);border:2px solid var(--course-rule);border-radius:8px;
  padding:10px 12px;cursor:pointer}
.idea[aria-pressed=true]{border-color:var(--course-blue)}
.idea .label{font-size:12px;color:var(--course-muted);text-transform:uppercase;letter-spacing:.04em}
.idea .words{font-size:16px}
.expects{font-size:13.5px}
.expects b,.slot b{font-weight:500;color:var(--course-muted)}
.expects.open{color:var(--slot-ink);font-style:italic}
.builds{font-size:12.5px;color:var(--course-muted);overflow-wrap:anywhere}
.gain{font-size:14px;font-weight:500;color:var(--course-muted)}
.gain.better{color:var(--course-green)}
.gain.worse{color:var(--course-orange)}
.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:8px}
.three[hidden]{display:none}
.panel{background:var(--course-paper);border-radius:8px;padding:12px;min-width:0}
.panel .missing,.card .missing{background:var(--course-surface)}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr));gap:12px}
.card{background:var(--course-paper);border-radius:8px;padding:10px;min-width:0}
svg{width:100%;height:auto;display:block;overflow:visible}
svg .t{font-size:12px;fill:var(--course-ink)}
svg .s{font-size:10px;fill:var(--course-ink)}
svg .m{fill:var(--course-muted)}
svg .b{font-weight:500}
svg .on{fill:var(--course-paper)}
svg .track{stroke:var(--course-muted);stroke-width:1}
svg .err{stroke:var(--course-ink);stroke-width:2}
svg .win{fill:var(--course-blue)}
svg .base{fill:var(--course-paper);stroke:var(--course-orange);stroke-width:2.5}
svg .dot{fill:none;stroke:var(--course-muted);stroke-width:1.2}
svg .seg{fill:var(--course-paper);stroke:var(--course-rule)}
svg .seg.on{fill:var(--course-blue);stroke:var(--course-blue)}
svg .fixed{fill:var(--course-paper);stroke:var(--course-rule);stroke-dasharray:3 3}
svg .band{fill:var(--course-orange-wash);opacity:.7}
svg .baseline{stroke:var(--course-orange);stroke-width:1.5;stroke-dasharray:2 3}
svg .faint{fill:none;stroke:var(--course-blue);stroke-width:1.1;stroke-opacity:.7}
svg .wincurve{fill:none;stroke:var(--course-blue);stroke-width:2.6}
svg .basecurve{fill:none;stroke:var(--course-orange);stroke-width:2;stroke-dasharray:2 3}
.stages{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:14px}
.stages li{border:1.5px solid var(--course-rule);border-radius:6px;padding:5px 9px;display:flex;
  flex-wrap:wrap;gap:2px 10px;align-items:baseline;position:relative}
.stages li+li::before{content:'↓';position:absolute;top:-16px;left:12px;font-size:12px;line-height:1;color:var(--course-muted)}
.stages li b{font-weight:500;font-family:'JetBrains Mono',monospace;font-size:13px}
.stages .in{border-style:none;padding-left:0}
.stages .optional{border-style:dashed;border-color:var(--course-blue)}
.stages .optional.off{opacity:.6}
.stages .opt{font-size:11.5px;color:var(--course-muted);width:100%}
.gain+ul.plain{margin:4px 0}
ul.plain{list-style:none;padding:0;font-size:13px}
ul.plain b{font-weight:500}
table.metrics{border-collapse:collapse;font-size:13px;width:100%}
table.metrics th{text-align:left;font-weight:400;padding:3px 8px 3px 0;overflow-wrap:anywhere}
table.metrics td{padding:3px 6px;text-align:right}
table.metrics thead th{color:var(--course-muted);font-size:12px}
table.per-class tbody tr:nth-child(odd){background:var(--course-paper)}
details{margin-top:10px}
summary{cursor:pointer;color:var(--course-muted);font-size:13.5px}
.design iframe{width:100%;height:78vh;border:1px solid var(--course-rule);border-radius:8px;
  background:var(--course-surface)}
[role=tabpanel][hidden]{display:none}
@media (max-width:860px){
  .three,.two{grid-template-columns:minmax(0,1fr)}
  .ideas{flex-direction:column;gap:20px}
  .ideas li{max-width:none;flex:none}
  .ideas li+li::before{content:'↓';left:16px;top:-19px}
  section{padding:12px}
  h1{font-size:22px}
}
"""

SCRIPT = """
(function () {
  const root = document.documentElement;
  const key = 'mads-course-theme';
  const button = document.getElementById('theme');
  try { const saved = JSON.parse(localStorage.getItem(key));
        if (saved === 'dark' || saved === 'light') root.dataset.theme = saved; } catch (e) {}
  const media = matchMedia('(prefers-color-scheme: dark)');
  const dark = () => root.dataset.theme ? root.dataset.theme === 'dark' : media.matches;
  const label = () => { button.textContent = dark() ? 'light theme' : 'dark theme'; };
  button.addEventListener('click', function () {
    root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem(key, JSON.stringify(root.dataset.theme)); } catch (e) {}
    label();
  });
  media.addEventListener('change', label);
  label();

  const tabs = Array.from(document.querySelectorAll('.tabs button'));
  function atlas() {
    const frame = document.getElementById('atlas');
    const data = document.getElementById('atlas-page');
    if (!frame || !data || frame.dataset.loaded) return;
    const bytes = Uint8Array.from(atob(data.textContent.trim()), function (c) { return c.charCodeAt(0); });
    frame.srcdoc = new TextDecoder('utf-8').decode(bytes);
    frame.dataset.loaded = '1';
  }
  function show(id) {
    if (!document.getElementById('tab-' + id)) id = 'overview';
    tabs.forEach(function (t) {
      const on = t.dataset.tab === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      document.getElementById('tab-' + t.dataset.tab).hidden = !on;
    });
    if (id === 'design') atlas();
    try { history.replaceState(null, '', '#' + id); } catch (e) {}
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { show(t.dataset.tab); });
    t.addEventListener('keydown', function (e) {
      const to = {ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1}[e.key];
      if (to === undefined) return;
      e.preventDefault();
      const next = tabs[(to + tabs.length) % tabs.length];
      show(next.dataset.tab);
      next.focus();
    });
  });
  show(decodeURIComponent(location.hash.slice(1)) || 'overview');

  const boxes = Array.from(document.querySelectorAll('.idea'));
  const cap = document.getElementById('panels-cap');
  function pick(slug) {
    boxes.forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.study === slug ? 'true' : 'false'); });
    document.querySelectorAll('[data-panels]').forEach(function (p) { p.hidden = p.dataset.panels !== slug; });
    const box = boxes.find(function (b) { return b.dataset.study === slug; });
    if (cap && box) cap.querySelector('b').textContent = box.querySelector('.label').textContent;
  }
  boxes.forEach(function (b) { b.addEventListener('click', function () { pick(b.dataset.study); }); });
  if (boxes.length) pick(boxes[boxes.length - 1].dataset.study);
})();
"""


def studies_text(n: int) -> str:
    return f"{n} study" if n == 1 else f"{n} studies"


def build_page(
    studies: Sequence[Study],
    lesson: int,
    login: str,
    source: str,
    own: Mapping[str, str] | None,
    page: Path | None,
    design: str,
) -> str:
    """The page of `login`'s `studies` of `lesson`, with what they wrote in `own`, read
    from the page.toml at `page`; `own` None for a page of submitted runs without one.
    A picture is looked for in the folder of `page`."""
    folder = page.parent if page is not None else COURSE_ROOT
    tabs = [("overview", "Overview"), ("design", "Design")] + [
        (s.slug, s.label) for s in studies
    ]
    nav = "".join(
        f"<button type='button' role='tab' id='tabbtn-{t}' data-tab='{t}' "
        f"aria-controls='tab-{t}' tabindex='-1'>"
        f"{html.escape(name)}</button>"
        for t, name in tabs
    )
    body = (
        f"<div role='tabpanel' id='tab-overview' aria-labelledby='tabbtn-overview'>{overview(studies, own, folder)}</div>"
        f"<div role='tabpanel' id='tab-design' aria-labelledby='tabbtn-design' hidden>{design}</div>"
        + "".join(
            f"<div role='tabpanel' id='tab-{s.slug}' aria-labelledby='tabbtn-{s.slug}' hidden>{study_tab(s)}</div>"
            for s in studies
        )
    )
    title = (
        f"Lesson {lesson}: what {login} studied"
        if login
        else f"Lesson {lesson}: your study"
    )
    yours = (
        f"What is yours on the page comes from {html.escape(shown(page))}."
        if own is not None and page is not None
        else "What is the student's own on the page is not part of the submitted runs."
    )
    runs = sum(len(s.df) for s in studies)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{font_faces()}{course_tokens()}{CSS}</style></head><body><main>"
        f"<header><div><h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{studies_text(len(studies))}, {plural(runs, 'run')}, read from "
        f"{html.escape(source)}. {yours}</p></div>"
        "<button type='button' class='theme' id='theme'>dark theme</button></header>"
        f"<nav class='tabs' role='tablist'>{nav}</nav>{body}"
        f"</main><script>{SCRIPT}</script></body></html>"
    )


def page_of(
    lesson: int,
    folder: Path,
    *,
    store: Store | None = None,
    submitted: str | None = None,
    runs_repo: Path | None = None,
    fetch: bool = True,
    page: Path | None = None,
    which: Callable[[str], str | None] | None = None,
) -> tuple[str, list[Study], str]:
    """The page of `lesson`, its studies and its login: from `store` (the default
    store), or from the runs `submitted` names in `runs_repo`. `folder` is the lesson's
    `exercises/` folder: atlas draws it, and it holds the default page.toml. A page of
    submitted runs reads a page.toml only when `page` names one; otherwise the default
    one is read, and made once runs are found. No runs stop the script."""
    if submitted:
        repo = (runs_repo or COURSE_ROOT.parent / "mads-runs").resolve()
        subs, login, source = read_submitted(repo, lesson, submitted, fetch)
    else:
        subs, login, source = read_store(store or Store.default(), lesson)
    if not subs:
        sys.exit(
            f"no runs of lesson {lesson} in {source}"
            + ("" if submitted else f": run `make study LESSON={lesson}` first")
        )
    toml = page or (None if submitted else folder / "page.toml")
    own = None if toml is None else read_own(toml, lesson, create=page is None)
    studies = studies_of(subs)
    return (
        build_page(
            studies, lesson, login, source, own, toml, design_tab(folder, which)
        ),
        studies,
        login,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lesson", type=int, required=True)
    parser.add_argument(
        "--submitted",
        metavar="LOGIN",
        help="read this login's submitted runs from the class runs repository",
    )
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(
            os.environ.get("MADS_RUNS_REPO", COURSE_ROOT.parent / "mads-runs")
        ),
        help="the clone of the class runs repository (with --submitted)",
    )
    parser.add_argument(
        "--no-pull", action="store_true", help="with --submitted: read main as fetched"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="the page (default: artefacts/study-page/lesson<N>/index.html)",
    )
    parser.add_argument(
        "--page",
        type=Path,
        help="the page.toml to read (default: notebooks/<N>_*/exercises/page.toml, made "
        "when missing; none with --submitted)",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the page in the browser"
    )
    args = parser.parse_args()

    html_, studies, login = page_of(
        args.lesson,
        exercises_folder(args.lesson),
        submitted=args.submitted,
        runs_repo=args.runs_repo,
        fetch=not args.no_pull,
        page=args.page,
    )
    out = (
        args.out
        or COURSE_ROOT
        / "artefacts"
        / "study-page"
        / f"lesson{args.lesson}"
        / "index.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_)
    print(
        f"{studies_text(len(studies))}, {plural(sum(len(x.df) for x in studies), 'run')} of {login or 'this store'}"
    )
    print(f"page: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
