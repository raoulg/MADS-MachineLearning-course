"""A student's page: one page that shows what one student studied in a lesson.

    uv run scripts/study_page.py --lesson 1                      # from your own store
    uv run scripts/study_page.py --lesson 1 --open               # ... served and opened
    uv run scripts/study_page.py --lesson 1 --submitted raoulg   # from a login's submitted runs
    make page LESSON=1                                           # the second

The page is the class view (`scripts/class_analysis.py`) read for one student. By
default it reads the student's own store (`hpo.record.Store.default()`): every study of
the lesson, the search and its `-runs` companion, and the handed-out exercise's runs when
they are there. With `--submitted <login>` it reads that login's runs of the lesson on
the class runs repository's `main`, as `class_analysis.py` reads them, so a page can be
built from published runs alone.

It writes one self-contained page, `artefacts/study-page/lesson<N>/index.html`, and
prints its path. With `--open`, a page of the student's own store is served on this
machine until Ctrl-C (`page_server`), so that the views they star are saved in
`exercises/stars.json` and shown again the next time; `--no-serve` opens the file, where a
star lasts until the page is closed. The page has six tabs:

- **Overview**: the student's message and picture, their ideas as a tree of titles (each
  under the idea it builds on, opened to what it expected and found), what surprised and
  what comes next, and, when the student asks for it, everything they starred.
- **Studies**: one study at a time, picked by its title: its idea and expectation, three
  panels (its architecture, its search space, its winner next to its baseline), an
  explorer of its trials (two fields against each other and a third in colour, chosen by
  the student, with the views they starred beside it), the learning curves, and the
  winner's other metrics.
- **Scaling**: two pictures over every trial, the score against the parameters of the
  head and against the FLOPs of training it, and the training curves (train and test
  loss) per size of the head; each picture the student can star.
- **Starred**: every starred picture and every starred view of every study, gathered by
  study.
- **Design**: the page atlas draws of the lesson's `exercises/` folder, opened on the
  forward of its design.

What is the student's own comes from `notebooks/<lesson>/exercises/page.toml`, or the
file `--page` names: the message, the picture, what surprised, what next, and a title
per study under `[titles]`. A key left
empty shows as an open slot that asks its question. Once runs of the lesson are found,
the script creates the default file with empty keys when it is missing, and never writes
it otherwise. The file is not tracked: a student commits theirs on a branch of their own.
A page of submitted runs reads no page.toml unless `--page` names one; its own slots say
they are not part of the submitted runs.

Every score is read on its own study's objective and its own dataset: scores of two
datasets, or of two objectives, are never put on one axis. The capacity pictures of the
overview put the trials of several studies on one axis when they share the dataset and
the objective: one study searches a narrow band of head sizes, and only all of them
together show where a bigger head stops paying.
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import json
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_analysis import (
    HEAD_SIZES,
    OFF,
    ON,
    TRAIN_ROWS,
    budget_text,
    chip_value,
    column,
    decade_ticks,
    flops_text,
    idea_depths,
    number,
    page_metric,
    plural,
    read_class,
    runs_frame,
    signed,
    size_title,
    standard_error,
    study_of,
    train_flops,
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

# A title of your own for a study, instead of the name the page gives it: the study's
# name in study.py, then your title. The page shows that name when you open an idea.
[titles]
# "wide-head-on-dtd" = "a wider head"
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
        path.write_text(PAGE_TOML.format(lesson=lesson), encoding="utf-8")
        print(f"wrote {shown(path)}: fill it in, and build the page again")
    try:
        held = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{path} is not valid TOML: {e}")
    return {key: str(held.get(key) or "").strip() for key in OWN}


def read_titles(path: Path | None) -> dict[str, str]:
    """The `[titles]` of the page.toml at `path`: a study's name to the title the student
    gives it. Empty without a file, without the table, or for a value that is not text."""
    if path is None or not path.exists():
        return {}
    try:
        held = tomllib.loads(path.read_text(encoding="utf-8")).get("titles")
    except tomllib.TOMLDecodeError:
        return {}
    if not isinstance(held, dict):
        return {}
    return {
        str(k): v.strip() for k, v in held.items() if isinstance(v, str) and v.strip()
    }


def titled(studies: Sequence[Study], titles: Mapping[str, str]) -> None:
    """Give every study the student titled in `titles` (by its label or its key) that
    title as the name it goes by on the page."""
    for s in studies:
        s.short = titles.get(s.label) or titles.get(s.key) or s.short


TITLE_MAX = 80


def write_title(path: Path, study: str, title: str) -> None:
    """Set the title of `study` (its name in study.py) under `[titles]` in the page.toml
    at `path`, or take it out when `title` is empty; every other line stays as the
    student wrote it. A `ValueError` for a title the page cannot hold."""
    title = title.strip()
    if "\n" in title or len(title) > TITLE_MAX or not study or "\n" in study:
        raise ValueError(f"a title is one line of at most {TITLE_MAX} characters")
    key = json.dumps(study, ensure_ascii=False)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    head = next((i for i, x in enumerate(lines) if x.strip() == "[titles]"), None)
    if head is None:
        lines += ["", "[titles]"]
        head = len(lines) - 1
    end = next(
        (i for i in range(head + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    mine = [
        i
        for i in range(head + 1, end)
        if lines[i].split("=", 1)[0].strip() in (key, study)
        and not lines[i].lstrip().startswith("#")
    ]
    for i in reversed(mine):
        del lines[i]
        end -= 1
    if title:
        at = end
        while at > head + 1 and not lines[at - 1].strip():
            at -= 1
        lines.insert(at, f"{key} = {json.dumps(title, ensure_ascii=False)}")
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # a TOMLDecodeError is a ValueError: nothing is written
    path.write_text(text, encoding="utf-8")


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
        body = path.read_text(encoding="utf-8")
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
        held = kept.read_text(encoding="utf-8").strip() if kept.is_file() else ""
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


def difference(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """`a` against `b` in points, and the standard error of that difference in points
    from the spread of their repeats; NaN for the error when one side has no spread."""
    se = standard_error((a["std"], a["n"]), (b["std"], b["n"]))
    return 100 * (a.score - b.score), 100 * se


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
    short: str = ""
    auto: str = ""  # the short name the page gives it, before a title of the student's

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

    def gain(self) -> tuple[float, float]:
        """The winner against its own baseline (`difference`); NaN without both."""
        if self.base is None or self.final is None:
            return math.nan, math.nan
        return difference(self.final, self.base)

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
    for s, name in zip(out, short_names(out), strict=True):
        s.short = s.auto = name
    return out


def short_names(studies: Sequence[Study]) -> list[str]:
    """The name each study goes by on the page: its label without what every label of
    the student shares (their own prefix, the dataset at the end) and without
    `on-<dataset>`, the words joined by spaces. A name two studies would share stays the
    whole label."""
    words = [s.label.split("-") for s in studies]
    for w in words:
        if len(w) > 2 and w[-2] == "on":
            del w[-2:]
    own = [w for s, w in zip(studies, words, strict=True) if not s.handed_out]
    if len(own) > 1:
        shortest = min(len(w) for w in own) - 1
        lead = 0
        while lead < shortest and len({w[lead] for w in own}) == 1:
            lead += 1
        for w in own:
            del w[:lead]
    names = [" ".join(w) for w in words]
    return [
        name if names.count(name) == 1 else s.label
        for s, name in zip(studies, names, strict=True)
    ]


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


LABEL_CHAR = 7.4  # the width of one character of a small label (`svg .s`)


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
                    f"rx='1' class='{'seg on' if on else 'seg'}'/>"
                )
                fits = int((w - 10) / LABEL_CHAR)
                short = c if len(c) <= fits else c[: max(1, fits - 1)] + "…"
                body.append(
                    text(x + w / 2, y + 16, short, "s on" if on else "s", "middle")
                )
        else:
            body.append(
                f"<rect x='{left}' y='{y + 2}' width='{right - left}' height='20' rx='1' class='fixed'/>"
            )
            body.append(text(left + 8, y + 16, f"= {a.value}", "s m"))
    height = 8 + len(axes) * row
    note = "" if won else missing("No winner yet: no value is marked.")
    return (
        svg(width, height, "".join(body), f"the search space of {study.label}")
        + "<p class='foot'>● the winner's value · bold: searched</p>"
        + note
    )


def se_text(score: float, std: object, n: object) -> str:
    """A mean score ± its standard error over the runs; the score alone with one run."""
    se = standard_error((std, n))
    return f"{score:.4f}" + ("" if math.isnan(se) else f" ± {se:.4f}")


def spread_text(score: float, std: object, n: object) -> str:
    runs = number(n)
    out = se_text(score, std, n)
    if not math.isnan(runs):
        out += f" · {plural(int(runs), 'run')}"
    return out


def gain_words(points: float, se: float) -> str:
    """The gain in points, ± the standard error of the difference where both sides were
    run more than once: how far the winner moved from its own baseline. One run is a
    result too, shown without a spread."""
    if math.isnan(points):
        return "no winner against a baseline yet"
    return f"{pm(points, se)} against its baseline"


def fig_results(study: Study) -> str:
    """The baseline and the winner on the study's objective, each ± the standard error
    of its mean over its runs."""
    b, f = study.base, study.final
    rows = [(n, r) for n, r in (("baseline", b), ("winner", f)) if r is not None]
    if not rows:
        return missing(
            "No finished trial yet." if study.finished == 0 else
            "No baseline or winner rerun was logged."
        )  # fmt: skip
    ses = [standard_error((r["std"], r["n"])) for _, r in rows]
    lows = [
        r.score - (0 if math.isnan(e) else e)
        for (_, r), e in zip(rows, ses, strict=True)
    ]
    highs = [
        r.score + (0 if math.isnan(e) else e)
        for (_, r), e in zip(rows, ses, strict=True)
    ]
    low, high = min(lows), max(highs)
    pad = max((high - low) * 0.25, 0.002)
    low, high = low - pad, high + pad
    width, left, right = 320, 70, 308
    body = []
    for i, ((name, r), s) in enumerate(zip(rows, ses, strict=True)):
        y = 22 + i * 40
        x = left + fraction(r.score, low, high, False) * (right - left)
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
    gain = gain_words(*study.gain())
    lines = "".join(
        f"<li><b>{n}</b> {html.escape(spread_text(r.score, r['std'], r['n']))}</li>"
        for n, r in rows
    )
    return (
        svg(width, y + 22, "".join(body), f"winner and baseline of {study.label}")
        + f"<p class='gain'>{html.escape(gain)}</p>"
        + f"<ul class='plain'>{lines}</ul>"
        + f"<p class='foot'>{html.escape(study.objective)} on {html.escape(study.on_text())}"
        " · ± one standard error over the runs</p>"
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


# What a scaling picture sets the score against: its axis, and what that axis counts.
SCALES = {
    "params": (
        "parameters of the head",
        "parameters: how heavy the head is, the same for every image in every epoch",
    ),
    "flops": (
        "FLOPs of training the head",
        (
            "FLOPs (floating point operations) of all training: about 6 × parameters × "
            "training rows × epochs"
        ),
    ),
}


def fig_flops(
    objective: str,
    done: pd.DataFrame,
    b: pd.Series | None,
    dataset: str | None,
    by: str = "flops",
) -> str:
    """The objective of every finished trial of `done` (one dataset's) against the
    parameters of its head (`by="params"`) or the FLOPs of training it
    (`class_analysis.train_flops`), on a log axis, the winner filled and the baseline
    `b` as its score ± one standard deviation: where the dots stop rising as the head
    grows, more head buys no more score."""
    head = f"<p class='ds'>on {html.escape(dataset)}</p>" if dataset else ""
    if done.empty:
        return head + missing("No finished trial yet.")
    t = done.assign(
        flops=train_flops(done) if by == "flops" else column(done, "n_params")
    )
    t = t[t.flops.notna() & (t.flops > 0)]
    if t.empty:
        return head + missing(
            "The FLOPs of these trials are not known: the page knows the training rows "
            f"of {', '.join(sorted(TRAIN_ROWS))} only, and every trial needs n_params."
            if by == "flops"
            else "These trials logged no n_params."
        )
    axis_name, counts = SCALES[by]
    ticks, words = decade_ticks(t.flops)
    low, high = ticks[0], ticks[-1]
    ys = list(t.score)
    if b is not None:
        s = 0.0 if math.isnan(number(b["std"])) else float(b["std"])
        ys += [b.score - s, b.score + s]
    q10, q90 = np.quantile(t.score, [0.1, 0.9])
    floor = q10 - (q90 - q10)  # below it a dot is drawn on the floor, as a triangle
    ylow, yhigh = max(min(ys), floor), max(ys)
    pad = (yhigh - ylow) * 0.06 or 0.01
    ylow, yhigh = ylow - pad, yhigh + pad
    width, height, left, right, top, bottom = 320, 210, 46, 312, 10, 176

    def ax(x: float) -> float:
        return left + fraction(x, low, high, True) * (right - left)

    def ay(y: float) -> float:
        return bottom - fraction(y, ylow, yhigh, False) * (bottom - top)

    body = [
        f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' class='track'/>",
        f"<line x1='{left}' x2='{left}' y1='{top}' y2='{bottom}' class='track'/>",
        text(left - 4, top + 8, tick(yhigh), "s m", "end"),
        text(left - 4, bottom, tick(ylow), "s m", "end"),
        text((left + right) / 2, bottom + 28, axis_name, "s", "middle"),
    ]  # fmt: skip
    for v, w in zip(ticks, words, strict=True):
        body.append(text(ax(v), bottom + 14, w.replace(" ", ""), "s m", "middle"))
    if b is not None:
        s = 0.0 if math.isnan(number(b["std"])) else float(b["std"])
        body.append(
            f"<rect x='{left}' y='{ay(b.score + s):.1f}' width='{right - left}' "
            f"height='{max(ay(b.score - s) - ay(b.score + s), 1):.1f}' class='band'/>"
            f"<line x1='{left}' x2='{right}' y1='{ay(b.score):.1f}' "
            f"y2='{ay(b.score):.1f}' class='baseline'/>"
        )
    below = 0
    enc = done.p_encoder.dropna() if "p_encoder" in done else pd.Series(dtype=object)
    by = "p_encoder" if enc.nunique() > 1 else "by_study"
    names = (
        sorted({str(v) for v in done[by].dropna()}, key=natural_key)
        if by in done
        else []
    )
    hue = (
        {n: HUES[i % len(HUES)] for i, n in enumerate(names)} if len(names) > 1 else {}
    )
    for r in t.sort_values("winner").itertuples():
        hint = (
            f"<title>{html.escape(flops_text(r.flops))} "
            f"{'FLOPs' if by == 'flops' else 'params'} · {r.score:.3f}</title>"
        )
        x = ax(r.flops)
        if r.score < ylow:
            below += 1
            body.append(
                f"<path d='M{x - 4:.1f},{bottom - 7} L{x + 4:.1f},{bottom - 7} "
                f"L{x:.1f},{bottom - 1} Z' class='dot'>{hint}</path>"
            )
            continue
        paint = (
            f" style='fill:{hue[str(getattr(r, by))]};stroke:{hue[str(getattr(r, by))]}'"
            if hue and not r.winner
            else ""
        )
        body.append(
            f"<circle cx='{x:.1f}' cy='{ay(float(r.score)):.1f}' "
            f"r='{5 if r.winner else 3.2}' class='{'win' if r.winner else 'dot'}'{paint}>"
            f"{hint}</circle>"
        )
    left_out = len(done) - len(t)
    key = (
        "<p class='legend'>"
        + ("<b>encoder</b> " if by == "p_encoder" else "<b>study</b> ")
        + " ".join(
            f"<svg class='sw' viewBox='-6 -6 12 12' aria-hidden='true'><circle r='4.6' "
            f"fill='{c}'/></svg>{html.escape(n)}"
            for n, c in hue.items()
        )
        + "</p>"
        if hue
        else ""
    )
    return (
        head
        + f"<p class='ylab'>{html.escape(objective)}</p>"
        + svg(width, height, "".join(body), f"{objective} against {axis_name}")
        + key
        + f"<p class='foot'>one dot per finished trial, the best ringed · {counts} · "
        "dotted with its band: the baseline ± one standard deviation"
        + (f" · ▼ on the floor: {plural(below, 'trial')} far below the others" if below else "")
        + (f" · {plural(left_out, 'trial')} without known FLOPs left out" if left_out else "")
        + "</p>"
    )  # fmt: skip


# The Okabe and Ito hues, in the explorer's order: a study is one colour on every picture.
HUES = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
        "#5a544d")  # fmt: skip


def natural_key(v: str) -> tuple[int, float, str]:
    """The explorer's order of categories: numbers by value, the rest by name."""
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v.casefold())


MIN_MEDIAN = 3  # curves of a panel that reach an epoch before its median is drawn


def fig_overfitting(trials: pd.DataFrame, dataset: str | None) -> str:
    """The train loss and the test loss after every epoch of every trial of `trials` (one
    dataset's) that logged them, each in its own colour, one panel per size of the head
    (`class_analysis.HEAD_SIZES`), with both medians of a panel bold where it has enough
    curves and the gap between them shaded. A test loss that turns up while the train loss keeps falling is
    the head learning its training images by heart. Nothing else is coloured: depth,
    skip, norm, dropout and the rest all move these curves at once."""
    head = f"<p class='ds'>on {html.escape(dataset)}</p>" if dataset else ""
    has = trials[trials.losses.map(lambda v: isinstance(v, (list, tuple)))]
    if has.empty:
        return head + missing("No trial of this study logged its train and test loss.")
    sizes = column(has, "n_params")
    epochs = max(len(c[1]) for c in has.losses)
    rows = TRAIN_ROWS.get(str(dataset or has.dataset.iloc[0]))
    width, height, left, right, ptop, bottom = 300, 190, 34, 292, 12, 160

    def ax(i: int) -> float:
        return left + (i / max(epochs - 1, 1)) * (right - left)

    def ay(y: float) -> float:
        return bottom - fraction(min(y, top), 0.0, top, False) * (bottom - ptop)

    def path(c: Sequence[float]) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{ax(i):.1f},{ay(v):.1f}" for i, v in enumerate(c)
        )

    def median_of(part: pd.DataFrame, which: int = 1) -> list[float]:
        curves = [list(c[which]) for c in part.losses]
        out = []
        for e in range(epochs):
            at = [c[e] for c in curves if len(c) > e]
            if len(at) < MIN_MEDIAN:
                break
            out.append(float(np.median(at)))
        return out if len(out) > 1 else []

    cells = [has[(sizes >= lo) & (sizes < hi)] for lo, hi, _ in HEAD_SIZES]
    medians = [median_of(cell) for cell in cells]
    # the axis holds nearly every point of the medians; a curve above it runs along the top
    held = [v for m in medians for v in m] or [v for c in has.losses for v in c[1]]
    top = 1.15 * float(np.quantile(held, 0.95))
    panels_ = []
    for (lo, hi, words), cell, median in zip(HEAD_SIZES, cells, medians, strict=True):
        title = "".join(
            f"<p class='{'ds' if i == 0 else 'foot'}'>{html.escape(line)}</p>"
            for i, line in enumerate(size_title(lo, hi, words, rows).split("<br>"))
        )
        if cell.empty:
            panels_.append(
                f"<div class='card'>{title}{missing('No trial of this size.')}</div>"
            )
            continue
        body = [
            f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' class='track'/>",
            f"<line x1='{left}' x2='{left}' y1='{ptop}' y2='{bottom}' class='track'/>",
            text(left - 4, ptop + 8, tick(top), "s m", "end"),
            text(left - 4, bottom, "0", "s m", "end"),
            text(left, bottom + 14, "1", "s m"),
            text(right, bottom + 14, str(epochs), "s m", "end"),
            text((left + right) / 2, bottom + 26, "epoch", "s", "middle"),
        ]  # fmt: skip
        for c in cell.losses:
            body.append(f"<path d='{path(c[0])}' class='loss train'/>")
            body.append(f"<path d='{path(c[1])}' class='loss test'/>")
        trained = median_of(cell, 0)
        if median and trained:
            n = min(len(median), len(trained))
            back = " ".join(
                f"L{ax(i):.1f},{ay(trained[i]):.1f}" for i in reversed(range(n))
            )
            body.append(f"<path d='{path(median[:n])} {back} Z' class='gap'/>")
        if trained:
            body.append(f"<path d='{path(trained)}' class='loss train median'/>")
        if median:
            body.append(f"<path d='{path(median)}' class='loss test median'/>")
        panels_.append(
            f"<div class='card'>{title}"
            + svg(width, height, "".join(body), f"losses of heads of {words}")
            + f"<p class='foot'>{plural(len(cell), 'trial')}</p></div>"
        )
    return (
        head
        + "<p class='cap key'><i class='k test'></i>test loss <i class='k train'></i>"
        "train loss <i class='k gap'></i>the gap between them: what the head knows of "
        "its training images and not of new ones</p>"
        f"<p class='foot'>bold: the median of the panel, where {MIN_MEDIAN} or more "
        "trials reach that epoch · faint: every trial · a curve above the axis runs "
        "along its top</p>"
        + "<div class='sizes'>" + "".join(panels_) + "</div>"
    )  # fmt: skip


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


def gain_line(study: Study) -> str:
    """The winner against its own baseline: both scores, then the gain in points and
    its standard error, so a gain is never read without what it compares with."""
    b, f = study.base, study.final
    if study.finished == 0 and f is None:
        return "<p class='gain unmeasured'>No finished trial yet.</p>"
    scores = (
        f"<span class='vs'>baseline {se_text(b.score, b['std'], b['n'])} → winner "
        f"{se_text(f.score, f['std'], f['n'])}</span>"
        if b is not None and f is not None
        else ""
    )
    return f"<p class='gain'>{scores}{html.escape(gain_words(*study.gain()))}</p>"


def expects_line(study: Study) -> str:
    if study.expects:
        return f"<p class='expects'><b>expected</b> {html.escape(study.expects)}</p>"
    where = (
        "the handed-out study states no expectation"
        if study.handed_out
        else "EXPECTS in study.py"
    )
    return f"<p class='expects open'><b>expected</b> not said: {where}</p>"


# The button beside an idea's title that edits it; shown only where a title is saved.
PEN = (
    "<button type='button' class='pen' aria-label='edit the title'>"
    "<svg viewBox='0 0 16 16' width='15' height='15' aria-hidden='true'>"
    "<path d='M11.2 2.3l2.5 2.5-7.9 7.9-3.2.7.7-3.2z M9.8 3.7l2.5 2.5' fill='none' "
    "stroke='currentColor' stroke-width='1.5' stroke-linejoin='round'/></svg></button>"
)


def idea_detail(study: Study, known: Mapping[str, Study], parent: Study | None) -> str:
    """What an idea of the graph opens to: the idea in the student's words, what they
    expected, the gain over its own baseline, and the way to its study and to its title
    in page.toml."""
    away = [p for p in study.parents if p not in known]
    builds = (
        f"<p class='builds'>builds on {html.escape(', '.join(away))}, not on this page</p>"
        if away and parent is None
        else ""
    )
    return (
        f"<div class='idea-detail' id='idea-{study.slug}' hidden>"
        f"<h3><span class='t'>{html.escape(study.short)}</span>{PEN}</h3>"
        "<p class='words'>"
        f"{html.escape(study.idea or 'the study carries no idea')}</p>"
        f"{expects_line(study)}{parent_line(study, parent)}{gain_line(study)}{builds}"
        f"<p class='to'><a href='#studies/{study.slug}'>open the study</a>"
        "</p></div>"
    )


def on_parent(study: Study, parent: Study | None) -> tuple[float, float] | None:
    """The winner against the winner of the idea it builds on: the difference in points
    and its standard error in points (NaN without spread); None without both winners
    on one dataset."""
    if parent is None or study.final is None or parent.final is None:
        return None
    f, g = study.final, parent.final
    if str(f.dataset) != str(g.dataset):
        return None
    return difference(f, g)


def pm(points: float, se: float) -> str:
    return f"{signed(points)}" + ("" if math.isnan(se) else f" ± {se:.1f}") + " pt"


def parent_line(study: Study, parent: Study | None) -> str:
    """What the winner moved from the winner of the idea it builds on."""
    up = on_parent(study, parent)
    if up is None or parent is None or parent.final is None or study.final is None:
        return ""
    return (
        f"<p class='gain parent'>{pm(*up)} on {html.escape(parent.short)} "
        f"<span class='vs'>{parent.final.score:.4f} → {study.final.score:.4f}</span></p>"
    )


def idea_numbers(study: Study, parent: Study | None = None) -> str:
    """The winner's score, then what it moved from the idea it builds on, and small
    behind it what it moved from its own baseline; an idea without a parent on the page
    shows its gain on its baseline alone."""
    f = study.final
    points = study.gain()[0]
    if f is None:
        return "<span class='nums'>no winner yet</span>"
    base = "" if math.isnan(points) else f"{signed(points)} pt on baseline"
    up = on_parent(study, parent)
    if up is None:
        main = f" · <b>{base}</b>" if base else ""
        return f"<span class='nums'>{f.score:.3f}{main}</span>"
    small = f" <small>{base}</small>" if base else ""
    return (
        f"<span class='nums'>{f.score:.3f} · <b>{pm(*up)} on parent</b>{small}</span>"
    )


IDEA_ROW, IDEA_STEP, IDEA_R = 40, 26, 7  # a row of the graph, a step in, a circle


def idea_tree(studies: Sequence[Study]) -> str:
    """The ideas as a graph: a circle per idea, under and right of the idea it builds on
    (the first of its parents on this page), joined by a line. Beside each its title, the
    winner's score and its gain over its own baseline; a click opens what it expected
    and found. No idea is coloured by its outcome: one that found nothing can teach as
    much as one that rose."""
    known = {s.idea_name: s for s in studies if s.idea_name}
    parent = {
        s.slug: next(
            (known[p] for p in s.parents if p in known and known[p] is not s), None
        )
        for s in studies
    }
    children: dict[str | None, list[Study]] = {}
    for s in studies:
        up = parent[s.slug]
        children.setdefault(None if up is None else up.slug, []).append(s)
    rows: list[tuple[Study, int]] = []

    def walk(of: str | None, depth: int, seen: frozenset[str]) -> None:
        for s in children.get(of, []):
            if s.slug not in seen:
                rows.append((s, depth))
                walk(s.slug, depth + 1, seen | {s.slug})

    walk(None, 0, frozenset())
    at = {
        s.slug: (IDEA_R + 4 + d * IDEA_STEP, i * IDEA_ROW + IDEA_ROW / 2)
        for i, (s, d) in enumerate(rows)
    }
    lines, dots, nodes = [], [], []
    for s, _ in rows:
        x, y = at[s.slug]
        up = parent[s.slug]
        if up is not None and up.slug in at:
            px, py = at[up.slug]
            lines.append(
                f"<path d='M{px},{py + IDEA_R} V{y - 9} Q{px},{y} {px + 9},{y} "
                f"H{x - IDEA_R}' class='edge'/>"
            )
        dots.append(
            f"<circle cx='{x}' cy='{y}' r='{IDEA_R}' class='dot' "
            f"data-study='{s.slug}'/>"
        )
        nodes.append(
            f"<button type='button' class='node' data-study='{s.slug}' "
            f"data-parent='{up.slug if up is not None else ''}' aria-expanded='false' "
            f"aria-controls='idea-{s.slug}' style='padding-left:{x + IDEA_R + 12}px'>"
            f"<span class='title'>{html.escape(s.short)}</span>{idea_numbers(s, up)}</button>"
        )
    width = max(x for x, _ in at.values()) + IDEA_R + 4
    height = len(rows) * IDEA_ROW
    graph = (
        f"<div class='graph' style='height:{height}px'>"
        f"<svg class='edges' width='{width}' height='{height}' aria-hidden='true' "
        f"xmlns='http://www.w3.org/2000/svg'>{''.join(lines)}{''.join(dots)}</svg>"
        f"{''.join(nodes)}</div>"
    )
    legend = (
        "<p class='foot key'>the numbers: the winner's score, what it moved from the idea "
        "it builds on (± its standard error), and small what it moved from its own "
        "baseline, in points</p>"
    )
    details = "".join(idea_detail(s, known, parent[s.slug]) for s, _ in rows)
    return (
        f"<div class='ideas'>{graph}<div class='idea-panel'>"
        "<p class='hint'>Click an idea for what it expected and what it found.</p>"
        f"{details}</div></div>{legend}"
    )


def panels(study: Study) -> str:
    """The three panels of one study: what it built, what it searched, what it found;
    each one the student can star."""
    return (
        "<div class='three'>"
        + pinnable(f"architecture-{study.slug}", f"{study.short} · architecture",
                   architecture_panel(study), study.key, "panel", "Architecture")
        + pinnable(f"space-{study.slug}", f"{study.short} · search space",
                   fig_space(study), study.key, "panel", "Search space")
        + pinnable(f"results-{study.slug}", f"{study.short} · results",
                   fig_results(study), study.key, "panel", "Results")
        + "</div>"
    )  # fmt: skip


def pinnable(
    fid: str,
    title: str,
    inner: str,
    study: str = "",
    cls: str = "card",
    shown: str = "",
) -> str:
    """A picture the student can star: its title, the star, and the picture, kept under
    `fid` so the Starred tab and the Overview can show it again; `study` the key of the
    study it belongs to, `shown` the title on the page when it is shorter than `title`."""
    return (
        f"<div class='{cls} pinnable' data-figure='{html.escape(fid)}' "
        f"data-study='{html.escape(study)}' data-title='{html.escape(title)}'>"
        f"<div class='pin-head'><h3>{html.escape(shown or title)}"
        "</h3><button type='button' class='star pin' aria-pressed='false'>"
        f"☆ star</button></div><div class='fig'>{inner}</div></div>"
    )


def capacity(studies: Sequence[Study]) -> tuple[str, str]:
    """Both capacity pictures over every trial of the student's studies, once per
    dataset and objective: one study searches a narrow band of head sizes, and only
    the trials of all of them together show where a bigger head stops paying. The
    scaling pictures and the losses by size of the head, each the body of its tab."""
    groups: dict[tuple[str, str], list[Study]] = {}
    for st in studies:
        for d in st.datasets():
            groups.setdefault((d, st.objective), []).append(st)
    if not groups:
        none = missing("No study of this lesson is in the runs.")
        return f"<section><h2>Scaling FLOPs</h2>{none}</section>", ""
    several = len(groups) > 1
    flops, overfit = [], []
    for (d, objective), group in groups.items():
        trials = pd.concat(
            [st.trials_on(d).assign(by_study=st.short) for st in group],
            ignore_index=True,
        )
        done = trials[(trials.status == "COMPLETE") & trials.score.notna()]
        done = (
            done.assign(winner=done.index == done.score.idxmax()) if len(done) else done
        )
        bases = pd.DataFrame(
            [
                b
                for b in (st.base for st in group)
                if b is not None and str(b.dataset) == d
            ]
        )
        base = None if bases.empty else bases.sort_values("start").iloc[-1]
        name = f"{d} · {objective}" if several else None
        on = f" · {d} · {objective}" if several else ""
        key = re.sub(r"[^a-z0-9]+", "-", f"{d}-{objective}".lower()).strip("-")
        flops.append(
            "<div class='scaling'>"
            + "".join(
                pinnable(
                    f"scaling-{by}-{key}",
                    title + on,
                    fig_flops(objective, done, base, name, by),
                )
                for by, title in (
                    ("params", "By parameters: how heavy the head is"),
                    ("flops", "By FLOPs: all the compute of training it"),
                )
            )
            + "</div>"
        )
        overfit.append(
            pinnable(
                f"overfitting-{key}",
                "Training curves by size of the head" + on,
                fig_overfitting(trials, name),
            )
        )
    return (
        (
            "<section><h2>Scaling FLOPs</h2><p class='cap'>Every trial of all my studies, "
            "twice: against how heavy the head is, and against all the compute of "
            "training it. Both count the head only, so a bigger encoder lifts a dot "
            "without moving it right. Where the dots stop rising, more buys no more score."
            f"</p>{''.join(flops)}</section>"
        ),
        (
            "<section><h2>Training curves, by size of the head</h2><p class='cap'>A test "
            "loss that turns up while the train loss keeps falling: from that epoch on, more "
            "training makes the head worse on images it has not seen. Star a picture to "
            f"keep it on the Starred tab.</p>{''.join(overfit)}</section>"
        ),
    )


def glance(studies: Sequence[Study]) -> str:
    """The strip over the ideas: how many studies, and how many trials finished and were
    pruned (stopped early by the search), each a number under its name."""
    cells = [
        ("ink", "Studies", str(len(studies)), ""),
        ("green", "Finished trials", str(sum(s.finished for s in studies)), ""),
        ("amber", "Stopped early (pruned)", str(sum(s.pruned for s in studies)), ""),
    ]
    return (
        "<dl class='glance'>"
        + "".join(
            f"<div><dt><i class='dot-{c}'></i>{html.escape(name)}</dt><dd>{value}</dd>"
            + (f"<span class='under'>{html.escape(under)}</span>" if under else "")
            + "</div>"
            for c, name, value, under in cells
        )
        + "</dl>"
    )


def overview(
    studies: Sequence[Study], own: Mapping[str, str] | None, folder: Path
) -> str:
    tree = idea_tree(studies) if studies else missing(
        "No study of this lesson is in the runs."
    )  # fmt: skip
    return (
        f"<section><h2>My ideas</h2>{glance(studies)}"
        "<p class='cap'>Each idea under the idea it builds on.</p>"
        f"{tree}</section>"
        "<label class='here'><input type='checkbox' id='stars-here'> show what I "
        "starred here</label>"
        "<section id='starred-overview' class='starred-here' hidden><h2>Starred</h2>"
        "<div class='starred-list'></div></section>"
        f"{own_sections(own, folder)}"
    )


def own_sections(own: Mapping[str, str] | None, folder: Path) -> str:
    """What the student wrote: what they did, what surprised them, what they would try
    next. A page of submitted runs has none of it, and says so once."""
    if own is None:
        return (
            "<p class='foot own-note'>What the student wrote about the study (a message, "
            "a picture, what surprised them, what they would try next) stays in their "
            "page.toml and is not part of the submitted runs.</p>"
        )
    return (
        "<section class='own'><h2>What I did</h2>"
        f"{own_text(own, 'message')}{picture(own, folder)}</section>"
        "<section class='own'><h2>What surprised me</h2>"
        f"{own_text(own, 'surprised')}<h2>What I would try next</h2>"
        f"{own_text(own, 'next')}</section>"
    )


# ---- explore: two axes and a colour, chosen by the student ----------------------------

# What every trial carries besides its params, in the order the menus list them.
MEASURED = {
    "n_params": "params",
    "train_seconds": "train_seconds",
    "epochs_run": "epochs",
}
FLOPS = "FLOPs"


def explore_value(value: object) -> float | str | None:
    """A trial's value of one field for the page: a number, a category as text (a
    switch as on or off), or None where the trial has none."""
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool | np.bool_):
        return "on" if value else "off"
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return str(value)


def explore_fields(
    study: Study, done: pd.DataFrame, every: bool = False
) -> dict[str, list[float | str | None]]:
    """The fields the explorer offers for the finished trials `done`, each with every
    trial's value: the objective, the FLOPs of training the head, the searched axes, the
    other params that vary, and what every trial measured. A field with one value, or
    none, says nothing about these trials and is left out unless `every`; the dataset
    never is an axis."""
    cols: dict[str, pd.Series] = {
        study.objective: done.score,
        FLOPS: train_flops(done),
    }
    searched = [f"p_{a}" for a in study.axes]
    params = [c for c in done.columns if str(c).startswith("p_")]
    for c in searched + sorted(set(params) - set(searched)):
        if c in done and c != "p_dataset":
            cols.setdefault(str(c)[2:], done[c])
    for c, name in MEASURED.items():
        if c in done:
            cols.setdefault(name, done[c])
    out = {}
    for name, col in cols.items():
        values = [explore_value(v) for v in col]
        if every or len({v for v in values if v is not None}) > 1:
            out[name] = values
    return out


# Fields that grow by factors, drawn on a log scale.
LOG_FIELDS = {FLOPS, "params"}


def explore_kind(
    name: str, values: Sequence[object], space: Mapping[str, Axis]
) -> tuple[str, bool]:
    """How a field is drawn: `num` on a scale or `cat` as its values side by side. A
    number is on a log scale when the search space draws it on one, or when it grows by
    factors (`LOG_FIELDS`)."""
    present = [v for v in values if v is not None]
    if not all(isinstance(v, float) for v in present):
        return "cat", False
    axis = space.get(name)
    log = axis.log if axis is not None and axis.kind == "range" else name in LOG_FIELDS
    return "num", log and min(v for v in present if isinstance(v, float)) > 0


def explore_defaults(
    studies: Sequence[Study], names: Sequence[str], objective: str
) -> dict[str, str]:
    """The view the explorer opens on: the objective against the FLOPs, coloured by the
    encoder where the trials used more than one (a bigger encoder lifts the score without
    moving the head's FLOPs), else by the study where there are several, else by the
    depth of the model, else by the first searched axis."""
    axes = [a for st in studies for a in st.axes]
    x = (
        FLOPS
        if FLOPS in names
        else next((a for a in axes if a in names), names[min(1, len(names) - 1)])
    )
    depth = ("n_hidden_layers", "depth", "layers")
    first = ("encoder", "study") if len(studies) > 1 else ("encoder",)
    c = next((a for a in (*first, *depth, *axes) if a in names and a != x), "")
    return {"x": x, "y": objective, "c": c}


STUDY = "study"


def explorer(
    group: Sequence[Study],
    dataset: str,
    objective: str,
    b: pd.Series | None,
    head: str,
) -> str:
    """The explorer of the finished trials of every study in `group` on one dataset, all
    scored by `objective`: the student picks what goes along x, along y and in colour,
    which studies, and filters on any field (a number above or below a value, a category
    that is or is not one value, a metric a trial has); stars a view to keep it, with its
    filters; and sees the views they starred beside it. The script `EXPLORE` draws it
    from the data here."""
    parts = []
    for st in group:
        t = st.trials_on(dataset)
        done = t[(t.status == "COMPLETE") & t.score.notna()]
        if not done.empty:
            parts.append((st, done, explore_fields(st, done, every=True)))
    if not parts:
        return head + missing("No finished trial yet.")
    names = [objective, FLOPS, STUDY]
    for _, _, fields in parts:
        names += [n for n in fields if n not in names]
    columns: dict[str, list[float | str | None]] = {n: [] for n in names}
    keys: list[str] = []
    for st, done, fields in parts:
        size = len(done)
        keys += [st.key] * size
        for n in names:
            if n == STUDY:
                columns[n] += [st.short] * size
            else:
                columns[n] += fields.get(n, [None] * size)
    columns = {
        n: v
        for n, v in columns.items()
        if n == objective or len({x for x in v if x is not None}) > 1
    }
    if len(columns) < 2:
        return head + missing(
            "These trials differ in too little to set two against each other."
        )
    names = list(columns)
    space = {a.name: a for st in group for a in parse_space(st.space)}
    scores = [v if isinstance(v, float) else -math.inf for v in columns[objective]]
    best = max(range(len(scores)), key=scores.__getitem__) if scores else None
    slug = re.sub(r"[^a-z0-9]+", "-", f"{dataset}-{objective}".lower()).strip("-")
    data = {
        "id": slug,
        "dataset": dataset,
        "objective": objective,
        "fields": [
            {"name": n, "kind": k, "log": log}
            for n in names
            for k, log in [
                ("cat", False) if n == STUDY else explore_kind(n, columns[n], space)
            ]
        ],
        "rows": [list(r) for r in zip(*columns.values(), strict=True)],
        "keys": keys,
        "winner": best,
        "base": None
        if b is None
        else [float(b.score), 0.0 if math.isnan(number(b["std"])) else float(b["std"])],
        "defaults": explore_defaults([st for st, _, _ in parts], names, objective),
    }
    blob = json.dumps(data).replace("</", "<\\/")

    searched = {a for st in group for a in st.axes}
    measured = {objective, FLOPS, *MEASURED.values()}

    def kind(n: str) -> str:
        if n in measured:
            return "measured"
        if n == STUDY:
            return "study"
        if n in searched:
            return "searched"
        return "fixed" if n in space else "other params"

    order = ("measured", "searched", "study", "fixed", "other params")
    grouped = "".join(
        f"<optgroup label='{g}'>"
        + "".join(f"<option>{html.escape(n)}</option>" for n in names if kind(n) == g)
        + "</optgroup>"
        for g in order
        if any(kind(n) == g for n in names)
    )

    def menu(which: str, words: str, none: bool = False) -> str:
        options = ("<option value=''>none</option>" if none else "") + grouped
        return f"<label>{words} <select data-axis='{which}'>{options}</select></label>"

    studies = "".join(
        f"<option value='{html.escape(st.key)}'>{html.escape(st.short)}</option>"
        for st, _, _ in parts
    )
    fields = grouped
    return (
        head
        + f"<script type='application/json' class='explore-data'>{blob}</script>"
        f"<div class='explore' data-set='{slug}'>"
        "<div class='live'><div class='controls'>"
        "<label>studies <select class='which'><option value=''>all</option>"
        f"{studies}</select></label>"
        + menu("x", "along x") + menu("y", "along y") + menu("c", "colour", none=True)
        + "</div><div class='filters'><span class='lbl'>only trials where</span>"
        f"<select class='f-field' aria-label='field to filter on'>{fields}</select>"
        "<select class='f-op' aria-label='how to compare'></select>"
        "<input class='f-value' aria-label='value' size='8'>"
        "<select class='f-choice' aria-label='value' hidden></select>"
        "<button type='button' class='f-add'>add filter</button>"
        "<button type='button' class='f-reset'>reset view</button></div>"
        "<div class='plot-head'><span class='chips'></span><span class='count'></span>"
        "<button type='button' class='star big' aria-pressed='false'>☆ star this view"
        "</button></div>"
        "<div class='plot'></div><div class='legend'></div>"
        "<p class='foot'>one dot per finished trial, the best ringed · hover a dot for "
        "its values · ▼ on the floor: far below the others · a dot without a value for "
        "an axis is left out · a number may end in k, M or G</p></div>"
        "<div class='starred'><h3>Starred on this dataset</h3><div class='stars'></div>"
        "<p class='foot save-note'></p></div></div>"
    )  # fmt: skip


def explore_tab(studies: Sequence[Study]) -> str:
    """One explorer per dataset and objective over the trials of every study: the
    questions no single study answers, and never two datasets on one axis."""
    groups: dict[tuple[str, str], list[Study]] = {}
    for st in studies:
        for d in st.datasets():
            groups.setdefault((d, st.objective), []).append(st)
    if not groups:
        return f"<section><h2>Explore</h2>{missing('No study of this lesson is in the runs.')}</section>"
    several = len(groups) > 1
    out = []
    for (d, objective), group in groups.items():
        bases = [
            b
            for b in (st.base for st in group)
            if b is not None and str(b.dataset) == d
        ]
        base = max(bases, key=lambda b: float(b.start)) if bases else None
        head = (
            f"<p class='ds'>on {html.escape(d)} · {html.escape(objective)}</p>"
            if several
            else ""
        )
        out.append(explorer(group, d, objective, base, head))
    return (
        "<section><h2>Explore the trials</h2><p class='cap'>Every finished trial of all "
        "my studies. Set two things against each other and a third in colour, pick the "
        "studies, and filter: only the heads above 300k parameters, only the trials "
        "above 0.82, only those that logged recall. What changes together, and where "
        "does it stop paying? Star a view to keep it, with its filters, on the Starred "
        f"tab.</p>{''.join(out)}</section>"
    )


def study_view(study: Study) -> str:
    """One study on the Studies tab. When the search drew more than one dataset, every
    picture is drawn once per dataset, and no picture compares scores of two datasets."""
    trials = study.trials
    datasets = study.datasets()
    several = len(datasets) > 1
    on = {d: study.trials_on(d) for d in datasets}
    base = study.base
    drew = (
        f"<p class='cap'>The search drew the dataset ({html.escape(', '.join(datasets))}): "
        "every picture is drawn per dataset, and the best trial filled is the best on "
        "that dataset. Scores of two datasets are never compared.</p>"
        if several
        else ""
    )
    curves = "".join(
        fig_curves(
            study,
            on[d],
            base if base is not None and str(base.dataset) == d else None,
            d if several else None,
        )
        for d in datasets
    )
    counts = f"{plural(study.finished, 'finished trial')}, {plural(study.pruned, 'pruned trial')}"
    meta = " · ".join(
        html.escape(x)
        for x in (
            study.label,
            study.idea_name or study.key,
            f"{study.objective} on {study.on_text()}",
            counts,
            budget_text(study.budget, len(trials)),
        )
    )
    return (
        f"<div class='study' id='view-{study.slug}' data-key='{html.escape(study.key)}' hidden>"
        f"<section class='lead'><h2>{html.escape(study.short)}</h2>"
        f"<p class='words'>{html.escape(study.idea or 'the study carries no idea')}</p>"
        f"{expects_line(study)}{gain_line(study)}<p class='meta'>{meta}</p>"
        f"<p class='to'><a href='#explore' class='to-explore' "
        f"data-study='{html.escape(study.key)}'>explore the trials of this study</a></p>"
        "</section>"
        f"<section>{panels(study)}</section>"
        f"{drew}"
        "<div class='two'>"
        "<section><h2>Learning curves</h2>"
        + pinnable(
            f"curves-{study.slug}",
            f"{study.short} · learning curves",
            curves,
            study.key,
            "plain",
            " ",
        )
        + "</section>"
        f"<section><h2>The winner's other metrics</h2>{metrics_table(study)}</section>"
        "</div></div>"
    )


def studies_tab(studies: Sequence[Study]) -> str:
    """Every study, one at a time: a switch of their short names, then the study picked."""
    if not studies:
        return missing("No study of this lesson is in the runs.")
    links = "".join(
        f"<a href='#studies/{s.slug}' data-study='{s.slug}' title='{html.escape(s.label)}'>"
        f"{html.escape(s.short)}</a>"
        for s in studies
    )
    return f"<nav class='switch' aria-label='studies'>{links}</nav>" + "".join(
        study_view(s) for s in studies
    )


ATLAS_SECONDS = 60


def design_route(folder: Path) -> str | None:
    """The atlas route of the design in `folder`: the `forward` of the class that
    `DESIGN = Design(<class>, ...)` names in `designs.py`, as atlas names it from that
    folder (`exercises.designs.Head.forward`); None without one."""
    script = folder / "designs.py"
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DESIGN" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
        ):
            return f"{folder.name}.designs.{node.value.args[0].id}.forward"
    return None


def atlas_page(command: str, folder: Path) -> tuple[bytes | None, str]:
    """The page `command view` writes of `folder`, or None and why it gave none. Atlas
    reads the folder with what it imports, the course's `hpo` among it, so the page opens
    on the route of the design (`design_route`): its stages and the shape on every arrow."""
    route = design_route(folder)
    view = ["--view", json.dumps({"route": route})] if route else []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "atlas.html"
        try:
            ran = subprocess.run(
                [command, "view", str(folder), *view, "--out", str(out), "--no-open"],
                capture_output=True,
                text=True,
                timeout=ATLAS_SECONDS,
                check=False,
                encoding="utf-8",
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
                "out. Zoom in with + at the top right. 'error types' folds the errors the "
                "code can raise into one box; it is not a fault in the design.</p><iframe id='atlas' title='the design in atlas'></iframe>"
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
    css = (PRESENTATIONS / "course-ui.css").read_text(encoding="utf-8")
    head, sep, _ = css.partition("\nhtml{")
    return head if sep else css


CSS = """
:root{--slot-ink:#a8310f}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--slot-ink:var(--course-orange)}}
:root[data-theme="dark"]{--slot-ink:var(--course-orange)}
*{box-sizing:border-box}
body{margin:0;background:var(--course-paper);color:var(--course-ink);
  font-family:Outfit,Arial,sans-serif;font-size:18px;line-height:1.45}
::selection{background:var(--course-blue);color:var(--course-paper)}
:focus-visible{outline:3px solid var(--course-blue);outline-offset:2px;border-radius:2px}
.app{display:grid;grid-template-columns:92px minmax(0,1fr);min-height:100vh}
main{min-width:0}
.content{max-width:1320px;margin:0 auto;padding:20px 24px 48px}
.bar{display:flex;gap:14px;align-items:center;padding:16px 24px;border-bottom:1px solid var(--course-rule);
  position:sticky;top:0;z-index:2;background:color-mix(in srgb,var(--course-paper) 92%,transparent);
  backdrop-filter:blur(8px)}
.bar h1{flex:1 1 auto}
h1{font-size:28px;font-weight:600;margin:0;line-height:1.15;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:19px;font-weight:600;margin:0 0 8px;color:var(--course-ink);text-wrap:balance}
section>h2:first-child{margin:-2px -20px 14px;padding:0 20px 14px;
  border-bottom:1px solid color-mix(in srgb,var(--course-violet) 35%,var(--course-rule))}
section>h2:first-child::before{content:'';display:inline-block;width:10px;height:10px;margin-right:10px;
  background:var(--course-violet);border-radius:1px;vertical-align:1px}
.pill{display:inline-flex;align-items:center;gap:8px;flex:none;font-size:14px;color:var(--course-muted);
  border:1px solid var(--course-rule);background:var(--course-surface);border-radius:3px;padding:5px 11px}
.ic{width:18px;height:18px;flex:none}
.pill .ic{width:15px;height:15px}
h3{font-size:19px;font-weight:500;margin:0 0 8px}
h4{font-size:15px;font-weight:500;margin:0;font-family:'JetBrains Mono',monospace}
p{margin:4px 0;text-wrap:pretty}
code,.name,.metrics td{font-family:'JetBrains Mono',monospace;font-size:14px;color:var(--course-ink);
  overflow-wrap:anywhere}
.sub,.cap,.foot,.meta,.ylab{color:var(--course-muted);font-size:15px;overflow-wrap:anywhere}
.cap{max-width:68ch;margin-bottom:10px}
.foot{font-size:14px;margin-top:4px;max-width:80ch}
.ylab{margin:8px 0 0;font-size:14px}
.ds{margin:10px 0 0;font-size:15px;font-weight:500;color:var(--course-ink)}
footer.sub{margin:28px 0 0;max-width:90ch}
section{background:var(--course-surface);border:1px solid var(--course-rule);border-radius:8px;
  border-top:3px solid var(--course-violet);
  padding:18px 20px;margin:18px 0;min-width:0}
.rail{position:sticky;top:0;height:100vh;display:flex;flex-direction:column;align-items:center;gap:6px;
  padding:16px 8px;border-right:1px solid var(--course-rule);background:var(--course-surface)}
.badge{display:flex;align-items:center;justify-content:center;width:42px;height:42px;margin-bottom:14px;
  border-radius:4px;border:1px solid var(--course-rule);background:var(--course-paper);
  font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:var(--course-green)}
.rail button{position:relative;display:flex;flex-direction:column;align-items:center;gap:4px;width:76px;
  font:inherit;font-size:12.5px;background:none;border:1px solid transparent;border-radius:4px;
  color:var(--course-muted);padding:9px 4px 7px;cursor:pointer}
.rail button:hover{color:var(--course-ink);background:var(--course-paper)}
.rail button[aria-selected=true]{color:var(--course-green);background:var(--course-paper);
  border-color:color-mix(in srgb,var(--course-green) 40%,transparent)}
.rail .count{position:absolute;top:4px;right:12px;min-width:17px;height:17px;padding:0 4px;
  border-radius:3px;font-size:11px;line-height:17px;font-weight:600;
  background:var(--course-orange);color:var(--course-paper)}
.rail .count:empty{display:none}
.theme{font:inherit;font-size:14px;background:var(--course-surface);color:var(--course-muted);flex:none;
  border:1px solid var(--course-rule);border-radius:3px;padding:5px 11px;cursor:pointer}
.theme:hover{color:var(--course-ink)}
.glance{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 16px;
  padding:0 0 16px;border-bottom:1px solid color-mix(in srgb,var(--course-violet) 35%,var(--course-rule))}
.glance div{min-width:0}
.glance div:nth-child(1) .ic{color:var(--course-violet)}
.glance div:nth-child(2) .ic{color:var(--course-green)}
.glance div:nth-child(3) .ic{color:var(--course-orange)}
.glance dt{display:flex;align-items:center;gap:7px;font-size:14px;color:var(--course-muted)}
.glance dt i{width:7px;height:7px;border-radius:50%;flex:none}
.glance dd{margin:2px 0 0;font-size:30px;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.glance .under{display:block;font-size:13px;color:var(--course-muted);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.dot-ink{background:var(--course-muted)}
.dot-green{background:var(--course-green)}
.dot-amber{background:var(--course-orange)}
.dot-blue{background:var(--course-blue)}
.side .filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 2px;font-size:15px}
.filters .lbl{color:var(--course-muted);font-size:15px}
.filters select,.filters input{font:inherit;font-size:15px;background:var(--course-surface);color:var(--course-ink);
  border:1px solid var(--course-rule);border-radius:2px;padding:3px 6px}
.filters input{width:11ch}
.filters input::placeholder{color:var(--course-muted);opacity:.8;font-style:italic}
.f-reset{font:inherit;font-size:15px;background:none;color:var(--course-muted);border:0;text-decoration:underline;text-underline-offset:3px;cursor:pointer}
.f-add{font:inherit;font-size:15px;background:var(--course-surface);color:var(--course-ink);
  border:1px solid var(--course-rule);border-radius:2px;padding:3px 9px;cursor:pointer}
.chips{display:contents}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 4px 3px 9px;font-family:'JetBrains Mono',monospace;
  font-size:13px;border:1px solid var(--course-green);color:var(--course-ink);border-radius:2px;
  background:color-mix(in srgb,var(--course-green) 18%,transparent)}
.plot-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 4px}
.plot-head .count{color:var(--course-muted);font-size:14px}
.star.big{margin-left:auto;background:var(--course-orange);color:#1a1405;border-color:var(--course-orange);
  font-weight:500;padding:6px 14px}
.star.big[aria-pressed=true]{background:transparent;color:var(--course-orange)}
.filters select[hidden],.filters input[hidden]{display:none}
.chip button{font:inherit;background:none;border:0;color:inherit;cursor:pointer;padding:0 4px}
.slot{border-radius:3px;padding:10px 14px;margin:8px 0}
.slot.filled{background:var(--course-paper);border:1.5px solid var(--course-green);font-size:21px}
.slot.open{border:2px dashed var(--slot-ink);color:var(--slot-ink)}
.slot.open p{font-style:italic}
.slot.absent{border:2px dotted var(--course-rule);color:var(--course-muted)}
.slot .tag{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--course-muted)}
.slot.picture{text-align:center}
.slot.picture svg,.slot.picture img{max-width:100%;max-height:360px;height:auto}
.warn{color:var(--slot-ink);font-size:15px}
.missing{color:var(--course-muted);font-size:15px;background:var(--course-paper);border-radius:2px;
  padding:8px 12px;margin:6px 0;overflow-wrap:anywhere}
.ideas{display:grid;grid-template-columns:minmax(0,auto) minmax(240px,1fr);gap:12px 28px;align-items:start}
.graph{position:relative;min-width:0}
.graph svg.edges{position:absolute;left:0;top:0;width:auto;overflow:visible}
svg .edge{fill:none;stroke:var(--course-rule);stroke-width:2}
svg .dot{stroke-width:2;fill:var(--course-paper);stroke:var(--course-muted)}
svg .dot.picked{stroke:var(--course-ink);stroke-width:3}
.graph svg .dot{fill:var(--course-blue);stroke:var(--course-blue);stroke-width:2}
.graph svg .dot.picked{stroke:var(--course-ink);stroke-width:3}
.node{position:relative;display:flex;align-items:baseline;gap:12px;width:100%;height:40px;font:inherit;
  font-size:17px;color:var(--course-ink);background:none;border:0;border-radius:2px;
  padding:0 10px 0 0;text-align:left;cursor:pointer;white-space:nowrap;line-height:40px}
.node .title{min-width:0;overflow:hidden;text-overflow:ellipsis}
.node:hover .title{text-decoration:underline;text-underline-offset:3px}
.node[aria-expanded=true] .title{font-weight:500}
.nums{flex:none;margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:14px;color:var(--course-muted);
  font-variant-numeric:tabular-nums}
.nums b{font-weight:500;color:var(--course-ink)}
.nums small{font-size:12px;margin-left:6px}
.idea-panel{background:var(--course-paper);border-radius:3px;padding:12px 16px;min-height:80px}
.idea-panel .hint{color:var(--course-muted);font-size:15px}
.idea-panel h3{font-size:19px;margin:0 0 4px;display:flex;align-items:center;gap:6px}
.idea-panel h3[hidden]{display:none}
.pen{display:none;align-items:center;justify-content:center;width:28px;height:28px;padding:0;
  background:none;border:0;border-radius:2px;color:var(--course-muted);cursor:pointer}
.titles .pen{display:inline-flex}
.pen:hover{color:var(--course-ink);background:var(--course-surface)}
.title-edit{font:inherit;font-size:19px;font-weight:500;width:100%;max-width:32ch;margin:0 0 4px;
  padding:2px 8px;color:var(--course-ink);background:var(--course-surface);
  border:1.5px solid var(--course-blue);border-radius:2px}
.idea-detail .words{font-size:17px}
.idea-detail .expects,.idea-detail .gain{font-size:15px}
.key svg{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin:0 2px 0 6px}
.to{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;margin-top:6px}
.to a{color:var(--course-blue);text-underline-offset:3px}
.to .tag{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--course-muted);overflow-wrap:anywhere}
.words{font-size:19px}
.lead .words{font-size:22px;max-width:65ch}
.expects{font-size:16px;max-width:75ch}
.expects b,.slot b{font-weight:500;color:var(--course-muted)}
.expects.open{color:var(--slot-ink);font-style:italic}
.builds{font-size:15px;color:var(--course-muted);overflow-wrap:anywhere}
.gain{font-size:16px;font-weight:500;color:var(--course-muted)}
.gain .vs{display:block;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:400;color:var(--course-ink)}
.lead .meta{margin-top:8px}
.switch{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 0}
.switch a{font-size:17px;color:var(--course-ink);text-decoration:none;border:1.5px solid var(--course-rule);
  border-radius:2px;padding:5px 14px;background:var(--course-surface)}
.switch a:hover{border-color:var(--course-blue)}
.switch a[aria-current]{background:var(--course-ink);color:var(--course-paper);border-color:var(--course-ink)}
.study[hidden]{display:none}
.group+.group{margin-top:22px}
.group .words{margin-bottom:8px;color:var(--course-muted);font-size:17px;max-width:75ch}
.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.panel{background:var(--course-paper);border-radius:3px;padding:12px;min-width:0}
.panel .missing,.card .missing{background:var(--course-surface)}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr));gap:12px}
.card{background:var(--course-paper);border-radius:3px;padding:10px;min-width:0}
svg{width:100%;height:auto;display:block;overflow:visible}
svg .t{font-size:14px;fill:var(--course-ink)}
svg .s{font-size:13px;fill:var(--course-ink)}
svg .m{fill:var(--course-muted)}
svg .b{font-weight:500}
svg .on{fill:var(--course-paper)}
svg .track{stroke:var(--course-muted);stroke-width:1}
svg .err{stroke:var(--course-ink);stroke-width:2}
svg .win{fill:none;stroke:var(--course-ink);stroke-width:2.4}
svg .base{fill:var(--course-paper);stroke:var(--course-orange);stroke-width:2.5}
svg .dot{fill:none;stroke:var(--course-muted);stroke-width:1.2}
svg .seg{fill:var(--course-paper);stroke:var(--course-rule)}
svg .seg.on{fill:var(--course-green);stroke:var(--course-green)}
svg .fixed{fill:var(--course-paper);stroke:var(--course-rule);stroke-dasharray:3 3}
svg .band{fill:var(--course-orange-wash);opacity:.7}
svg .baseline{stroke:var(--course-orange);stroke-width:1.5;stroke-dasharray:2 3}
svg .faint{fill:none;stroke:var(--course-green);stroke-width:1.1;stroke-opacity:.55}
svg .wincurve{fill:none;stroke:var(--course-ink);stroke-width:2.6}
svg .basecurve{fill:none;stroke:var(--course-orange);stroke-width:2;stroke-dasharray:2 3}
svg .loss{fill:none;stroke-width:1;stroke-opacity:.18}
svg .loss.test{stroke:var(--course-blue)}
svg .loss.train{stroke:var(--course-orange);stroke-dasharray:3 3}
svg .loss.median{stroke-width:2.8;stroke-opacity:1;stroke-dasharray:none}
svg .gap{fill:var(--course-blue);fill-opacity:.16;stroke:none}
.key{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px}
.key .k{display:inline-block;width:18px;height:3px;margin-left:6px}
.key .k:first-child{margin-left:0}
.key .k.test{background:var(--course-blue)}
.key .k.train{background:var(--course-orange)}
:root[data-theme="light"] svg .loss.train{stroke:var(--course-green)}
:root[data-theme="light"] .key .k.train{background:var(--course-green)}
.key .k.gap{height:10px;background:color-mix(in srgb,var(--course-blue) 22%,transparent)}
.sizes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:8px}
.sizes .card p{margin:0}
.explore{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;margin-top:8px}
.explore .live,.explore .starred{background:var(--course-paper);border-radius:3px;padding:12px;min-width:0}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:15px}
.controls select{font:inherit;font-size:15px;background:var(--course-surface);color:var(--course-ink);
  border:1px solid var(--course-rule);border-radius:2px;padding:3px 6px;max-width:190px}
.star,.card button{font:inherit;font-size:15px;background:var(--course-surface);color:var(--course-ink);
  border:1px solid var(--course-rule);border-radius:2px;padding:3px 10px;cursor:pointer}
.star[aria-pressed=true]{border-color:var(--slot-ink);color:var(--slot-ink)}
.stars{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:10px}
figure.card{margin:0;background:var(--course-surface)}
.starred-list figure.card{background:var(--course-paper)}
.pinnable{padding:12px}
.pin-head{display:flex;gap:10px;align-items:baseline;justify-content:space-between}
.pin-head h3{font-size:17px}
.panel>h3,.pinnable h3{padding-left:9px;border-left:3px solid var(--course-violet)}
.pin{flex:none}
.cards.pictures{margin-bottom:12px;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr))}
label.here{display:flex;gap:8px;align-items:center;margin-top:14px;font-size:15px;color:var(--course-muted)}
.starred-list .fig svg,.starred-list figure svg{max-width:100%;height:auto}
figure.card figcaption{font-size:15px;color:var(--course-muted);display:flex;flex-wrap:wrap;gap:6px;align-items:center}
figure.card .drop{padding:0 9px}
.legend{font-size:14px;color:var(--course-muted);margin-top:4px;display:flex;flex-wrap:wrap;gap:4px 8px;align-items:center}
.legend b{font-weight:500;color:var(--course-ink)}
.legend .sw{display:inline-block;width:13px;height:13px;margin-right:3px;vertical-align:-2px;overflow:visible}
.legend .ramp{display:inline-block;width:90px;height:9px;border-radius:2px}
svg .grid{stroke:var(--course-rule);stroke-width:.6}
svg .pt{stroke:var(--course-paper);stroke-width:.6;fill-opacity:.9}
svg .pt.best{stroke:var(--course-ink);stroke-width:2}
.scaling{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.scaling h3{font-size:17px}
.stages{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:14px}
.stages li{border:1.5px solid var(--course-rule);border-radius:2px;padding:5px 9px;display:flex;
  flex-wrap:wrap;gap:2px 10px;align-items:baseline;position:relative}
.stages li+li::before{content:'↓';position:absolute;top:-16px;left:12px;font-size:13px;line-height:1;color:var(--course-muted)}
.stages li b{font-weight:500;font-family:'JetBrains Mono',monospace;font-size:14px}
.stages .in{border-style:none;padding-left:0}
.stages .optional{border-style:dashed;border-color:var(--course-muted)}
.stages .optional.off{border-color:var(--course-rule)}
.stages .optional.off code{color:var(--course-muted)}
.stages .opt{font-size:13px;color:var(--course-muted);width:100%}
.gain+ul.plain{margin:4px 0}
ul.plain{list-style:none;padding:0;font-size:15px}
ul.plain b{font-weight:500}
table.metrics{border-collapse:collapse;font-size:15px;width:100%;font-variant-numeric:tabular-nums}
table.metrics th{text-align:left;font-weight:400;padding:3px 8px 3px 0;overflow-wrap:anywhere}
table.metrics td{padding:3px 6px;text-align:right}
table.metrics thead th{color:var(--course-muted);font-size:14px}
table.per-class tbody tr:nth-child(odd){background:var(--course-paper)}
details{margin-top:10px}
summary{cursor:pointer;color:var(--course-muted);font-size:15px}
.design iframe{width:100%;height:78vh;border:1px solid var(--course-rule);border-radius:3px;
  background:var(--course-surface)}
[role=tabpanel][hidden]{display:none}
@media (max-width:1100px){
  .row{grid-template-columns:minmax(0,1fr)}
}
@media (max-width:860px){
  .app{grid-template-columns:minmax(0,1fr)}
  .rail{position:sticky;top:0;z-index:3;height:auto;flex-direction:row;justify-content:space-around;
    padding:6px 8px;border-right:0;border-bottom:1px solid var(--course-rule)}
  .badge{display:none}
  .rail button{width:auto;flex:1 1 0}
  .bar{position:static;flex-wrap:wrap;padding:14px 16px}
  .content{padding:14px 12px 40px}
  .glance{grid-template-columns:repeat(2,minmax(0,1fr))}
  section>h2:first-child{margin:-2px -12px 12px;padding:0 12px 12px}
  .three,.two,.sizes,.explore,.scaling,.ideas{grid-template-columns:minmax(0,1fr)}
  section{padding:14px 12px}
  body{font-size:17px}
  h1{font-size:26px}
  h2{font-size:22px}
  .lead .words{font-size:20px}
  .node{font-size:16px;gap:8px}
  .nums{font-size:12.5px}
}
/* dark by default: near black, red the accent, yellow the baseline, teal what rose,
   violet the structure (sections, headings, rules) */
:root:not([data-theme="light"]){
  --course-paper:#0f0f0f;--course-surface:#181818;--course-ink:#f2f2f2;--course-muted:#a3a3a3;
  --course-rule:#3a2c2c;--course-blue:#ef4444;--course-blue-wash:#3a1414;--course-orange:#e9c46a;
  --course-orange-wash:#3a3018;--course-green:#2fb3a3;--course-violet:#a78bfa;--slot-ink:#e9c46a;
  color-scheme:dark;
}
/* light: one warm family, sand under warm off-white, no pure white; red-orange the accent,
   ochre the baseline, teal what rose, plum the structure */
:root[data-theme="light"]{
  --course-paper:#f4ebe5;--course-surface:#fbf6f2;--course-ink:#2a1a14;--course-muted:#7b6158;
  --course-rule:#e6d3c9;--course-blue:#c83600;--course-blue-wash:#fbe3d8;--course-orange:#a8790f;
  --course-orange-wash:#f6ead0;--course-green:#1f7a70;--course-violet:#6d3f8f;--slot-ink:#a8790f;
  color-scheme:light;
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
  const dark = () => root.dataset.theme !== 'light';
  const label = () => { button.textContent = dark() ? 'light theme' : 'dark theme'; };
  button.addEventListener('click', function () {
    root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem(key, JSON.stringify(root.dataset.theme)); } catch (e) {}
    label();
  });
  media.addEventListener('change', label);
  label();

  const tabs = Array.from(document.querySelectorAll('.tabs button'));
  const links = Array.from(document.querySelectorAll('.switch a'));
  function atlas() {
    const frame = document.getElementById('atlas');
    const data = document.getElementById('atlas-page');
    if (!frame || !data || frame.dataset.loaded) return;
    const bytes = Uint8Array.from(atob(data.textContent.trim()), function (c) { return c.charCodeAt(0); });
    frame.srcdoc = new TextDecoder('utf-8').decode(bytes);
    frame.dataset.loaded = '1';
  }
  // the study the Studies tab shows: the one asked for, else the last one shown, else the latest
  let picked = links.length ? links[links.length - 1].dataset.study : '';
  function pick(slug) {
    if (!document.getElementById('view-' + slug)) return;
    picked = slug;
    links.forEach(function (a) {
      if (a.dataset.study === slug) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    document.querySelectorAll('.study').forEach(function (v) { v.hidden = v.id !== 'view-' + slug; });
    document.dispatchEvent(new Event('study-shown'));
  }
  // a hash is a tab, or studies/<study>
  function show(hash, keep) {
    let [id, slug] = hash.split('/');
    if (id && id.indexOf('study-') === 0) { slug = id; id = 'studies'; }
    if (!document.getElementById('tab-' + id)) id = 'overview';
    tabs.forEach(function (t) {
      const on = t.dataset.tab === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      document.getElementById('tab-' + t.dataset.tab).hidden = !on;
    });
    if (id === 'studies') pick(slug || picked);
    if (id === 'design') atlas();
    const want = '#' + id + (id === 'studies' && picked ? '/' + picked : '');
    if (!keep && location.hash !== want) {
      try { history.replaceState(null, '', want); } catch (e) {}
    }
    if (slug) window.scrollTo(0, 0);
    document.dispatchEvent(new Event('study-shown'));
  }
  window.studyPage = {show: show};
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { show(t.dataset.tab); });
    t.addEventListener('keydown', function (e) {
      const to = {ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1,
                  Home: 0, End: tabs.length - 1}[e.key];
      if (to === undefined) return;
      e.preventDefault();
      const next = tabs[(to + tabs.length) % tabs.length];
      show(next.dataset.tab);
      next.focus();
    });
  });
  // an idea of the graph opens what it expected and found; a second click closes it
  const nodes = Array.from(document.querySelectorAll('.node'));
  nodes.forEach(function (n) {
    n.addEventListener('click', function () {
      const open = n.getAttribute('aria-expanded') !== 'true';
      nodes.forEach(function (m) {
        const on = open && m === n;
        m.setAttribute('aria-expanded', on ? 'true' : 'false');
        document.getElementById('idea-' + m.dataset.study).hidden = !on;
        document.querySelector('.dot[data-study="' + m.dataset.study + '"]').classList.toggle('picked', on);
      });
      const hint = document.querySelector('.idea-panel .hint');
      if (hint) hint.hidden = open;
    });
  });
  window.addEventListener('hashchange', function () { show(decodeURIComponent(location.hash.slice(1))); });
  if (picked) pick(picked);
  show(decodeURIComponent(location.hash.slice(1)) || 'overview');
})();
"""

# The explorer of every study tab: draws the view the student picks, and keeps their
# starred views (`stars.json` through `serve`, else for as long as the page is open).
EXPLORE = r"""
(function () {
  // viridis for a number, the Okabe and Ito hues for a category
  const VIRIDIS = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89',
                   '#35b779', '#6ece58', '#b5de2b', '#fde725'];
  const HUES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00',
                '#F0E442', '#5a544d'];
  const store = document.getElementById('stars-data');
  if (!store) return;
  const sets = {};
  document.querySelectorAll('script.explore-data').forEach(function (s) {
    const d = JSON.parse(s.textContent);
    sets[d.id] = d;
  });
  let stars = JSON.parse(store.textContent || '[]');
  const named = document.getElementById('study-names');
  const names = named ? JSON.parse(named.textContent) : {};
  let saving = false;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }
  function short(v) {
    if (v === null || v === undefined) return '–';
    if (typeof v === 'string') return v;
    const a = Math.abs(v);
    for (const [u, s] of [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']]) {
      if (a >= u) return +(v / u).toPrecision(3) + s;
    }
    if (Number.isInteger(v)) return String(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return +v.toPrecision(3) + '';
    if (a >= 0.001) return +v.toPrecision(3) + '';
    return v.toExponential(1);
  }
  function natural(a, b) {
    const x = parseFloat(a), y = parseFloat(b);
    return !isNaN(x) && !isNaN(y) ? x - y : String(a).localeCompare(String(b));
  }
  function jitter(r) { return (((r * 37) % 13) - 6) / 6; }
  function nice(a, b) {
    const raw = (b - a) / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(function (m) { return m * mag; })
      .find(function (st) { return (b - a) / st <= 5; });
    const out = [];
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-12; t += step) out.push(+t.toPrecision(12));
    return out;
  }
  function field(d, name) { return d.fields.findIndex(function (f) { return f.name === name; }); }

  // where a value lies between lo and hi, with the ticks of that axis
  function quantile(xs, q) {
    const v = xs.slice().sort(function (a, b) { return a - b; });
    const at = (v.length - 1) * q, k = Math.floor(at);
    return k + 1 < v.length ? v[k] + (at - k) * (v[k + 1] - v[k]) : v[k];
  }
  // floor: a value far below the others sits on the floor of the axis, so one bad
  // trial does not squash every other dot
  function axis(d, i, lo, hi, floor) {
    const f = d.fields[i];
    const vs = d.rows.map(function (r) { return r[i]; });
    if (f.kind === 'cat') {
      const cats = Array.from(new Set(vs.filter(function (v) { return v !== null; }))).sort(natural);
      const step = (hi - lo) / Math.max(cats.length, 1);
      const pos = function (c) { return lo + step * (cats.indexOf(c) + 0.5); };
      return {
        cats: cats,
        at: function (v, r) { return v === null ? null : pos(v) + jitter(r) * step * 0.28; },
        ticks: cats.map(function (c) { return [pos(c), c]; }),
      };
    }
    const tf = f.log ? Math.log10 : function (x) { return x; };
    const num = vs.filter(function (v) { return v !== null && (!f.log || v > 0); });
    const tnum = num.map(tf);
    let ta = Math.min.apply(null, tnum), tb = Math.max.apply(null, tnum);
    const q10 = quantile(tnum, 0.1), q90 = quantile(tnum, 0.9);
    const cut = floor && tnum.length >= 8 ? q10 - (q90 - q10) : -Infinity;
    if (ta < cut) ta = Math.min.apply(null, tnum.filter(function (t) { return t >= cut; }));
    if (ta === tb) { ta -= 0.5; tb += 0.5; }
    const pad = (tb - ta) * 0.05;
    ta -= pad; tb += pad;
    const at = function (v) {
      return v === null || (f.log && v <= 0) ? null : lo + ((tf(v) - ta) / (tb - ta)) * (hi - lo);
    };
    const low = function (v) { return v !== null && (!f.log || v > 0) && tf(v) < ta; };
    let ticks;
    if (f.log) {
      ticks = [];
      for (let e = Math.ceil(ta); e <= Math.floor(tb); e++) ticks.push(Math.pow(10, e));
      if (ticks.length < 2) ticks = nice(Math.pow(10, ta), Math.pow(10, tb)).filter(function (t) { return t > 0; });
    } else {
      ticks = nice(ta, tb);
    }
    ticks = ticks.filter(function (t) { return tf(t) >= ta && tf(t) <= tb; });
    return {at: at, low: low, ticks: ticks.map(function (t) { return [at(t), short(t)]; }), range: [ta, tb], tf: tf,
            ends: [Math.min.apply(null, num), Math.max.apply(null, num)]};
  }

  // the marks of a category's values, about equal in area: circle, square, triangle,
  // diamond, cross, pentagon (no downward triangle: that is a dot on the floor)
  const SHAPES = 6;
  function mark(k, x, y, r) {
    const poly = function (n, turn, size) {
      const pts = [];
      for (let j = 0; j < n; j++) {
        const a = turn + (2 * Math.PI * j) / n;
        pts.push((x + size * Math.sin(a)).toFixed(1) + ',' + (y - size * Math.cos(a)).toFixed(1));
      }
      return 'M' + pts.join('L') + 'Z';
    };
    if (k === 1) return poly(4, Math.PI / 4, r * 1.2);
    if (k === 2) return poly(3, 0, r * 1.45);
    if (k === 3) return poly(4, 0, r * 1.3);
    if (k === 4) {
      const a = r * 1.15, b = r * 0.42;
      return 'M' + [[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b], [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]]
        .map(function (q) { return (x + q[0]).toFixed(1) + ',' + (y + q[1]).toFixed(1); }).join('L') + 'Z';
    }
    if (k === 5) return poly(5, 0, r * 1.2);
    return 'M' + (x - r).toFixed(1) + ',' + y.toFixed(1) + 'a' + r + ',' + r + ' 0 1,0 ' + (2 * r) + ',0a' + r + ',' + r + ' 0 1,0 ' + (-2 * r) + ',0Z';
  }
  function viridis(t) {
    t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
    const i = Math.min(Math.floor(t), VIRIDIS.length - 2), u = t - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    const ch = function (h, k) { return parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16); };
    return 'rgb(' + [0, 1, 2].map(function (k) { return Math.round(ch(a, k) + u * (ch(b, k) - ch(a, k))); }).join(',') + ')';
  }
  // the colour of every row, and the legend that reads it
  function colour(d, i) {
    const round = function () { return 0; };
    if (i < 0) return {of: function () { return 'var(--course-green)'; }, shape: round, legend: ''};
    const f = d.fields[i];
    const ax = axis(d, i, 0, 1);
    if (f.kind === 'cat') {
      // a category has its own shape as well as its own hue: on a pale projector or in
      // grey the shapes still tell the values apart
      const k = function (v) { return ax.cats.indexOf(v); };
      const of = function (v) { return v === null ? 'var(--course-rule)' : HUES[k(v) % HUES.length]; };
      const shape = function (v) { return v === null ? 0 : k(v) % SHAPES; };
      return {of: of, shape: shape, legend: '<b>' + esc(f.name) + '</b> ' + ax.cats.map(function (c) {
        return '<svg class="sw" viewBox="-6 -6 12 12" aria-hidden="true"><path d="' + mark(shape(c), 0, 0, 4.6) +
          '" fill="' + of(c) + '"/></svg>' + esc(c);
      }).join(' ')};
    }
    const of = function (v) {
      if (v === null || (f.log && v <= 0)) return 'var(--course-rule)';
      return viridis((ax.tf(v) - ax.range[0]) / (ax.range[1] - ax.range[0]));
    };
    const ends = ax.ends;
    return {of: of, shape: round, legend: '<b>' + esc(f.name) + '</b> ' + esc(short(ends[0])) +
      ' <span class="ramp" style="background:linear-gradient(to right,' + VIRIDIS.join(',') + ')"></span> ' +
      esc(short(ends[1])) + (f.log ? ' <span class="m">log</span>' : '')};
  }

  // one picture of d: y against x, coloured by c
  // width: the pixels the picture gets, so its labels keep their size on any screen
  function draw(d, view, small, width) {
    const all = d;
    d = kept(d, view);
    if (!d.rows.length) return '<p class="missing">No trial passes these filters.</p>';
    const W = small ? 320 : Math.round(Math.max(300, Math.min(760, width || 560)));
    const H = small ? 210 : Math.round(Math.max(240, Math.min(380, W * 0.6)));
    const L = small ? 44 : 58, R = W - 10, T = 10, B = H - (small ? 34 : 42);
    const xi = field(d, view.x), yi = field(d, view.y), ci = view.c ? field(d, view.c) : -1;
    if (xi < 0 || yi < 0) return '<p class="missing">This view names a field these trials do not have.</p>';
    const X = axis(d, xi, L, R), Y = axis(d, yi, B, T, true), C = colour(all, ci);
    const out = ['<line x1="' + L + '" x2="' + R + '" y1="' + B + '" y2="' + B + '" class="track"/>',
                 '<line x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + B + '" class="track"/>'];
    X.ticks.forEach(function (t, k) {
      if (t[0] === null || (X.cats && small && X.ticks.length > 6 && k % 2)) return;
      out.push('<text x="' + t[0].toFixed(1) + '" y="' + (B + 14) + '" class="s m" text-anchor="middle">' + esc(t[1]) + '</text>');
    });
    Y.ticks.forEach(function (t) {
      if (t[0] === null) return;
      out.push('<text x="' + (L - 5) + '" y="' + (t[0] + 3).toFixed(1) + '" class="s m" text-anchor="end">' + esc(t[1]) + '</text>');
      out.push('<line x1="' + L + '" x2="' + R + '" y1="' + t[0].toFixed(1) + '" y2="' + t[0].toFixed(1) + '" class="grid"/>');
    });
    out.push('<text x="' + ((L + R) / 2) + '" y="' + (H - 4) + '" class="s" text-anchor="middle">' + esc(d.fields[xi].name) + '</text>');
    if (d.base && view.y === d.objective) {
      const hi = Y.at(d.base[0] + d.base[1]), lo = Y.at(d.base[0] - d.base[1]), at = Y.at(d.base[0]);
      out.push('<rect x="' + L + '" width="' + (R - L) + '" y="' + Math.max(T, hi).toFixed(1) + '" height="' +
               Math.max(1, Math.min(B, lo) - Math.max(T, hi)).toFixed(1) + '" class="band"/>');
      if (at >= T && at <= B) out.push('<line x1="' + L + '" x2="' + R + '" y1="' + at.toFixed(1) + '" y2="' + at.toFixed(1) + '" class="baseline"/>');
    }
    const order = d.rows.map(function (_, r) { return r; }).sort(function (a, b) { return (a === d.winner) - (b === d.winner); });
    order.forEach(function (r) {
      const row = d.rows[r];
      const x = X.at(row[xi], r), y = Y.at(row[yi], r + 5);
      if (x === null || y === null) return;
      const hint = d.fields.map(function (f, k) { return f.name + ': ' + short(row[k]); }).join('\n');
      const win = r === d.winner;
      if (Y.low && Y.low(row[yi])) {
        out.push('<path d="M' + (x - 4).toFixed(1) + ',' + (B - 8) + ' L' + (x + 4).toFixed(1) + ',' + (B - 8) +
                 ' L' + x.toFixed(1) + ',' + (B - 2) + ' Z" fill="' + C.of(ci < 0 ? null : row[ci]) +
                 '" class="pt"><title>' + esc(hint) + '</title></path>');
        return;
      }
      const v = ci < 0 ? null : row[ci];
      out.push('<path d="' + mark(C.shape(v), x, y, win ? 6.5 : small ? 3.6 : 4.6) + '" fill="' + C.of(v) +
               '" class="pt' + (win ? ' best' : '') + '"><title>' +
               esc((win ? 'the best trial\n' : '') + hint) + '</title></path>');
    });
    return '<p class="ylab">' + esc(d.fields[yi].name) + '</p><svg viewBox="0 0 ' + W + ' ' + H +
           '" role="img" aria-label="' + esc(view.y + ' against ' + view.x) + '" xmlns="http://www.w3.org/2000/svg">' +
           out.join('') + '</svg>' + (small ? '<div class="legend">' + C.legend + '</div>' : '');
  }

  function same(a, b) {
    if (a.figure || b.figure) return a.figure === b.figure;
    return (a.study || '') === (b.study || '') && a.dataset === b.dataset && a.x === b.x && a.y === b.y &&
      (a.c || '') === (b.c || '') && JSON.stringify(a.f || []) === JSON.stringify(b.f || []);
  }
  // a number as the student types it: 300k, 1.2M, 4e9, 0.82
  function number(text) {
    const m = String(text).trim().match(/^(-?[0-9.]+(?:e-?[0-9]+)?)\s*([kKmMgGtT]?)$/);
    if (!m) return NaN;
    return parseFloat(m[1]) * ({k: 1e3, m: 1e6, g: 1e9, t: 1e12}[m[2].toLowerCase()] || 1);
  }
  function passes(d, row, f) {
    const i = field(d, f[0]);
    if (i < 0) return false;
    const v = row[i];
    if (f[1] === 'has') return v !== null;
    if (v === null) return false;
    if (f[1] === 'is') return String(v) === f[2];
    if (f[1] === 'is not') return String(v) !== f[2];
    const n = number(f[2]);
    if (typeof v !== 'number' || isNaN(n)) return false;
    return f[1] === '>=' ? v >= n : v <= n;
  }
  // the trials of d a view keeps: its study, if it names one, and every filter
  function kept(d, view) {
    const fs = view.f || [];
    const rows = [], keys = [];
    let winner = null;
    d.rows.forEach(function (row, r) {
      if (view.study && d.keys[r] !== view.study) return;
      if (!fs.every(function (f) { return passes(d, row, f); })) return;
      if (r === d.winner) winner = rows.length;
      rows.push(row); keys.push(d.keys[r]);
    });
    return Object.assign({}, d, {rows: rows, keys: keys, winner: winner});
  }
  function filterText(f) { return f[0] + ' ' + (f[1] === 'has' ? 'is logged' : f[1] + ' ' + f[2]); }
  function current(box) {
    const v = {f: (box.filters || []).slice()};
    box.querySelectorAll('select[data-axis]').forEach(function (s) { v[s.dataset.axis] = s.value; });
    v.study = box.querySelector('.which').value;
    return v;
  }
  function viewOf(box) {
    const d = sets[box.dataset.set], v = current(box);
    const view = {study: v.study, dataset: d.dataset, x: v.x, y: v.y, c: v.c};
    if (v.f.length) view.f = v.f;
    return view;
  }
  // the comparisons a field offers: above or below a number, is or is not a category
  function ops(box) {
    const d = sets[box.dataset.set];
    const f = d.fields[field(d, box.querySelector('.f-field').value)];
    const list = f && f.kind === 'num' ? ['>=', '<=', 'has'] : ['is', 'is not', 'has'];
    const op = box.querySelector('.f-op');
    op.innerHTML = list.map(function (o) { return '<option value="' + o + '">' + (o === 'has' ? 'is logged' : o) + '</option>'; }).join('');
    const val = box.querySelector('.f-value'), choice = box.querySelector('.f-choice');
    const i = f ? field(d, f.name) : -1;
    const vs = i < 0 ? [] : d.rows.map(function (r) { return r[i]; }).filter(function (x) { return x !== null; });
    const cat = !f || f.kind !== 'num';
    val.hidden = cat;
    choice.hidden = !cat;
    if (cat) {
      choice.innerHTML = Array.from(new Set(vs.map(String))).sort(natural).map(function (c) {
        return '<option>' + esc(c) + '</option>';
      }).join('');
    } else {
      val.value = '';
      val.placeholder = vs.length ? 'e.g. ' + short(+quantile(vs, 0.5).toPrecision(1)) : '';
    }
    sync(box);
  }
  // a comparison that needs no value hides the value
  function sync(box) {
    const has = box.querySelector('.f-op').value === 'has';
    const d = sets[box.dataset.set];
    const f = d.fields[field(d, box.querySelector('.f-field').value)];
    const cat = !f || f.kind !== 'num';
    box.querySelector('.f-value').hidden = has || cat;
    box.querySelector('.f-choice').hidden = has || !cat;
  }
  function apply(box, s) {
    box.querySelectorAll('select[data-axis]').forEach(function (sel) { sel.value = s[sel.dataset.axis] || ''; });
    box.querySelector('.which').value = s.study || '';
    box.filters = (s.f || []).slice();
    render(box);
  }
  function noteText() {
    if (store.dataset.mode === 'submitted') return 'Stars on a page of submitted runs last until you close it.';
    if (saving) return 'Saved in ' + store.dataset.file + ': you see them again the next time you open the page.';
    return 'Not saved: this page was not opened by make page, so a star lasts until you close it.';
  }

  function render(box) {
    const d = sets[box.dataset.set];
    const view = viewOf(box);
    const plot = box.querySelector('.plot');
    plot.innerHTML = draw(d, view, false, plot.clientWidth);
    const shown = kept(d, view);
    box.querySelector('.legend').innerHTML = shown.rows.length ? colour(d, view.c ? field(d, view.c) : -1).legend : '';
    box.querySelector('.chips').innerHTML = (view.f || []).map(function (f, k) {
      return '<span class="chip">' + esc(filterText(f)) + '<button type="button" data-k="' + k +
        '" aria-label="remove this filter">×</button></span>';
    }).join('');
    box.querySelectorAll('.chips button').forEach(function (b) {
      b.addEventListener('click', function () { box.filters.splice(+b.dataset.k, 1); render(box); });
    });
    box.querySelector('.count').textContent = shown.rows.length === d.rows.length ? d.rows.length + ' trials'
      : shown.rows.length + ' of ' + d.rows.length + ' trials';
    const on = stars.some(function (s) { return same(s, view); });
    const button = box.querySelector('.star');
    button.setAttribute('aria-pressed', on ? 'true' : 'false');
    button.textContent = on ? '★ starred' : '☆ star this view';
    const mine = stars.filter(function (s) { return !s.figure && s.dataset === d.dataset; });
    box.querySelector('.stars').innerHTML = mine.length ? mine.map(function (s, k) { return card(d, s, k, ''); }).join('')
      : '<p class="foot">None yet. Star a view to keep it here.</p>';
    box.querySelectorAll('.stars figure').forEach(function (fig) { wire(fig, mine[+fig.dataset.k]); });
    box.querySelector('.save-note').textContent = noteText();
  }
  // a starred view as a small picture, with the buttons that open and remove it
  function card(d, s, k, where) {
    return '<figure class="card" data-k="' + k + '">' + (d ? draw(d, s, true) :
      '<p class="missing">This view is of trials that are not on this page.</p>') + '<figcaption>' +
      esc(s.y) + ' against ' + esc(s.x) + (s.c ? ', colour ' + esc(s.c) : '') + where +
      (s.study && names[s.study] ? ' · only ' + esc(names[s.study].short) : '') +
      (s.f && s.f.length ? ' · ' + s.f.map(function (f) { return esc(filterText(f)); }).join(', ') : '') +
      (d ? ' <button type="button" class="open">explore</button>' : '') +
      '<button type="button" class="drop" aria-label="remove this star">×</button></figcaption></figure>';
  }
  function boxOf(s) {
    return Array.from(document.querySelectorAll('.explore')).find(function (b) {
      return sets[b.dataset.set].dataset === s.dataset;
    });
  }
  function wire(fig, s) {
    const box = boxOf(s);
    const open = fig.querySelector('.open');
    if (open) open.addEventListener('click', function () {
      if (window.studyPage) window.studyPage.show('explore');
      apply(box, s);
      box.scrollIntoView({block: 'start'});
    });
    fig.querySelector('.drop').addEventListener('click', function () {
      stars = stars.filter(function (t) { return !same(t, s); });
      save(); renderAll();
    });
  }
  // a starred picture, copied from where it was starred
  function figureCard(s) {
    const src = document.querySelector('.pinnable[data-figure="' + s.figure + '"]');
    const panel = src && src.classList.contains('panel') ? ' panel-pic' : '';
    return '<figure class="card' + panel + '" data-figure="' + esc(s.figure) + '">' +
      (src ? src.querySelector('.fig').innerHTML : '<p class="missing">This picture is not on this page.</p>') +
      '<figcaption>' + esc(src ? src.dataset.title : s.figure) +
      (src ? ' <button type="button" class="open">open</button>' : '') +
      '<button type="button" class="drop" aria-label="remove this star">×</button></figcaption></figure>';
  }
  function wireFigure(fig, s) {
    // a picture of a panel keeps the size it has in its narrow panel, not the room of the card
    if (fig.classList.contains('panel-pic')) {
      let widest = 0;
      fig.querySelectorAll('svg[viewBox]').forEach(function (g) {
        g.style.maxWidth = g.viewBox.baseVal.width + 'px';
        widest = Math.max(widest, g.viewBox.baseVal.width);
      });
      fig.style.maxWidth = widest ? (widest + 24) + 'px' : '420px';
    }
    const open = fig.querySelector('.open');
    if (open) open.addEventListener('click', function () {
      const src = document.querySelector('.pinnable[data-figure="' + s.figure + '"]');
      const tab = src.closest('[role=tabpanel]').id.slice(4);
      if (window.studyPage) window.studyPage.show(tab);
      src.scrollIntoView({block: 'start'});
    });
    fig.querySelector('.drop').addEventListener('click', function () {
      stars = stars.filter(function (t) { return !same(t, s); });
      save(); renderAll();
    });
  }
  // every star: the starred pictures first, then the views gathered by study in the order
  // of the page; drawn on the Starred tab, and on the Overview when the student asks
  function renderStarred() {
    const count = document.querySelector('#tabbtn-starred .count');
    if (count) count.textContent = stars.length ? String(stars.length) : '';
    document.querySelectorAll('.pinnable').forEach(function (f) {
      const on = stars.some(function (s) { return s.figure === f.dataset.figure; });
      const b = f.querySelector('.pin');
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.textContent = on ? '★ starred' : '☆ star';
    });
    const figures = stars.filter(function (s) { return s.figure; });
    const views = stars.filter(function (s) { return !s.figure; });
    // a picture of a study goes with that study; one of every study with all of them
    const ofStudy = function (s) {
      const src = document.querySelector('.pinnable[data-figure="' + s.figure + '"]');
      return src ? src.dataset.study || '' : '';
    };
    const order = Object.keys(names);
    views.forEach(function (s) { if (s.study && order.indexOf(s.study) < 0) order.push(s.study); });
    const html = [''].concat(order).map(function (key) {
        const mine = views.filter(function (s) { return (s.study || '') === key; });
        const pics = figures.filter(function (s) { return ofStudy(s) === key; });
        if (!mine.length && !pics.length) return '';
        const name = key ? names[key] || {short: key, idea: ''} : {short: 'All studies', idea: ''};
        const several = new Set(mine.map(function (s) { return s.dataset; })).size > 1;
        return '<div class="group" data-study="' + esc(key) + '"><h3>' + esc(name.short) + '</h3>' +
          (name.idea ? '<p class="words">' + esc(name.idea) + '</p>' : '') +
          (pics.length ? '<div class="cards pictures">' + pics.map(figureCard).join('') + '</div>' : '') +
          '<div class="cards">' +
          mine.map(function (s, k) {
            const box = boxOf(s);
            return card(box ? sets[box.dataset.set] : null, s, k, several ? ' · on ' + esc(s.dataset) : '');
          }).join('') + '</div></div>';
      }).join('');
    document.querySelectorAll('.starred-list').forEach(function (all) {
      all.innerHTML = html || '<p class="missing">Nothing starred yet. Star a picture on the Studies or ' +
        'Scaling tab, or a view of the explorer, to gather it here.</p>';
      all.querySelectorAll('figure[data-figure]').forEach(function (fig) {
        wireFigure(fig, figures.find(function (s) { return s.figure === fig.dataset.figure; }));
      });
      all.querySelectorAll('.group[data-study]').forEach(function (g) {
        const mine = views.filter(function (s) { return (s.study || '') === g.dataset.study; });
        g.querySelectorAll('figure[data-k]').forEach(function (fig) { wire(fig, mine[+fig.dataset.k]); });
      });
    });
    const note = document.querySelector('#tab-starred .save-note');
    if (note) note.textContent = noteText();
  }
  function renderAll() { document.querySelectorAll('.explore').forEach(render); renderStarred(); }
  function save() {
    if (!saving) return;
    fetch('stars', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(stars)})
      .then(function (r) { if (!r.ok) throw new Error(r.status); })
      .catch(function () { saving = false; renderAll(); });
  }

  document.querySelectorAll('.explore').forEach(function (box) {
    const d = sets[box.dataset.set];
    box.filters = [];
    box.querySelectorAll('select[data-axis]').forEach(function (sel) {
      const want = d.defaults[sel.dataset.axis] || '';
      if (Array.from(sel.options).some(function (o) { return o.value === want; })) sel.value = want;
      sel.addEventListener('change', function () { render(box); });
    });
    box.querySelector('.which').addEventListener('change', function () { render(box); });
    box.querySelector('.f-field').addEventListener('change', function () { ops(box); });
    box.querySelector('.f-op').addEventListener('change', function () { sync(box); });
    ops(box);
    const value = box.querySelector('.f-value');
    function add() {
      const kind = d.fields[field(d, box.querySelector('.f-field').value)].kind;
      const typed = kind === 'num' ? value.value.trim() : box.querySelector('.f-choice').value;
      const f = [box.querySelector('.f-field').value, box.querySelector('.f-op').value, typed];
      if (f[1] !== 'has' && (!f[2] || (kind === 'num' && isNaN(number(f[2]))))) {
        value.setCustomValidity(kind === 'num' ? 'a number, such as 300k or 0.82' : 'a value of this field');
        value.reportValidity();
        return;
      }
      if (f[1] === 'has') f[2] = '';
      box.filters.push(f);
      value.value = '';
      render(box);
    }
    box.querySelector('.f-add').addEventListener('click', add);
    box.querySelector('.f-reset').addEventListener('click', function () {
      apply(box, {study: '', x: d.defaults.x, y: d.defaults.y, c: d.defaults.c, f: []});
    });
    value.addEventListener('keydown', function (e) { if (e.key === 'Enter') add(); });
    value.addEventListener('input', function () { value.setCustomValidity(''); });
    box.querySelector('.star').addEventListener('click', function () {
      const view = viewOf(box);
      const on = stars.some(function (s) { return same(s, view); });
      stars = on ? stars.filter(function (s) { return !same(s, view); }) : stars.concat([view]);
      save(); renderAll();
    });
  });
  document.querySelectorAll('.pinnable .pin').forEach(function (b) {
    b.addEventListener('click', function () {
      const view = {figure: b.closest('.pinnable').dataset.figure};
      const on = stars.some(function (s) { return same(s, view); });
      stars = on ? stars.filter(function (s) { return !same(s, view); }) : stars.concat([view]);
      save(); renderAll();
    });
  });
  // the starred items on the Overview: a choice of this viewer, kept in their browser
  const here = document.getElementById('stars-here');
  const shelf = document.getElementById('starred-overview');
  const hereKey = 'mads-study-page-stars-on-overview';
  if (here && shelf) {
    try { here.checked = localStorage.getItem(hereKey) === '1'; } catch (e) {}
    shelf.hidden = !here.checked;
    here.addEventListener('change', function () {
      shelf.hidden = !here.checked;
      try { localStorage.setItem(hereKey, here.checked ? '1' : '0'); } catch (e) {}
    });
  }
  document.querySelectorAll('.to-explore').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      const key = a.dataset.study;
      if (window.studyPage) window.studyPage.show('explore');
      document.querySelectorAll('.explore').forEach(function (box) {
        const which = box.querySelector('.which');
        if (Array.from(which.options).some(function (o) { return o.value === key; })) { which.value = key; render(box); }
      });
      window.scrollTo(0, 0);
    });
  });
  renderAll();
  // a study's plot is drawn for the width it has once it is shown
  document.addEventListener('study-shown', renderAll);
  let resized = null;
  window.addEventListener('resize', function () { clearTimeout(resized); resized = setTimeout(renderAll, 150); });
  if (location.protocol.indexOf('http') === 0 && store.dataset.mode === 'own') {
    fetch('stars', {cache: 'no-store'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (list) { stars = list; saving = true; renderAll(); })
      .catch(function () {});
    fetch('titles', {cache: 'no-store'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (titles) {
        Object.keys(names).forEach(function (key) { retitle(key, titles[names[key].label] || ''); });
        document.body.classList.add('titles');
      })
      .catch(function () {});
  }

  // a study's title, given by the student under [titles] in page.toml, everywhere it shows
  function retitle(key, title) {
    const name = names[key];
    const shown = title || name.auto;
    name.short = shown;
    const slug = name.slug;
    document.querySelectorAll('.node[data-study="' + slug + '"] .title, #idea-' + slug + ' h3 .t, ' +
      '.switch a[data-study="' + slug + '"], #view-' + slug + ' .lead h2, ' +
      '.which option[value="' + key + '"]').forEach(function (e) {
      e.textContent = shown;
    });
    Object.keys(sets).forEach(function (id) {
      const d = sets[id], i = field(d, 'study');
      if (i >= 0) d.rows.forEach(function (row, r) { if (d.keys[r] === key) row[i] = shown; });
    });
    document.querySelectorAll('.explore').forEach(render);
    renderStarred();
  }
  function keyOf(detail) {
    return Object.keys(names).find(function (k) { return 'idea-' + names[k].slug === detail.id; });
  }
  document.querySelectorAll('.idea-detail .pen').forEach(function (pen) {
    pen.addEventListener('click', function () {
      const detail = pen.closest('.idea-detail');
      const key = keyOf(detail);
      const h3 = pen.parentElement;
      const t = h3.querySelector('.t');
      const input = document.createElement('input');
      input.className = 'title-edit';
      input.maxLength = 80;
      input.value = t.textContent;
      input.setAttribute('aria-label', 'title of this study; empty gives its own name back');
      h3.hidden = true;
      h3.after(input);
      input.focus();
      input.select();
      let done = false;
      function close() { if (done) return; done = true; input.remove(); h3.hidden = false; pen.focus(); }
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') close();
        if (e.key !== 'Enter') return;
        const title = input.value.trim() === names[key].auto ? '' : input.value.trim();
        fetch('titles', {method: 'POST', headers: {'Content-Type': 'application/json'},
                         body: JSON.stringify({study: names[key].label, title: title})})
          .then(function (r) { if (!r.ok) return r.text().then(function (m) { throw new Error(m); }); })
          .then(function () { retitle(key, title); close(); })
          .catch(function (err) { input.setCustomValidity(String(err.message)); input.reportValidity(); });
      });
      input.addEventListener('input', function () { input.setCustomValidity(''); });
      input.addEventListener('blur', close);
    });
  });
})();
"""


# The icons of the rail and the header: one stroke, drawn on a 24 grid.
ICONS = {
    "overview": "M4 4h6v6H4z M14 4h6v6h-6z M4 14h6v6H4z M14 14h6v6h-6z",
    "studies": "M5 20v-9 M12 20V4 M19 20v-6 M3 20h18",
    "starred": "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.8-5.2 2.8 1-5.8-4.3-4.1 5.9-.9z",
    "design": "M12 3l9 5-9 5-9-5z M3 13l9 5 9-5",
    "scaling": "M3 20h18 M5 16l4-5 4 3 6-8",
    "explore": "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13z M15.5 15.5L21 21",
    "runs": "M4 6h16 M4 12h16 M4 18h10",
}


def icon(path: str) -> str:
    return (
        "<svg class='ic' viewBox='0 0 24 24' aria-hidden='true'><path d='"
        + path
        + "' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


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
    stars: Sequence[Mapping[str, object]] = (),
    stars_file: Path | None = None,
) -> str:
    """The page of `login`'s `studies` of `lesson`, with what they wrote in `own`, read
    from the page.toml at `page`; `own` None for a page of submitted runs without one.
    A picture is looked for in the folder of `page`. `stars` are the views the student
    starred, kept in `stars_file`; a page of submitted runs has none."""
    folder = page.parent if page is not None else COURSE_ROOT
    tabs = [
        ("overview", "Overview"),
        ("studies", "Studies"),
        ("explore", "Explore"),
        ("scaling", "Scaling"),
        ("starred", "Starred <span class='count'></span>"),
        ("design", "Design"),
    ]
    nav = "".join(
        f"<button type='button' role='tab' id='tabbtn-{t}' data-tab='{t}' "
        f"aria-controls='tab-{t}' tabindex='-1'>{icon(ICONS[t])}"
        f"<span class='lbl'>{name}</span></button>"
        for t, name in tabs
    )
    names = json.dumps(
        {
            s.key: {
                "short": s.short,
                "auto": s.auto,
                "label": s.label,
                "idea": s.idea,
                "slug": s.slug,
            }
            for s in studies
        }
    ).replace("</", "<\\/")
    scaling, overfitting = capacity(studies)
    panes = {
        "overview": overview(studies, own, folder),
        "studies": studies_tab(studies),
        "explore": explore_tab(studies),
        "scaling": scaling + overfitting,
        "starred": "<section><h2>Starred</h2><p class='cap'>Every starred picture, and every "
        "view starred in the explorer, gathered by study. Open takes it back to where it was starred.</p>"
        "<div id='starred-all' class='starred-list'></div>"
        "<p class='foot save-note'></p>"
        "</section>"
        f"<script type='application/json' id='study-names'>{names}</script>",
        "design": design,
    }
    body = "".join(
        f"<div role='tabpanel' id='tab-{t}' aria-labelledby='tabbtn-{t}'"
        f"{'' if t == 'overview' else ' hidden'}>{panes[t]}</div>"
        for t, _ in tabs
    )
    title = (
        f"Lesson {lesson}: what {login} studied"
        if login
        else f"Lesson {lesson}: your study"
    )
    yours = (
        f"What is yours on the page comes from {html.escape(shown(page))}."
        if own is not None and page is not None
        else "The student's own words are not part of the submitted runs."
    )
    runs = sum(len(s.df) for s in studies)
    mode = "own" if stars_file is not None else "submitted"
    where = shown(stars_file) if stars_file is not None else ""
    starred = json.dumps(list(stars)).replace("</", "<\\/")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{font_faces()}{course_tokens()}{CSS}</style></head><body>"
        "<div class='app'><nav class='rail tabs' role='tablist' aria-label='the page'>"
        f"<span class='badge' title='lesson {lesson}'>L{lesson}</span>{nav}</nav><main>"
        f"<header class='bar'><h1>{html.escape(title)}</h1>"
        f"<span class='pill'>{icon(ICONS['runs'])}{studies_text(len(studies))} · "
        f"{plural(runs, 'run')}</span>"
        "<button type='button' class='theme' id='theme'>light theme</button></header>"
        f"<div class='content'>{body}"
        f"<footer class='sub'>{studies_text(len(studies))}, {plural(runs, 'run')}, read "
        f"from {html.escape(source)}. {yours}</footer></div>"
        f"<script type='application/json' id='stars-data' data-mode='{mode}' "
        f"data-file='{html.escape(where)}'>{starred}</script>"
        f"</main></div><script>{SCRIPT}</script><script>{EXPLORE}</script></body></html>"
    )


STAR_KEYS = ("study", "dataset", "x", "y", "c")
OPS = (">=", "<=", "is", "is not", "has")


def clean_stars(value: object) -> list[dict[str, object]]:
    """The starred views in `value`, each with `STAR_KEYS` as text (an empty study: all
    of them) and its filters under `f`, each a field, one of `OPS` and a value; and the
    starred pictures, each only its `figure`; in order and without doubles. Anything
    else raises TypeError or ValueError."""
    if not isinstance(value, list):
        raise TypeError("the stars are not a list")
    out: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict) and set(item) == {"figure"}:
            if not (isinstance(item["figure"], str) and item["figure"]):
                raise ValueError(f"a starred picture names its figure: {item!r}")
            if item not in out:
                out.append({"figure": item["figure"]})
            continue
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k, ""), str) for k in STAR_KEYS
        ):
            raise TypeError(f"not a starred view: {item!r}")
        star: dict[str, object] = {k: str(item.get(k, "")) for k in STAR_KEYS}
        if not (star["x"] and star["y"]):
            raise ValueError(f"a starred view names its x and y: {item!r}")
        filters = item.get("f") or []
        if not isinstance(filters, list) or not all(
            isinstance(f, list)
            and len(f) == 3
            and all(isinstance(v, str) for v in f)
            and f[1] in OPS
            for f in filters
        ):
            raise ValueError(f"not filters of a starred view: {filters!r}")
        if filters:
            star["f"] = [list(f) for f in filters]
        if star not in out:
            out.append(star)
    return out


def read_stars(path: Path) -> list[dict[str, object]]:
    """The views starred in `path`; none when it is missing or not a list of views."""
    if not path.is_file():
        return []
    try:
        return clean_stars(json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, ValueError) as e:
        print(f"{shown(path)} is left as it is and not read: {e}")
        return []


def write_stars(path: Path, stars: Sequence[Mapping[str, object]]) -> None:
    path.write_text(json.dumps(list(stars), indent=2) + "\n", encoding="utf-8")


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
    titled(studies, read_titles(toml))
    stars_file = None if submitted else folder / "stars.json"
    stars = [] if stars_file is None else read_stars(stars_file)
    return (
        build_page(
            studies,
            lesson,
            login,
            source,
            own,
            toml,
            design_tab(folder, which),
            stars,
            stars_file,
        ),
        studies,
        login,
    )


def page_server(
    page: str, stars_file: Path, toml: Path | None = None
) -> ThreadingHTTPServer:
    """A server of `page` on this machine only, on a free port, that keeps the views
    the student stars in `stars_file` (GET and POST /stars) and the titles they give
    their studies under `[titles]` in the page.toml `toml` (GET and POST /titles). A
    request from another site's page is refused."""
    body = page.encode("utf-8")

    class Page(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def send(self, code: int, data: bytes, kind: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send(200, body, "text/html; charset=utf-8")
            elif path == "/stars":
                stars = json.dumps(read_stars(stars_file)).encode("utf-8")
                self.send(200, stars, "application/json")
            elif path == "/titles" and toml is not None:
                titles = json.dumps(read_titles(toml)).encode("utf-8")
                self.send(200, titles, "application/json")
            else:
                self.send(404, b"not here")

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if self.path not in ("/stars", "/titles") or (
                self.path == "/titles" and toml is None
            ):
                return self.send(404, b"not here")
            if origin is not None and origin != address:
                return self.send(403, b"only the page itself saves")
            size = int(self.headers.get("Content-Length") or 0)
            try:
                sent = json.loads(self.rfile.read(size))
                if self.path == "/stars":
                    write_stars(stars_file, clean_stars(sent))
                elif toml is not None:
                    study, title = sent.get("study"), sent.get("title")
                    if not isinstance(study, str) or not isinstance(title, str):
                        raise TypeError("a title is {study: text, title: text}")
                    write_title(toml, study, title)
            except (TypeError, ValueError, AttributeError) as e:
                return self.send(400, str(e).encode("utf-8"))
            return self.send(200, b"saved")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    address = f"http://127.0.0.1:{server.server_address[1]}"
    return server


def serve(page: str, stars_file: Path, toml: Path | None = None) -> None:
    """Serve `page` (`page_server`), open it, and keep its stars and titles until
    Ctrl-C."""
    server = page_server(page, stars_file, toml)
    address = f"http://127.0.0.1:{server.server_address[1]}"
    print(
        f"served at {address}/ · stars are saved in {shown(stars_file)} · Ctrl-C stops"
    )
    webbrowser.open(address + "/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


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
        "--open",
        action="store_true",
        help="open the page in the browser; a page of your own store is served on "
        "this machine, so the views you star are saved in exercises/stars.json",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="with --open: open the written file, not served (stars are not saved)",
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
    out.write_text(html_, encoding="utf-8")
    print(
        f"{studies_text(len(studies))}, {plural(sum(len(x.df) for x in studies), 'run')} of {login or 'this store'}"
    )
    print(f"page: {out}")
    if args.open and not args.submitted and not args.no_serve:
        folder = exercises_folder(args.lesson)
        serve(html_, folder / "stars.json", args.page or folder / "page.toml")
    elif args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
