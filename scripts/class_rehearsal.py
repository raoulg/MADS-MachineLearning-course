"""A rehearsal of the class on this machine: four students, six studies that build on
each other, handed in through pull requests to a local class repository, and the
teacher's class page of what was merged.

    uv run scripts/class_rehearsal.py                 # one minute per study
    uv run scripts/class_rehearsal.py --minutes 2
    make rehearsal ARGS="--minutes 2"

`PLAN` below is the whole rehearsal: who runs which study in which round, which earlier
studies it builds on, and what the student answers in `exercises/study.py`. Everything
happens in a sandbox, `artefacts/rehearsal/<timestamp>/`:

- `gh-root/`: the class repository and a fork per student, as bare git repositories,
  reached through their github.com URLs by git's `insteadOf` in `gitconfig`;
- `bin/gh`: a fake `gh` that answers `gh api user` with the student's login and records
  the pull requests in `gh-prs.json` (`scripts/gitworld.py`);
- `students/<login>/course`: a clone of the course repository at the commit this script
  runs from, with its own environment and its own Optuna store, and next to it
  `students/<login>/mads-runs`, the student's clone of their fork;
- `teacher/`: the teacher's clones, and `dashboard/index.html`, the class page.

A student's steps are the commands a student types, in their own clone: the class read
as `make dashboard LESSON=1` reads it (without a browser or optuna-dashboard), the
answers written into `exercises/study.py`, `make study LESSON=1`, and
`make share LESSON=1`. After each round, every pull request the fake `gh` recorded is
checked with the steps of the class repository's workflow
(`docs/runs-repo-template/.github/workflows/validate.yml`), run against the class
repository as it would be after the merge, and squash-merged into its `main` when the
check passes. A refused pull request stops the rehearsal with the check's message.

Each student's store gets the handed-out search `lesson1-notebook` of this checkout's
store, so a study's budget comes from trials measured on this machine. Nothing reaches
GitHub: before anything runs, the script makes sure that every command finds the fake
`gh`, that the `insteadOf` rewriting lives in the sandbox, and that no remote of any
sandbox repository resolves to github.com.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gitworld import FAKE_GH, World, gitconfig

COURSE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_STORE = COURSE_ROOT / "artefacts" / "optuna" / "studies.db"
CLASS_REPO = "raoulg/mads-ml-runs-2026-11"
TEACHER = "raoulg"
LESSON = 1
EXERCISE = "notebooks/1_representation_and_head/05_exercises.ipynb"
STUDY_FILE = "notebooks/1_representation_and_head/exercises/study.py"
WORKFLOW = "docs/runs-repo-template/.github/workflows/validate.yml"
HANDED_OUT_STUDY = f"lesson{LESSON}-notebook"
OWN_STUDIES = f"lesson{LESSON}-study*"


@dataclass(frozen=True)
class Planned:
    """One study of the rehearsal: who runs it in which round, the earlier studies it
    builds on (none: the idea handed out with the lesson), and the student's answers.
    `changes` are the axes of `SPACE`, as typed in `study.py`, that differ from the
    space of its parents (for a root, the space as handed out)."""

    key: str
    student: str
    round: int
    builds_on: tuple[str, ...]
    study: str
    idea: str
    expects: str
    changes: dict[str, str] = field(default_factory=dict)


PLAN: tuple[Planned, ...] = (
    Planned(
        key="a1", student="a", round=1, builds_on=(),
        study="a-depth-with-skip",
        idea="does a skip connection let a deeper head help on dtd",
        expects="the skip keeps the linear path of the baseline open, so two or three "
        "layers should no longer lose to one",
        changes={"n_hidden_layers": "Int(1, 3)", "gate": "False"},
    ),
    Planned(
        key="b2", student="b", round=1, builds_on=(),
        study="b-gate-on-dtd",
        idea="does a gate on the hidden block help on dtd",
        expects="a gate can shut hidden units that only fit noise, so it should help "
        "most on the wider heads",
        changes={"skip": "False", "hidden_dim": "Int(128, 1024, log=True)"},
    ),
    Planned(
        key="b1", student="b", round=2, builds_on=("a1",),
        study="b-wide-skip",
        idea="with the skip on does a wider hidden layer help",
        expects="with the skip fixed on, the trials go to the width, and a wider layer "
        "should add a point or two",
        changes={"skip": "True", "hidden_dim": "Int(256, 2048, log=True)"},
    ),
    Planned(
        key="c1", student="c", round=2, builds_on=("b2",),
        study="c-gated-dropout",
        idea="with the gate on how much dropout does the head take",
        expects="the gate already shuts units, so dropout above 0.3 should start to "
        "hurt",
        changes={"gate": "True", "dropout": "Float(0.0, 0.7)"},
    ),
    Planned(
        key="c2", student="c", round=3, builds_on=("b1", "c1"),
        study="c-skip-and-gate",
        idea="skip and gate together on a wide head",
        expects="the skip helped depth and the gate took dropout, so both together "
        "should beat either alone, by less than the sum of the two",
        changes={
            "skip": "True", "gate": "True", "hidden_dim": "Int(256, 2048, log=True)",
        },
    ),
    Planned(
        key="d1", student="d", round=3, builds_on=("c1",),
        study="d-norm-under-gate",
        idea="does a norm layer help the gated head",
        expects="a layer norm keeps what the gate sees in range, so it should help; "
        "a batch norm less, with batches of 64",
        changes={"norm": "Choice(['none', 'layer', 'batch'])"},
    ),
)  # fmt: skip


class Stop(Exception):
    """The rehearsal stops here, with this message."""


# ---- the plan ---------------------------------------------------------------------------


def check_plan(plan: tuple[Planned, ...]) -> None:
    """A plan is a DAG in rounds: every key and study name once, and a study builds
    only on studies of earlier rounds."""
    keys = [p.key for p in plan]
    if len(set(keys)) < len(keys):
        raise Stop(f"plan: a key is there twice in {keys}")
    studies = [p.study for p in plan]
    if len(set(studies)) < len(studies):
        raise Stop(f"plan: a study name is there twice in {studies}")
    by_key = {p.key: p for p in plan}
    for p in plan:
        if p.round < 1:
            raise Stop(f"plan: {p.key} is in round {p.round}; rounds start at 1")
        for parent in p.builds_on:
            if parent not in by_key:
                raise Stop(f"plan: {p.key} builds on {parent}, which is not planned")
            if by_key[parent].round >= p.round:
                raise Stop(
                    f"plan: {p.key} (round {p.round}) builds on {parent} "
                    f"(round {by_key[parent].round}); a parent is of an earlier round"
                )


def changes_of(key: str, plan: tuple[Planned, ...]) -> dict[str, str]:
    """What study `key` searches differently from the space handed out: the changes of
    its parents, in the order it names them, and then its own."""
    planned = next(p for p in plan if p.key == key)
    changes: dict[str, str] = {}
    for parent in planned.builds_on:
        changes.update(changes_of(parent, plan))
    changes.update(planned.changes)
    return changes


def space_of(
    key: str, plan: tuple[Planned, ...], handed_out: dict[str, str]
) -> dict[str, str]:
    """The `SPACE` entries of study `key`: the space handed out, with `changes_of`."""
    return {**handed_out, **changes_of(key, plan)}


# ---- what the student writes into study.py ---------------------------------------------


ANSWERED = ("STUDY", "IDEA", "EXPECTS", "SPACE", "MINUTES")


@dataclass(frozen=True)
class Answers:
    """The answers at the top of `study.py`. `builds_on` holds the names of the ideas
    the study builds on; empty, it builds on the idea handed out with the lesson."""

    study: str
    idea: str
    builds_on: tuple[str, ...]
    expects: str
    space: dict[str, str]
    minutes: float

    def source(self, name: str) -> str:
        """The assignment of `name` as the student types it."""
        if name == "STUDY":
            return f"STUDY = {json.dumps(self.study)}\n"
        if name == "IDEA":
            if not self.builds_on:
                parents = f"Idea.handed_out({LESSON}).name"
            elif len(self.builds_on) == 1:
                parents = json.dumps(self.builds_on[0])
            else:
                parents = "[" + ", ".join(map(json.dumps, self.builds_on)) + "]"
            return f"IDEA = Idea.of({json.dumps(self.idea)}, builds_on={parents})\n"
        if name == "EXPECTS":
            return f"EXPECTS = {json.dumps(self.expects)}\n"
        if name == "SPACE":
            entries = "".join(
                f"        {json.dumps(k)}: {v},\n" for k, v in self.space.items()
            )
            return f"SPACE = Space(\n    {{\n{entries}    }}\n)\n"
        if name == "MINUTES":
            return f"MINUTES = {self.minutes:g}\n"
        raise KeyError(name)


def assignments(text: str) -> dict[str, ast.Assign]:
    """The top-level assignments of `study.py` that hold an answer, by name."""
    out = {}
    for node in ast.parse(text).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ANSWERED
        ):
            out[node.targets[0].id] = node
    missing = [name for name in ANSWERED if name not in out]
    if missing:
        raise Stop(f"study.py: no assignment of {missing} at the top level")
    return out


def handed_out_space(text: str) -> dict[str, str]:
    """The entries of `SPACE` in `study.py` as handed out, each value as typed."""
    value = assignments(text)["SPACE"].value
    if not (
        isinstance(value, ast.Call)
        and value.args
        and isinstance(value.args[0], ast.Dict)
    ):
        raise Stop("study.py: SPACE is not Space({...})")
    out = {}
    for k, v in zip(value.args[0].keys, value.args[0].values, strict=True):
        if k is None:
            raise Stop("study.py: SPACE unpacks another mapping")
        out[ast.literal_eval(k)] = ast.get_source_segment(text, v) or ""
    return out


def edit_study(text: str, answers: Answers) -> str:
    """`study.py` with the student's answers in place of the assignments of `ANSWERED`;
    the questions above them stay as they are."""
    lines = text.splitlines(keepends=True)
    found = assignments(text)
    for name, node in sorted(found.items(), key=lambda kv: -kv[1].lineno):
        lines[node.lineno - 1 : node.end_lineno] = [answers.source(name)]
    return "".join(lines)


# ---- the guards ---------------------------------------------------------------------------


def check_gh(path: str, fake: Path) -> None:
    """The `gh` that `path` finds is `fake`."""
    found = shutil.which("gh", path=path)
    if found is None or Path(found).resolve() != fake.resolve():
        raise Stop(
            f"guard: the gh on PATH is {found}, not the fake {fake}; "
            "the rehearsal never runs a real gh"
        )


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def check_rewrite(env: dict[str, str], sandbox: Path) -> None:
    """git reads its global config from the sandbox, no system config, and every
    `insteadOf` it knows comes from a file inside the sandbox; github.com is rewritten."""
    global_config = env.get("GIT_CONFIG_GLOBAL")
    if not global_config or not inside(Path(global_config), sandbox):
        raise Stop(
            f"guard: GIT_CONFIG_GLOBAL is {global_config!r}, not a file inside "
            f"{sandbox}; the rehearsal never touches your own git config"
        )
    if env.get("GIT_CONFIG_NOSYSTEM") != "1":
        raise Stop("guard: GIT_CONFIG_NOSYSTEM is not 1")
    listed = subprocess.run(
        ["git", "config", "--show-origin", "--get-regexp", r"^url\..*\.insteadof$"],
        cwd=sandbox, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    for line in listed.stdout.splitlines():
        origin = line.split("\t", 1)[0].removeprefix("file:")
        if not inside(Path(origin), sandbox):
            raise Stop(f"guard: an insteadOf rewrite lives outside the sandbox: {line}")
    url = f"https://github.com/{CLASS_REPO}.git"
    got = git_out(["ls-remote", "--get-url", url], sandbox, env)
    if "github.com" in got:
        raise Stop(f"guard: git does not rewrite {url}; it resolves to {got}")


def git_out(args: list[str], cwd: Path, env: dict[str, str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def repositories(sandbox: Path) -> list[Path]:
    """Every git repository of the sandbox: the bare ones under `gh-root`, and every
    clone up to three folders deep."""
    bare = sorted(sandbox.glob("gh-root/*/*.git"))
    clones = sorted(
        p.parent
        for depth in ("*", "*/*", "*/*/*")
        for p in sandbox.glob(f"{depth}/.git")
    )
    return bare + clones


def check_remotes(repos: list[Path], env: dict[str, str]) -> None:
    """No remote of `repos` fetches from or pushes to github.com, as git resolves it."""
    for repo in repos:
        for remote in git_out(["remote"], repo, env).split():
            urls = git_out(["remote", "get-url", "--all", remote], repo, env).split()
            urls += git_out(
                ["remote", "get-url", "--push", "--all", remote], repo, env
            ).split()
            for url in urls:
                if "github.com" in url:
                    raise Stop(
                        f"guard: the remote {remote} of {repo} resolves to {url}"
                    )


# ---- the workflow of the class repository ----------------------------------------------


EXPRESSION = re.compile(r"\$\{\{\s*(.+?)\s*\}\}")


def outputs_of(path: Path) -> dict[str, str]:
    """The outputs a step wrote to its `GITHUB_OUTPUT`: `key=value` or a heredoc."""
    out, lines = {}, path.read_text().splitlines() if path.exists() else []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "<<" in line and ("=" not in line or line.index("<<") < line.index("=")):
            key, delimiter = line.split("<<", 1)
            end = lines.index(delimiter, i + 1)
            out[key] = "\n".join(lines[i + 1 : end])
            i = end + 1
            continue
        key, _, value = line.partition("=")
        out[key] = value
        i += 1
    return out


# ---- the rehearsal ------------------------------------------------------------------------


# environment variables a sandbox command never inherits
DROPPED = (
    "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "NOTEBOOKTESTER_RUNNING",
    "GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
    "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
    "EMAIL", "GH_TOKEN", "GITHUB_TOKEN", "GH_HOST", "MADS_RUNS_REPO", "OPTUNA_STORE",
)  # fmt: skip


@dataclass
class Outcome:
    """What one study of the plan gave, read from what its commands printed."""

    planned: Planned
    parents: tuple[str, ...]
    idea: str = ""
    lines: dict[str, str] = field(default_factory=dict)


class Rehearsal:
    def __init__(self, sandbox: Path, store: Path, minutes: float) -> None:
        self.sandbox = sandbox
        self.store = store
        self.minutes = minutes
        self.bin = sandbox / "bin"
        self.fake = self.bin / "gh"
        self.gitconfig = sandbox / "gitconfig"
        self.students = sorted({p.student for p in PLAN})
        self.outcomes: dict[str, Outcome] = {}
        self.verdicts: list[str] = []
        self.world: World | None = None
        self.head = ""

    # -- running a command

    def env(self, login: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in DROPPED}
        venv_bin = str(COURSE_ROOT / ".venv" / "bin")
        path = [p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin]
        env.update(
            PATH=os.pathsep.join([str(self.bin), *path]),
            GIT_CONFIG_GLOBAL=str(self.gitconfig),
            GIT_CONFIG_NOSYSTEM="1",
            FAKE_GH_LOG=str(self.sandbox / "gh-log.jsonl"),
            FAKE_GH_STATE=str(self.sandbox / "gh-prs.json"),
            FAKE_GH_LOGIN=login,
        )
        return env

    def run(self, cmd: list[str], cwd: Path, login: str, check: bool = True) -> str:
        """Run `cmd` in `cwd` as `login`, print it and everything it prints, and return
        that; stop when it fails and `check`."""
        env = self.env(login)
        check_gh(env["PATH"], self.fake)
        where = cwd.relative_to(self.sandbox) if inside(cwd, self.sandbox) else cwd
        print(f"\n$ {shlex.join(cmd)}\n  (in {where}, as {login})", flush=True)
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )  # fmt: skip
        assert proc.stdout is not None
        seen = []
        for line in proc.stdout:
            seen.append(line)
            print("  | " + line.rstrip(), flush=True)
        code = proc.wait()
        print(f"  -> exit {code}", flush=True)
        if code != 0 and check:
            raise Stop(f"`{shlex.join(cmd)}` in {cwd} exited {code}")
        return "".join(seen)

    def step(self, text: str) -> None:
        print(f"\n# {text}", flush=True)

    # -- the sandbox

    def build(self) -> None:
        self.sandbox.mkdir(parents=True)
        self.bin.mkdir()
        self.fake.write_text(FAKE_GH)
        self.fake.chmod(0o755)
        self.gitconfig.write_text(gitconfig(self.sandbox))
        base = self.env("rehearsal")
        check_gh(base["PATH"], self.fake)
        check_rewrite(base, self.sandbox)
        # git and gh called by this process find the sandbox too
        os.environ.clear()
        os.environ.update(base)
        self.step(f"sandbox {self.sandbox}: guards hold; fake gh at {self.fake}")

        self.world = World(
            self.sandbox, upstream=CLASS_REPO, students=tuple(self.students),
            clone=False,
        )  # fmt: skip
        # GitHub serves a pull request's head by its commit
        git_out(
            ["config", "uploadpack.allowAnySHA1InWant", "true"],
            self.world.upstream, base,
        )  # fmt: skip
        self.step(
            f"class repository {CLASS_REPO} at {self.world.upstream}, a fork per "
            f"student: {', '.join(self.students)}"
        )
        self.head = git_out(["rev-parse", "HEAD"], COURSE_ROOT, base)
        dirty = git_out(
            ["status", "--porcelain", "--untracked-files=no"], COURSE_ROOT, base
        )
        if dirty:
            self.step(
                f"the checkout {COURSE_ROOT} has uncommitted changes; the clones are "
                f"at its commit {self.head[:12]} and do not have them"
            )
        for who in self.students:
            home = self.sandbox / "students" / who
            home.mkdir(parents=True)
            self.world.clone_fork(home / "mads-runs", who)
            self.clone_course(home / "course")
        teacher = self.sandbox / "teacher"
        teacher.mkdir()
        self.clone_course(teacher / "course")
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{CLASS_REPO}.git",
             str(teacher / "mads-runs")],
            env=base, check=True, capture_output=True,
        )  # fmt: skip
        check_remotes(repositories(self.sandbox), base)
        self.step(
            f"{len(repositories(self.sandbox))} repositories in the sandbox; none has "
            "a remote that resolves to github.com"
        )

        for who in self.students:
            course = self.sandbox / "students" / who / "course"
            self.run(["uv", "sync", "--quiet"], course, who)
            copied = self.copy_handed_out(
                course / "artefacts" / "optuna" / "studies.db"
            )
            self.step(
                f"{who}: {HANDED_OUT_STUDY} ({copied} trials) copied from {self.store}"
            )
        self.run(["uv", "sync", "--quiet"], teacher / "course", TEACHER)

    def clone_course(self, where: Path) -> None:
        env = self.env("rehearsal")
        git_out(["clone", "--quiet", "--no-checkout", str(COURSE_ROOT), str(where)],
                self.sandbox, env)  # fmt: skip
        git_out(["checkout", "--quiet", "--detach", self.head], where, env)

    def copy_handed_out(self, to: Path) -> int:
        import optuna

        to.parent.mkdir(parents=True, exist_ok=True)
        optuna.copy_study(
            from_study_name=HANDED_OUT_STUDY,
            from_storage=f"sqlite:///{self.store}",
            to_storage=f"sqlite:///{to}",
        )
        study = optuna.load_study(
            study_name=HANDED_OUT_STUDY, storage=f"sqlite:///{to}"
        )
        return len(study.trials)

    # -- a student's study

    def read_class(self, who: str) -> Path:
        """The class as `make dashboard LESSON=1` reads it in `who`'s clone, without
        opening the page and without optuna-dashboard; the page it wrote."""
        course = self.sandbox / "students" / who / "course"
        self.run(
            ["uv", "run", "scripts/class_analysis.py", "--lesson", str(LESSON)],
            course, who,
        )  # fmt: skip
        self.run(
            ["uv", "run", "scripts/collect_runs.py", "--lesson", str(LESSON),
             "--no-pull"],
            course, who,
        )  # fmt: skip
        return (
            course / "artefacts" / "class-analysis" / f"lesson{LESSON}" / "index.html"
        )

    def parents_of(self, planned: Planned) -> tuple[str, ...]:
        """The names of the ideas `planned` builds on, read from the class store the
        student just built; each must be on the class page too."""
        import optuna

        page = self.read_class(planned.student)
        course = self.sandbox / "students" / planned.student / "course"
        url = f"sqlite:///{course / 'artefacts' / 'optuna' / 'class.db'}"
        names = []
        by_key = {p.key: p for p in PLAN}
        for key in planned.builds_on:
            parent = by_key[key]
            study = f"lesson{LESSON}-study-{parent.study}@{parent.student}"
            try:
                attrs = optuna.load_study(study_name=study, storage=url).user_attrs
            except KeyError:
                raise Stop(
                    f"{planned.key}: the class store of {planned.student} has no study "
                    f"{study}; {key} is not in the class repository"
                ) from None
            name = attrs.get("idea_name", "")
            if name not in page.read_text():
                raise Stop(
                    f"{planned.key}: the idea {name!r} of {key} is not on the class page"
                )
            names.append(name)
            self.step(
                f"{planned.student} reads the idea of {key} on the class page: {name}"
            )
        return tuple(names)

    def study(self, planned: Planned, handed_out: dict[str, str]) -> None:
        who = planned.student
        course = self.sandbox / "students" / who / "course"
        parents = self.parents_of(planned) if planned.builds_on else ()
        answers = Answers(
            study=planned.study,
            idea=planned.idea,
            builds_on=parents,
            expects=planned.expects,
            space=space_of(planned.key, PLAN, handed_out),
            minutes=self.minutes,
        )
        path = course / STUDY_FILE
        path.write_text(edit_study(path.read_text(), answers))
        self.step(
            f"{who} answers the questions of exercises/study.py for {planned.key}:\n"
            + "".join("    " + answers.source(name).replace("\n", "\n    ").rstrip() + "\n"
                      for name in ANSWERED)
        )  # fmt: skip
        out = self.run(["make", "study", f"LESSON={LESSON}"], course, who)
        outcome = Outcome(planned, parents or ("(handed out)",))
        for line in out.splitlines():
            for label in ("budget", "winner", "idea"):
                if line.startswith(f"{label:<9}:"):
                    outcome.lines[label] = line.split(":", 1)[1].strip()
            if line.startswith("the winner against the baseline:"):
                outcome.lines["sigma"] = line.split(":", 1)[1].strip()
        if "idea" not in outcome.lines:
            raise Stop(f"{planned.key}: make study printed no idea line")
        outcome.idea = outcome.lines["idea"].split()[0]
        self.outcomes[planned.key] = outcome

    # -- handing in

    def hand_in(self, planned: Planned) -> None:
        who = planned.student
        course = self.sandbox / "students" / who / "course"
        check_remotes(repositories(self.sandbox), self.env(who))
        out = self.run(
            ["make", "share", f"LESSON={LESSON}",
             f"ARGS=--experiment {shlex.quote(OWN_STUDIES)}"],
            course, who,
        )  # fmt: skip
        urls = re.findall(r"^pull request: (\S+)$", out, re.MULTILINE)
        if not urls:
            raise Stop(f"{planned.key}: make share opened no pull request")
        self.check_and_merge(urls[-1], planned.key)

    def check_and_merge(self, url: str, key: str) -> None:
        assert self.world is not None
        prs = json.loads((self.sandbox / "gh-prs.json").read_text())
        pr = next(p for p in prs if p["url"] == url)
        number = int(url.rsplit("/", 1)[1])
        fork = self.world.forks[pr["head_owner"]]
        env = self.env(pr["author"])
        head = git_out(["--git-dir", str(fork), "rev-parse", pr["head_branch"]],
                       self.sandbox, env)  # fmt: skip
        # GitHub keeps the head of every pull request in the base repository
        git_out(
            ["--git-dir", str(self.world.upstream), "fetch", "--quiet", str(fork),
             f"+refs/heads/{pr['head_branch']}:refs/pull/{number}/head"],
            self.sandbox, env,
        )  # fmt: skip
        self.step(
            f"pull request #{number} of {pr['author']} ({pr['head_owner']}:"
            f"{pr['head_branch']} at {head[:12]}): the check of {WORKFLOW}"
        )
        passed, problems = self.validate(number, pr, head)
        verdict = f"#{number} {pr['author']} ({key}): " + (
            "passed, merged" if passed else "refused"
        )
        self.verdicts.append(verdict)
        if not passed:
            raise Stop(f"pull request {url} is refused by the check:\n{problems}")
        self.world.squash_merge(
            pr["head_branch"], pr["head_owner"],
            message=f"lesson {LESSON}: {pr['head_owner']} (#{number})",
        )  # fmt: skip
        self.step(f"the check passed; pull request #{number} squash-merged into main")

    def validate(self, number: int, pr: dict, head: str) -> tuple[bool, str]:
        """The job `validate-pr` of the workflow, step by step, in a workspace of its
        own: an action as what it does, a `run` step as the bash GitHub runs."""
        import yaml

        flow = yaml.safe_load((COURSE_ROOT / WORKFLOW).read_text())
        job = flow["jobs"]["validate-pr"]
        ws = self.sandbox / "checks" / f"pr-{number}"
        ws.mkdir(parents=True)
        context = {
            "github.event.pull_request.head.sha": head,
            "github.event.pull_request.user.login": pr["author"],
            "github.event.pull_request.number": str(number),
            "github.repository": CLASS_REPO,
        }
        outputs: dict[str, dict[str, str]] = {}

        def value(match: re.Match) -> str:
            expr = match.group(1)
            m = re.fullmatch(r"steps\.([\w-]+)\.outputs\.([\w-]+)", expr)
            if m:
                return outputs.get(m.group(1), {}).get(m.group(2), "")
            if expr not in context:
                raise Stop(
                    f"the rehearsal does not know the expression ${{{{ {expr} }}}}"
                )
            return context[expr]

        for i, step in enumerate(job["steps"]):
            name = step.get("name") or step.get("uses")
            if "uses" in step:
                self.action(step, ws)
                continue
            env = self.env(pr["author"])
            check_gh(env["PATH"], self.fake)
            env.update({k: str(v) for k, v in flow.get("env", {}).items()})
            env.update(
                {
                    k: EXPRESSION.sub(value, str(v))
                    for k, v in step.get("env", {}).items()
                }
            )
            output = ws / f"github-output-{i}"
            output.touch()
            env["GITHUB_OUTPUT"] = str(output)
            script = ws / f"step-{i}.sh"
            script.write_text(EXPRESSION.sub(value, step["run"]))
            print(f"\n  step: {name}", flush=True)
            cwd = ws / step.get("working-directory", ".")
            # an unnamed shell is bash -e on GitHub's runners
            proc = subprocess.run(
                ["bash", "-e", str(script)], cwd=cwd, env=env, capture_output=True,
                text=True, check=False,
            )  # fmt: skip
            for line in (proc.stdout + proc.stderr).splitlines():
                print("  | " + line, flush=True)
            outputs[step.get("id", str(i))] = outputs_of(output)
            if proc.returncode != 0:
                print(f"  -> exit {proc.returncode}: the job fails", flush=True)
                problems = outputs.get("check", {}).get("problems") or proc.stderr
                return False, problems
        return True, ""

    def action(self, step: dict, ws: Path) -> None:
        """What an action of the workflow does, done here: a checkout of the class
        repository (by its github.com URL) or of the course repository (its files at
        the commit this script runs from); uv is already installed."""
        uses, given = step["uses"], step.get("with", {})
        if uses.startswith("astral-sh/setup-uv@"):
            if shutil.which("uv") is None:
                raise Stop("the workflow sets up uv, and uv is not on PATH")
            print("\n  step: setup-uv (uv is on PATH)", flush=True)
            return
        if not uses.startswith("actions/checkout@"):
            raise Stop(f"the rehearsal does not know the action {uses}")
        where = ws / given.get("path", ".")
        repo = given.get("repository")
        env = self.env("rehearsal")
        if repo is None:
            print(f"\n  step: checkout {CLASS_REPO} into {where.name}", flush=True)
            git_out(["clone", "--quiet", f"https://github.com/{CLASS_REPO}.git",
                     str(where)], ws, env)  # fmt: skip
            return
        print(f"\n  step: checkout {repo} ({given.get('sparse-checkout')}) at "
              f"{self.head[:12]} into {where.name}", flush=True)  # fmt: skip
        for rel in str(given.get("sparse-checkout", "")).split():
            target = where / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                git_out(["show", f"{self.head}:{rel}"], COURSE_ROOT, env) + "\n"
            )

    # -- the teacher

    def teacher(self) -> Path:
        teacher = self.sandbox / "teacher"
        course = teacher / "course"
        page = self.sandbox / "dashboard" / "index.html"
        self.run(["uv", "run", "scripts/collect_runs.py", "--lesson", str(LESSON)],
                 course, TEACHER)  # fmt: skip
        self.run(
            ["uv", "run", "scripts/class_analysis.py", "--lesson", str(LESSON),
             "--out", str(page)],
            course, TEACHER,
        )  # fmt: skip
        return page

    def ideas(self) -> list[tuple[str, str, str, list[str]]]:
        """(name, author, words, parents) of every idea of the lesson as the class
        repository's main holds it, read as the class page reads it."""
        from class_analysis import read_class

        subs, _, _ = read_class(
            self.sandbox / "teacher" / "mads-runs", LESSON, fetch=True
        )
        seen = {}
        for sub in subs:
            idea = sub.idea()
            if idea is not None and idea[0] not in seen:
                attrs = sub.run.study.attrs
                seen[idea[0]] = (idea[0], attrs["idea_author"], attrs["idea"], idea[2])
        return list(seen.values())


def handed_out_name() -> str:
    from hpo.record import Idea

    return Idea.handed_out(LESSON).name


def check_source(store: Path) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = 0
    if store.exists():
        url = f"sqlite:///{store}"
        if HANDED_OUT_STUDY in optuna.get_all_study_names(url):
            study = optuna.load_study(study_name=HANDED_OUT_STUDY, storage=url)
            done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    if not done:
        raise Stop(
            f"no finished trial of {HANDED_OUT_STUDY} in {store}: run Part 1 of "
            f"{EXERCISE} on this machine first; it measures what one trial costs here"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=1.0,
        help="the budget of each study (default: 1)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=SOURCE_STORE,
        help=f"the Optuna store to copy {HANDED_OUT_STUDY} from (default: %(default)s)",
    )
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sandbox = COURSE_ROOT / "artefacts" / "rehearsal" / stamp
    rehearsal = Rehearsal(sandbox, args.store.resolve(), args.minutes)
    try:
        check_plan(PLAN)
        check_source(args.store)
        rehearsal.build()
        handed_out = handed_out_space(
            (sandbox / "teacher" / "course" / STUDY_FILE).read_text()
        )
        for rnd in sorted({p.round for p in PLAN}):
            planned = [p for p in PLAN if p.round == rnd]
            rehearsal.step(f"round {rnd}: {', '.join(p.key for p in planned)}")
            for p in planned:
                rehearsal.study(p, handed_out)
            for p in planned:
                rehearsal.hand_in(p)
        page = rehearsal.teacher()
        ideas = rehearsal.ideas()
    except Stop as stop:
        summary(rehearsal)
        sys.exit(f"\nthe rehearsal stops: {stop}\nsandbox: {sandbox}")
    summary(rehearsal)
    root = handed_out_name()
    key_of = {o.idea: k for k, o in rehearsal.outcomes.items()} | {root: "handed out"}
    print("\nthe ideas of the class, as the class repository holds them:")
    for name, author, words, parents in ideas:
        named = ", ".join(f"{p} ({key_of.get(p, '?')})" for p in parents) or "nothing"
        print(f"  {name:<28} {key_of.get(name, '?'):<10} by {author:<6} {words!r}")
        print(f"  {'':<28} builds on {named}")
    expected = {
        o.idea: sorted(o.parents if o.planned.builds_on else (root,))
        for o in rehearsal.outcomes.values()
    }
    held = {name: sorted(parents) for name, _, _, parents in ideas}
    print(
        "the ideas and their parents are as planned"
        if all(held.get(n) == ps for n, ps in expected.items())
        else f"the ideas differ from the plan: planned {expected}, held {held}"
    )
    print(f"\nclass page: {page}\nsandbox: {sandbox}")


def summary(rehearsal: Rehearsal) -> None:
    if not rehearsal.outcomes and not rehearsal.verdicts:
        return
    print("\n==== summary ====")
    for key, o in rehearsal.outcomes.items():
        p = o.planned
        print(f"{key} ({p.student}, round {p.round}) {p.study}")
        print(f"  idea   : {o.lines.get('idea', '-')}")
        print(f"  parents: {', '.join(o.parents)}")
        for label in ("budget", "winner", "sigma"):
            print(f"  {label:<7}: {o.lines.get(label, '-')}")
    for verdict in rehearsal.verdicts:
        print(f"check {verdict}")


if __name__ == "__main__":
    main()
