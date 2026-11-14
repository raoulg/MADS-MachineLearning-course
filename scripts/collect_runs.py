"""Import the class's runs into a local Optuna store, for student and teacher alike.

    uv run scripts/collect_runs.py --lesson 1 --ui

The class's runs are the `lesson<N>/<student>/<run_id>.json` files on the class
repository's `main`. The script fetches that branch into the clone (`../mads-runs`): from
the remote `upstream` when the clone has one (a student's clone of their fork), else
from `origin` (the teacher's clone of the class repository), and reads the files from
`<remote>/main` with git. It never switches branch or touches the working tree, so it
works on whatever branch the clone is on, `lesson<N>` from `submit_runs.py` included.

The class store is rebuilt from those files on every run, so it always holds exactly what
`main` holds (Optuna cannot replace one trial). The handed-out studies of the exercise
(`lesson<N>-notebook` and `lesson<N>-notebook-runs`) are pooled: one study with every
student's trials, the way the class's runs are read together. Every other study is kept
per student, as `<study>@<student>`, because each student searched another space; a trial
that does not fit the pooled study (a param of another kind) goes there as well. Each
trial carries its own user attrs and `student`, `lesson`, `source_run_id`,
`source_run_name`, `submission_ref` (`lesson<N>/<student>/<run_id>`, the name a run that
builds on this one gives it), `submission_hash`, `submitted_at` and `code_*`; its note and
artifacts are restored. Invalid files are reported and skipped.

The class store is separate from your own runs: artefacts/optuna/class.db by default.
`docs/run-protocol.md` describes the protocol.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import warnings
from collections import Counter
from pathlib import Path

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts._upload import ARTIFACTS_ATTR_PREFIX
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    check_distribution_compatibility,
    json_to_distribution,
)
from optuna.trial import TrialState, create_trial
from optuna_dashboard._note import save_note_with_version
from run_submission import Submission, explain, parse

optuna.logging.set_verbosity(optuna.logging.WARNING)
# set_metric_names is marked experimental in Optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

COURSE_ROOT = Path(__file__).resolve().parents[1]
CLASS_DIR = COURSE_ROOT / "artefacts" / "optuna"
DEFAULT_STORE = CLASS_DIR / "class.db"
DEFAULT_RUNS_REPO = COURSE_ROOT.parent / "mads-runs"
BASE = "main"
# A submission sits at lesson<N>/<student>/<file>.json; the validator checks the rest.
SUBMISSION_PATH = re.compile(r"lesson(\d+)/[^/]+/[^/]+\.json")
# The studies every student runs the same way: one pooled study for the whole class.
POOLED = re.compile(r"lesson\d+-notebook(-runs)?")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )


def has_remote(repo: Path, name: str) -> bool:
    return git(repo, "config", "--get", f"remote.{name}.url").returncode == 0


def class_ref(repo: Path, fetch: bool) -> str:
    """`refs/remotes/<remote>/main`, the class repository's main: from `upstream` when
    the clone has it, else from `origin`. With `fetch`, it is fetched first; nothing else
    changes."""
    remote = next((r for r in ("upstream", "origin") if has_remote(repo, r)), None)
    if remote is None:
        sys.exit(f"{repo} has neither an `upstream` nor an `origin` remote")
    tracking = f"refs/remotes/{remote}/{BASE}"
    if fetch:
        fetched = git(
            repo, "fetch", "--quiet", remote, f"+refs/heads/{BASE}:{tracking}"
        )
        if fetched.returncode != 0:
            sys.exit(f"git fetch {remote} in {repo} failed:\n{fetched.stderr.decode()}")
    if git(repo, "rev-parse", "--verify", "--quiet", tracking).returncode != 0:
        sys.exit(f"{repo} has no {remote}/{BASE}; fetch it, or leave out --no-pull")
    return tracking


def submission_files(
    repo: Path, ref: str, lessons: list[int] | None
) -> list[tuple[str, str]]:
    """(path, git mode) of every `lesson<N>/<student>/<file>.json` on `ref`, sorted,
    for `lessons` (all when None)."""
    listed = git(repo, "ls-tree", "-r", "-z", "--full-tree", ref)
    if listed.returncode != 0:
        sys.exit(f"git ls-tree {ref} in {repo} failed:\n{listed.stderr.decode()}")
    out = []
    for entry in listed.stdout.decode(errors="surrogateescape").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        m = SUBMISSION_PATH.fullmatch(path)
        if m and (lessons is None or int(m.group(1)) in lessons):
            out.append((path, meta.split()[0]))
    return sorted(out)


def read_submission(repo: Path, ref: str, path: str, mode: str) -> Submission:
    """The submission at `path` on `ref`, validated as `run_submission.load` does."""
    if mode == "120000":
        raise ValueError("a symbolic link, not a file")
    if mode not in ("100644", "100755"):
        raise ValueError(f"not a regular file (mode {mode})")
    shown = git(repo, "show", f"{ref}:{path}")
    if shown.returncode != 0:
        raise ValueError(f"git show failed: {shown.stderr.decode().strip()}")
    return parse(shown.stdout, path)


def given(value: bool | float | str | None) -> BaseDistribution:
    """The distribution of a param that was given, not drawn: one value."""
    if isinstance(value, bool):
        return CategoricalDistribution([False, True])
    if isinstance(value, int):
        return IntDistribution(value, value)
    if isinstance(value, float):
        return FloatDistribution(value, value)
    return CategoricalDistribution([value])


class ClassStore:
    """The class store being built: Optuna studies in one sqlite file, their artifacts
    next to it."""

    def __init__(self, path: Path):
        for old in (path, path.parent / "class-artifacts"):
            if old.is_file():
                old.unlink()
            elif old.is_dir():
                shutil.rmtree(old)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.url = f"sqlite:///{path}"
        self.artifacts = FileSystemArtifactStore(str(path.parent / "class-artifacts"))
        (path.parent / "class-artifacts").mkdir()
        self.studies: dict[str, optuna.Study] = {}
        self.seen: dict[str, dict[str, BaseDistribution]] = {}

    def study(self, name: str, sub: Submission) -> optuna.Study:
        if name not in self.studies:
            study = optuna.create_study(
                study_name=name,
                storage=self.url,
                direction=sub.run.study.direction,
            )
            study.set_metric_names(["accuracy"])
            for key, value in sub.run.study.attrs.items():
                study.set_user_attr(key, value)
            study.set_user_attr("pooled", name == sub.run.experiment)
            if name != sub.run.experiment:
                study.set_user_attr("student", sub.student)
            self.studies[name] = study
            self.seen[name] = {}
        return self.studies[name]

    def home(self, sub: Submission, distributions: dict[str, BaseDistribution]) -> str:
        """The study a trial goes to: the pooled one when it is a handed-out study and
        its params fit what the study holds, else the student's own."""
        pooled = sub.run.experiment
        own = f"{pooled}@{sub.student}"
        if not POOLED.fullmatch(pooled):
            return own
        for name, dist in distributions.items():
            old = self.seen.get(pooled, {}).get(name)
            if old is None:
                continue
            try:
                check_distribution_compatibility(old, dist)
            except ValueError:
                return own
        return pooled

    def add(self, sub: Submission, digest: str) -> None:
        run = sub.run
        distributions = {
            name: (
                json_to_distribution(json.dumps(run.distributions[name].model_dump()))
                if name in run.distributions
                else given(value)
            )
            for name, value in run.params.items()
        }
        name = self.home(sub, distributions)
        study = self.study(name, sub)
        self.seen[name].update(distributions)

        attrs = {
            **run.attrs,
            "student": sub.student,
            "lesson": sub.lesson,
            "source_run_id": run.run_id,
            "source_run_name": run.run_name,
            "submission_ref": sub.ref(),
            "submission_hash": digest,
            "submitted_at": sub.submitted_at.isoformat(),
            "code_commit": sub.code.commit or "",
            "code_dirty": "" if sub.code.dirty is None else str(sub.code.dirty).lower(),
            "code_remote": sub.code.remote or "",
        }
        system_attrs = {}
        blobs = {}
        for path, content in run.artifacts.items():
            artifact_id = str(uuid.uuid4())
            text = json.dumps(content, indent=1) if path.endswith(".json") else content
            blobs[artifact_id] = text.encode()
            meta = {
                "artifact_id": artifact_id,
                "filename": path.rsplit("/", 1)[-1],
                "mimetype": "application/json"
                if path.endswith(".json")
                else "text/plain",
                "encoding": None,
            }
            system_attrs[ARTIFACTS_ATTR_PREFIX + artifact_id] = json.dumps(meta)
        for artifact_id, raw in blobs.items():
            self.artifacts.write(artifact_id, io.BytesIO(raw))

        trial = create_trial(
            state=TrialState[run.status],
            value=run.value,
            params=run.params,
            distributions=distributions,
            user_attrs=attrs,
            system_attrs=system_attrs,
            intermediate_values={p.step: p.value for p in run.intermediate},
        )
        study.add_trial(trial)
        architecture = run.artifacts.get("architecture.json")
        if isinstance(architecture, dict):
            trial_id = study.trials[-1]._trial_id
            body = f"`{architecture.get('signature', '')}`\n\n```\n{architecture.get('summary', '')}\n```"
            save_note_with_version(study._storage, study._study_id, trial_id, 1, body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(os.environ.get("MADS_RUNS_REPO", DEFAULT_RUNS_REPO)),
        help="clone of the class runs repository, or of your fork of it "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--lesson",
        type=int,
        action="append",
        help="only this lesson (repeatable; default all)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help="class store, rebuilt on every run (default: %(default)s)",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="do not fetch; read <remote>/main as the clone has it",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="start optuna-dashboard on the class store afterwards",
    )
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()

    repo = args.runs_repo.resolve()
    if not repo.is_dir() or git(repo, "rev-parse", "--git-dir").returncode != 0:
        sys.exit(
            f"{repo} is not a clone of the class runs repository (git clone it there, or pass --runs-repo)"
        )
    ref = class_ref(repo, fetch=not args.no_pull)
    files = submission_files(repo, ref, args.lesson)

    store = ClassStore(args.store)

    counts: Counter[str] = Counter()
    students: dict[int, set[str]] = {}
    per_student: Counter[tuple[int, str]] = Counter()
    per_experiment: Counter[str] = Counter()
    for rel, mode in files:
        try:
            sub = read_submission(repo, ref, rel, mode)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            print(f"skipped {rel}: " + "; ".join(explain(e)))
            counts["invalid"] += 1
            continue

        store.add(sub, sub.content_hash())
        counts["new"] += 1
        students.setdefault(sub.lesson, set()).add(sub.student)
        per_student[(sub.lesson, sub.student)] += 1
        per_experiment[sub.run.experiment] += 1

    print(
        f"{len(files)} file(s) in {repo} at {ref.removeprefix('refs/remotes/')}: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    for lesson, who in sorted(students.items()):
        print(f"lesson {lesson}: {len(who)} student(s)")
        for student in sorted(who, key=str.lower):
            print(f"  {student:<40} {per_student[(lesson, student)]:>5} run(s)")
    if per_experiment:
        print("runs per experiment:")
        for name, n in sorted(per_experiment.items()):
            print(f"  {name:<40} {n:>5}")
    print(f"class store: {args.store}")

    ui = [
        "optuna-dashboard",
        store.url,
        "--artifact-dir",
        str(args.store.parent / "class-artifacts"),
        "--port",
        str(args.port),
    ]
    if args.ui:
        print(
            f"starting optuna-dashboard on http://127.0.0.1:{args.port} (ctrl-c to stop)"
        )
        os.execvp(ui[0], ui)
    print("browse with: uv run " + " ".join(ui))


if __name__ == "__main__":
    main()
