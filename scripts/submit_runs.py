"""Submit your Optuna runs for a lesson to the class runs repository.

Reads the trials of one lesson from your local Optuna store (artefacts/optuna/studies.db)
and writes each as a JSON file (lesson<N>/<your-login>/<run_id>.json) into your clone of
your fork of the class runs repository. It validates every file, commits on the branch
`lesson<N>`, pushes that branch to your fork, and opens a pull request against the class
repository. A check on the pull request validates it and merges it.

    uv run scripts/submit_runs.py --lesson 1 --list     # what would be submitted
    uv run scripts/submit_runs.py --lesson 1            # submit it

The clone has two remotes: `origin` (your fork) and `upstream` (the class repository).
By default it submits every finished or pruned trial in the studies `lesson<N>-notebook*`
(the handed-out part of the exercise) and `lesson<N>-study*` (the student's own studies);
any other study only when named with `--experiment`. It skips trials imported from the
class repository by collect_runs.py (a classmate's work) and trials without a `run_id`.
Submitting again is safe: a run already in the class repository or on your branch is
skipped, so only new runs are added, to the open pull request if there is one.

Instructions for a coding agent: docs/submit-runs.md. The format: scripts/run_submission.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import optuna
from optuna.artifacts import FileSystemArtifactStore, get_all_artifact_meta
from optuna.artifacts.exceptions import ArtifactNotFound
from optuna.distributions import distribution_to_json
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial, TrialState
from pydantic import ValidationError
from run_submission import (
    ARTIFACT_SUFFIXES,
    MAX_ARTIFACT_BYTES,
    Submission,
    explain,
    load,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

COURSE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = COURSE_ROOT / "artefacts" / "optuna" / "studies.db"
DEFAULT_RUNS_REPO = COURSE_ROOT.parent / "mads-runs"
CLASS_REPO = "raoulg/mads-ml-runs-2026-11"
BASE = "main"

# collect_runs.py writes these on every trial it imports from the class repository.
IMPORTED_ATTRS = ("submission_ref", "source_run_id")
GITHUB_REMOTE = re.compile(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/*$")
SPREAD_ATTRS = ("accuracy_std", "n_repeats")


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def gh(*args: str) -> subprocess.CompletedProcess | None:
    """Every call to the GitHub CLI goes through here. None when `gh` is not installed
    or does not answer."""
    try:
        return subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=60, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def gh_available() -> bool:
    return shutil.which("gh") is not None


def strip_credentials(url: str) -> str:
    return re.sub(r"//[^/@]+@", "//", url)


def remote_url(repo: Path, name: str) -> str | None:
    """The URL as configured, before any `url.<base>.insteadOf` rewriting."""
    out = git(repo, "config", "--get", f"remote.{name}.url", check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def github_slug(url: str | None) -> str | None:
    """`owner/repo` of a github.com remote URL, None for any other URL."""
    m = GITHUB_REMOTE.search(strip_credentials(url or ""))
    return f"{m.group(1)}/{m.group(2)}" if m else None


def resolve_login(student: str | None, repo: Path) -> str:
    """The student's GitHub login: --student, else `gh`, else the owner of `origin`.
    Every source that gives one must agree."""
    found = {}
    if student:
        found["--student"] = student
    answer = gh("api", "user", "--jq", ".login")
    if answer is not None and answer.returncode == 0 and answer.stdout.strip():
        found["gh api user"] = answer.stdout.strip()
    if git(repo, "rev-parse", "--git-dir", check=False).returncode == 0:
        slug = github_slug(remote_url(repo, "origin"))
        if slug:
            found["the owner of origin"] = slug.split("/")[0]
    if not found:
        sys.exit(
            "cannot tell who you are: pass --student <your GitHub login>, "
            "or log in with `gh auth login`"
        )
    if len({v.lower() for v in found.values()}) > 1:
        sys.exit(
            "these disagree about who you are: "
            + ", ".join(f"{k} says {v}" for k, v in found.items())
            + ". Pass the right --student, log `gh` in as yourself, "
            "or point origin at your own fork (docs/submit-runs.md)."
        )
    # GitHub logins ignore case, but the directory and the check do not: gh knows the
    # account's own spelling.
    return found.get("gh api user") or next(iter(found.values()))


def course_code() -> dict:
    head = git(COURSE_ROOT, "rev-parse", "HEAD", check=False)
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "remote": None}
    dirty = bool(
        git(COURSE_ROOT, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    )
    remote = git(COURSE_ROOT, "remote", "get-url", "origin", check=False).stdout.strip()
    return {
        "commit": head.stdout.strip(),
        "dirty": dirty,
        "remote": strip_credentials(remote) or None,
    }


def default_experiments(lesson: int) -> list[str]:
    """The study-name patterns a lesson submits without `--experiment`: the exercise's
    handed-out studies and the student's own, each with its `-runs` study."""
    return [f"lesson{lesson}-notebook*", f"lesson{lesson}-study*"]


@dataclass
class Found:
    """One trial of the store, with the study it belongs to."""

    study: str
    direction: str
    study_attrs: dict
    trial: FrozenTrial

    @property
    def run_id(self) -> str:
        return self.trial.user_attrs.get("run_id", "")

    @property
    def name(self) -> str:
        return str(
            self.trial.user_attrs.get("name") or f"trial-{self.trial.number:02d}"
        )


class Store:
    """The student's Optuna store: a sqlite file, with its artifacts in the folder
    `artifacts` next to it."""

    def __init__(self, path: Path):
        self.path = path
        self.storage = RDBStorage(f"sqlite:///{path}")
        self.artifacts = FileSystemArtifactStore(str(path.parent / "artifacts"))

    def study_names(self) -> list[str]:
        return sorted(
            s.study_name for s in optuna.get_all_study_summaries(self.storage)
        )


def select_runs(
    store: Store, lesson: int, experiments: list[str] | None, run_ids: list[str] | None
) -> list[Found]:
    """The finished and the pruned trials of the studies a lesson submits: the studies
    named by `experiments` (patterns), else `default_experiments`; with `run_ids`, those
    trials of any study of the lesson."""
    names = store.study_names()
    if run_ids:
        wanted = [f"lesson{lesson}-*"]
    else:
        wanted = experiments or default_experiments(lesson)
    chosen = [n for n in names if any(fnmatch.fnmatchcase(n, w) for w in wanted)]
    if not chosen:
        sys.exit(f"no study matches {wanted}; this store has {names}")

    found = []
    for name in chosen:
        study = optuna.load_study(study_name=name, storage=store.storage)
        for trial in study.trials:
            if trial.state not in (TrialState.COMPLETE, TrialState.PRUNED):
                continue
            hit = Found(
                name, study.direction.name.lower(), dict(study.user_attrs), trial
            )
            if run_ids and hit.run_id not in run_ids:
                continue
            found.append(hit)
    return sorted(found, key=lambda f: f.trial.datetime_start or datetime.min)


def own_runs(runs: list[Found]) -> list[Found]:
    """`runs` without the ones imported from the class repository, and without the ones
    that carry no `run_id`: those are a classmate's work, or were not recorded through
    optunalog."""
    own = [r for r in runs if not any(t in r.trial.user_attrs for t in IMPORTED_ATTRS)]
    if len(own) < len(runs):
        print(
            f"not submitted: {len(runs) - len(own)} run(s) imported from the class "
            "repository (attrs submission_ref or source_run_id); only your own runs are"
        )
    recorded = [r for r in own if r.run_id]
    if len(recorded) < len(own):
        print(
            f"not submitted: {len(own) - len(recorded)} trial(s) without a run_id, "
            "which are not recorded through optunalog; rerun them from the notebook"
        )
    return recorded


def read_artifacts(store: Store, found: Found) -> dict:
    out = {}
    for meta in get_all_artifact_meta(found.trial, storage=store.storage):
        name = meta.filename
        if not name.endswith(ARTIFACT_SUFFIXES):
            continue
        try:
            raw = store.artifacts.open_reader(meta.artifact_id).read()
        except (OSError, ArtifactNotFound) as e:  # artifacts on another machine, ...
            print(
                f"  warning: cannot read the artifact {name} of {found.run_id[:8]}: {e}",
                file=sys.stderr,
            )
            continue
        if len(raw) > MAX_ARTIFACT_BYTES:
            print(
                f"  left out of {found.run_id[:8]}: the artifact {name} is "
                f"{len(raw)} bytes, limit {MAX_ARTIFACT_BYTES}"
            )
            continue
        text = raw.decode()
        out[name] = json.loads(text) if name.endswith(".json") else text
    return out


def millis(moment: datetime | None) -> int | None:
    return None if moment is None else int(moment.timestamp() * 1000)


def to_submission(
    store: Store,
    found: Found,
    lesson: int,
    student: str,
    code: dict,
    now: datetime,
) -> Submission:
    trial = found.trial
    distributions = {
        name: json.loads(distribution_to_json(dist))
        for name, dist in trial.distributions.items()
        if not dist.single()  # a param that was given, not drawn, has no distribution
    }
    return Submission.model_validate(
        {
            "schema": "mads-run/2",
            "lesson": lesson,
            "student": student,
            "submitted_at": now,
            "code": code,
            "run": {
                "run_id": found.run_id,
                "run_name": found.name,
                "experiment": found.study,
                "status": trial.state.name,
                "start_time": millis(trial.datetime_start),
                "end_time": millis(trial.datetime_complete),
                "params": dict(trial.params),
                "distributions": distributions,
                "value": trial.value,
                "intermediate": [
                    {"step": step, "value": value}
                    for step, value in sorted(trial.intermediate_values.items())
                    if math.isfinite(value)
                ],
                "attrs": dict(trial.user_attrs),
                "study": {"direction": found.direction, "attrs": found.study_attrs},
                "artifacts": read_artifacts(store, found),
            },
        }
    )


def summarize(found: Found, limit: int = 4) -> str:
    """`accuracy=0.81 ± 0.012 (n=5), n_params=1000` for a run scored over repeats,
    `accuracy=0.81` for a trial, `pruned after 5 epochs` for a pruned one."""
    attrs = found.trial.user_attrs
    if found.trial.state == TrialState.PRUNED:
        return f"pruned after {len(found.trial.intermediate_values)} epochs"
    shown = []
    if found.trial.value is not None:
        text = f"accuracy={found.trial.value:.4g}"
        std, n = attrs.get("accuracy_std"), attrs.get("n_repeats")
        if std is not None and n is not None:
            text += f" ± {std:.2g} (n={int(n)})"
        shown.append(text)
    for key in ("n_params", "train_seconds"):
        if key in attrs:
            shown.append(f"{key}={attrs[key]:.4g}")
    return ", ".join(shown[:limit])


def show(runs: list[Found]) -> None:
    print(f"{'run_id':<10} {'study':<34} {'role':<11} {'name':<24} outcome")
    for r in runs:
        print(
            f"{r.run_id[:8]:<10} {r.study[:34]:<34} "
            f"{r.trial.user_attrs.get('role', '-')!s:<11} {r.name[:24]:<24} "
            f"{summarize(r)}"
        )


# --- the git part: everything below runs without GitHub, except through `gh` ---------


def ref_exists(repo: Path, ref: str) -> bool:
    return (
        git(repo, "rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0
    )


def is_ancestor(repo: Path, ref: str) -> bool:
    return (
        git(repo, "merge-base", "--is-ancestor", ref, "HEAD", check=False).returncode
        == 0
    )


def identity(repo: Path, login: str) -> list[str]:
    """`-c` options that give a commit an author when git has none configured."""
    name = git(repo, "config", "user.name", check=False).stdout.strip()
    email = git(repo, "config", "user.email", check=False).stdout.strip()
    if name and email:
        return []
    return [
        "-c",
        f"user.name={login}",
        "-c",
        f"user.email={login}@users.noreply.github.com",
    ]


def merge(repo: Path, ref: str, ident: list[str]) -> None:
    if is_ancestor(repo, ref):
        return
    result = git(repo, *ident, "merge", "--no-edit", "--quiet", ref, check=False)
    if result.returncode != 0:
        git(repo, "merge", "--abort", check=False)
        sys.exit(
            f"merging {ref} into your branch in {repo} failed:\n{result.stdout}{result.stderr}"
            "Show this to the teacher; do not resolve it by hand."
        )


def check_clone(repo: Path) -> None:
    if git(repo, "rev-parse", "--git-dir", check=False).returncode != 0:
        sys.exit(
            f"{repo} is not a git clone of your fork of the class runs repository; "
            "do the one-time setup in docs/submit-runs.md, or pass --runs-repo"
        )
    if remote_url(repo, "upstream") is None:
        sys.exit(
            f"{repo} has no `upstream` remote. Add the class repository as upstream:\n"
            f"  git -C {repo} remote add upstream https://github.com/{CLASS_REPO}.git\n"
            f"  git -C {repo} fetch upstream"
        )
    status = git(repo, "status", "--porcelain", "--untracked-files=all").stdout
    blocking = [
        line
        for line in status.splitlines()
        if not line.startswith("?? ") or re.match(r'"?lesson[^/]*/', line[3:])
    ]
    if blocking:  # untracked files outside lesson*/ are not the script's business
        sys.exit(
            f"{repo} has uncommitted changes; the files there are generated, "
            "so do not edit them by hand. Show `git status` there to the teacher."
        )


def prepare_branch(repo: Path, branch: str, ident: list[str]) -> None:
    """Check out `branch`: the local one, else the fork's, else a new one from
    upstream/main; with the fork's commits merged in when it has any."""
    for remote in ("upstream", "origin"):
        fetched = git(repo, "fetch", "--quiet", remote, check=False)
        if fetched.returncode != 0:
            sys.exit(f"git fetch {remote} in {repo} failed:\n{fetched.stderr}")
    if not ref_exists(repo, f"upstream/{BASE}"):
        sys.exit(f"the class repository has no branch {BASE} (upstream/{BASE})")

    fork_branch = f"origin/{branch}"
    if ref_exists(repo, f"refs/heads/{branch}"):
        how = [branch]
    elif ref_exists(repo, fork_branch):
        how = ["--no-track", "-c", branch, fork_branch]
    else:
        how = ["--no-track", "-c", branch, f"upstream/{BASE}"]
    switched = git(repo, "switch", "--quiet", *how, check=False)
    if switched.returncode != 0:
        sys.exit(f"git switch {branch} in {repo} failed:\n{switched.stderr}")
    if ref_exists(repo, fork_branch):
        merge(repo, fork_branch, ident)


def on_upstream(repo: Path, relpath: str) -> bool:
    return (
        git(
            repo, "cat-file", "-e", f"upstream/{BASE}:{relpath}", check=False
        ).returncode
        == 0
    )


def stale_files(
    repo: Path, branch: str, login: str, subs: list[Submission]
) -> list[str]:
    """Files in the student's directory on the branch that are neither in the class
    repository nor among the runs submitted now: runs deleted from the Optuna store since."""
    keep = {s.relpath() for s in subs}
    tracked = git(repo, "ls-files", "--", f"{branch}/{login}/").stdout.split()
    return [p for p in tracked if p not in keep and not on_upstream(repo, p)]


def submitted_upstream(repo: Path, sub: Submission) -> bool:
    shown = git(repo, "show", f"upstream/{BASE}:{sub.relpath()}", check=False)
    if shown.returncode != 0:
        return False
    try:
        return Submission.model_validate_json(shown.stdout).content_hash() == (
            sub.content_hash()
        )
    except ValidationError:
        return False


def submitted_on_branch(repo: Path, sub: Submission) -> bool:
    path = repo / sub.relpath()
    try:
        return path.exists() and load(path).content_hash() == sub.content_hash()
    except ValueError:
        return False


def compare_url(slug: str | None, login: str, branch: str) -> str | None:
    return (
        f"https://github.com/{slug}/compare/{BASE}...{login}:{branch}" if slug else None
    )


def ensure_pull_request(slug: str, login: str, branch: str, lesson: int) -> str:
    """The URL of the open pull request from `login:branch`, opened if there is none."""
    listed = gh(
        "pr", "list", "--repo", slug, "--author", login, "--head", branch,
        "--state", "open", "--json", "url,headRepositoryOwner",
    )  # fmt: skip
    if listed is None or listed.returncode != 0:
        sys.exit(
            "could not ask GitHub for your open pull requests"
            + (f":\n{listed.stderr}" if listed else " (`gh` is not installed)")
        )
    for pr in json.loads(listed.stdout or "[]"):
        owner = (pr.get("headRepositoryOwner") or {}).get("login", "")
        if owner.lower() == login.lower():
            return pr["url"]
    body = (
        f"Runs of lesson {lesson}, submitted with `scripts/submit_runs.py`. "
        "The check validates every file and merges this pull request when they pass."
    )
    created = gh(
        "pr", "create", "--repo", slug, "--base", BASE, "--head", f"{login}:{branch}",
        "--title", f"lesson {lesson}: {login}", "--body", body,
    )  # fmt: skip
    if created is None or created.returncode != 0:
        sys.exit(
            "pushed, but opening the pull request failed"
            + (f":\n{created.stderr}" if created else " (`gh` is not installed)")
            + "\nOpen it in the browser: "
            + (compare_url(slug, login, branch) or f"your fork's branch {branch}")
        )
    return created.stdout.strip().splitlines()[-1]


def submit(
    repo: Path,
    lesson: int,
    login: str,
    subs: list[Submission],
    push: bool = True,
    open_pr: bool = True,
    prune: bool = False,
) -> None:
    """Write `subs` on the branch lesson<N> of the clone at `repo`, commit, push to
    origin, and open the pull request against upstream. With `prune`, `subs` are all of
    the student's runs of the lesson, and a file on the branch that is not among them
    (and not in the class repository) is removed."""
    branch = f"lesson{lesson}"
    check_clone(repo)
    ident = identity(repo, login)
    prepare_branch(repo, branch, ident)

    new = [
        s
        for s in subs
        if not submitted_upstream(repo, s) and not submitted_on_branch(repo, s)
    ]
    stale = stale_files(repo, branch, login, subs) if prune else []
    if new or stale:
        merge(repo, f"upstream/{BASE}", ident)
    if stale:
        git(repo, "rm", "--quiet", "--", *stale)
        print(f"removed {len(stale)} run(s) that are no longer in your Optuna store:")
        for path in stale:
            print(f"  {path}")
    written = []
    for sub in new:
        path = repo / sub.relpath()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sub.dump())
        load(path, repo)  # what was written must read back as valid, at its place
        written.append(sub.relpath())
    if written:
        git(repo, "add", "--", *written)
    if new or stale:
        message = f"lesson {lesson}: {len(written)} run(s) from {login}"
        if stale:
            message += f", {len(stale)} removed"
        git(repo, *ident, "commit", "--quiet", "-m", message)
        if written:
            print(
                f"committed {len(written)} new or changed run(s) on {branch} in {repo}"
            )

    pending = git(
        repo, "diff", "--name-only", "--diff-filter=AM", f"upstream/{BASE}", "HEAD",
        "--", f"{branch}/{login}/",
    ).stdout.split()  # fmt: skip
    if not pending and not stale:
        print("nothing new: every run is already in the class runs repository")
        return
    if pending and not new and not stale:
        print(
            f"nothing new to write; {len(pending)} run(s) on {branch} wait for merging"
        )
    if not push:
        print("not pushed (--no-push)")
        return
    if open_pr and not gh_available():
        print("`gh` is not installed, so the pull request is not opened for you")
        open_pr = False

    fork_branch = f"origin/{branch}"
    if not ref_exists(repo, fork_branch) or (
        git(repo, "rev-parse", fork_branch).stdout
        != git(repo, "rev-parse", "HEAD").stdout
    ):
        pushed = git(
            repo, "push", "--quiet", "origin", f"{branch}:{branch}", check=False
        )
        if pushed.returncode != 0:
            sys.exit(
                f"git push origin {branch} failed:\n{pushed.stderr}Do not force-push."
            )
        git(repo, "fetch", "--quiet", "origin", check=False)
        print(f"pushed {branch} to your fork")
    if not pending:
        print(
            f"{branch} holds no runs that the class repository lacks; "
            "a pull request open from it has nothing left to merge"
        )
        return

    slug = github_slug(remote_url(repo, "upstream"))
    if not open_pr:
        url = compare_url(slug, login, branch)
        print(
            "Open the pull request in the browser"
            + (f": {url}" if url else f", from your fork's branch {branch}")
        )
        return
    url = ensure_pull_request(
        slug or str(remote_url(repo, "upstream")), login, branch, lesson
    )
    print(f"pull request: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lesson", type=int, required=True)
    parser.add_argument(
        "--student",
        help="your GitHub login (default: asked from `gh`, else the owner of origin)",
    )
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(os.environ.get("MADS_RUNS_REPO", DEFAULT_RUNS_REPO)),
        help="your clone of your fork of the class runs repository (default: %(default)s)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("OPTUNA_STORE", DEFAULT_STORE)),
        help="your Optuna store, a sqlite file (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="this study (a name or a pattern like 'lesson1-search*') instead of the "
        "default lesson<N>-notebook* and lesson<N>-study* (repeatable)",
    )
    parser.add_argument("--run-id", action="append", help="only this run (repeatable)")
    parser.add_argument(
        "--list", action="store_true", help="show what would be submitted and stop"
    )
    parser.add_argument(
        "--no-pr",
        action="store_true",
        help="push the branch, but do not open the pull request "
        "(the default when `gh` is not installed)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="write and commit on the branch, but do not push",
    )
    args = parser.parse_args()

    repo = args.runs_repo.resolve()
    student = resolve_login(args.student, repo)

    if not args.store.exists():
        sys.exit(
            f"no Optuna store at {args.store}: run the lesson's exercise notebook first"
        )
    store = Store(args.store)
    runs = own_runs(select_runs(store, args.lesson, args.experiment, args.run_id))
    if not runs:
        sys.exit(f"no trials to submit for lesson {args.lesson} in {args.store}")

    print(f"{len(runs)} run(s) from {args.store}, as {student}")
    if args.list:
        show(runs)
        return

    code = course_code()
    now = datetime.now(UTC).replace(microsecond=0)
    subs, skipped = [], 0
    for run in runs:
        try:
            subs.append(to_submission(store, run, args.lesson, student, code, now))
        except ValidationError as e:
            skipped += 1
            print(
                f"skipped run {run.run_id} ({run.name}), "
                "it does not fit the submission format: " + "; ".join(explain(e))
            )
    if not subs:
        sys.exit("no run fits the submission format; fix the runs above")
    # A skipped run is still in the store, so its file on the branch is not stale.
    submit(
        repo,
        args.lesson,
        student,
        subs,
        push=not args.no_push,
        open_pr=not args.no_pr,
        prune=not (args.run_id or args.experiment or skipped),
    )


if __name__ == "__main__":
    main()
