"""Submit your mlflow runs for a lesson to the class runs repository.

Reads the runs of one lesson from your local mlflow store, writes each as a JSON file
into your clone of the class runs repository (lesson<N>/<your-login>/<run_id>.json),
validates every file, and commits and pushes them.

    uv run scripts/submit_runs.py --lesson 1 --list     # what would be submitted
    uv run scripts/submit_runs.py --lesson 1            # submit it

By default it submits every finished run in the experiments named `lesson<N>-...`,
skipping runs logged while notebooktester was running. Submitting again is safe: a run
keeps its file, so only new runs show up as changes.

Instructions for a coding agent: docs/submit-runs.md. The format: scripts/run_submission.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# mlflow draws a progress bar per artifact and announces every database check; neither
# says anything useful here.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")

from mlflow.entities import Run as MlflowRun
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError
from run_submission import ARTIFACT_SUFFIXES, MAX_ARTIFACT_BYTES, Submission, load

COURSE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{COURSE_ROOT / 'artefacts' / 'mlflow' / 'mlruns.db'}"
DEFAULT_RUNS_REPO = COURSE_ROOT.parent / "mads-runs"

# mlflow's own tags carry local usernames and absolute paths; these are the ones kept.
KEEP_SYSTEM_TAGS = {"mlflow.note.content", "mlflow.parentRunId"}


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def detect_student() -> str | None:
    try:
        out = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    login = out.stdout.strip()
    return login if out.returncode == 0 and login else None


def strip_credentials(url: str) -> str:
    return re.sub(r"//[^/@]+@", "//", url)


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


def select_runs(
    client: MlflowClient,
    lesson: int,
    experiments: list[str] | None,
    run_ids: list[str] | None,
    include_test_runs: bool,
) -> list[MlflowRun]:
    if run_ids:
        return [client.get_run(r) for r in run_ids]

    prefix = f"lesson{lesson}-"
    exps = [
        e
        for e in client.search_experiments()
        if (e.name in experiments if experiments else e.name.startswith(prefix))
    ]
    if not exps:
        known = sorted(e.name for e in client.search_experiments())
        sys.exit(
            f"no experiment matches {experiments or prefix + '*'}; this store has {known}"
        )

    runs, token = [], None
    while True:
        page = client.search_runs(
            [e.experiment_id for e in exps],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1000,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if not token:
            break
    if not include_test_runs:
        runs = [r for r in runs if r.data.tags.get("test_run") != "true"]
    return sorted(runs, key=lambda r: r.info.start_time)


def read_artifacts(client: MlflowClient, run_id: str) -> dict:
    out = {}
    try:
        todo = list(client.list_artifacts(run_id))
    except (
        OSError,
        MlflowException,
    ) as e:  # artifacts on another machine, a deleted folder, ...
        print(f"  warning: cannot list artifacts of {run_id[:8]}: {e}", file=sys.stderr)
        return out
    with tempfile.TemporaryDirectory() as tmp:
        while todo:
            info = todo.pop()
            if info.is_dir:
                todo.extend(client.list_artifacts(run_id, info.path))
                continue
            if (
                not info.path.endswith(ARTIFACT_SUFFIXES)
                or (info.file_size or 0) > MAX_ARTIFACT_BYTES
            ):
                continue
            local = Path(client.download_artifacts(run_id, info.path, tmp))
            text = local.read_text()
            out[info.path] = json.loads(text) if info.path.endswith(".json") else text
    return out


def to_submission(
    client: MlflowClient,
    run: MlflowRun,
    lesson: int,
    student: str,
    code: dict,
    experiment_names: dict[str, str],
    now: datetime,
) -> Submission:
    run_id = run.info.run_id
    tags = run.data.tags
    metrics = {
        key: [
            {"step": m.step, "value": m.value, "timestamp": m.timestamp}
            for m in client.get_metric_history(run_id, key)
        ]
        for key in run.data.metrics
    }
    run_commit = tags.get("mlflow.source.git.commit")
    return Submission.model_validate(
        {
            "schema": "mads-run/1",
            "lesson": lesson,
            "student": student,
            "submitted_at": now,
            "code": {**code, "commit": run_commit, "dirty": None, "source": "run"}
            if run_commit
            else {**code, "source": "submit"},
            "run": {
                "run_id": run_id,
                "run_name": run.info.run_name or "",
                "experiment": experiment_names[run.info.experiment_id],
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "tags": {
                    k: v
                    for k, v in tags.items()
                    if not k.startswith("mlflow.") or k in KEEP_SYSTEM_TAGS
                },
                "metrics": metrics,
                "artifacts": read_artifacts(client, run_id),
            },
        }
    )


def pull(repo: Path) -> None:
    result = git(repo, "pull", "--rebase", "--quiet", "origin", "HEAD", check=False)
    if result.returncode != 0:
        sys.exit(f"git pull in {repo} failed:\n{result.stderr}")


def push(repo: Path, attempts: int = 4) -> None:
    for attempt in range(1, attempts + 1):
        result = git(repo, "push", "origin", "HEAD", check=False)
        if result.returncode == 0:
            return
        # Someone else pushed first. Everyone writes only in their own directory, so the
        # rebase cannot conflict.
        print(
            f"  push rejected (attempt {attempt}), rebasing on the latest version and retrying"
        )
        pull(repo)
    sys.exit(f"push failed after {attempts} attempts:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lesson", type=int, required=True)
    parser.add_argument(
        "--student", help="your GitHub login (default: asked from `gh`)"
    )
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(os.environ.get("MADS_RUNS_REPO", DEFAULT_RUNS_REPO)),
        help="your clone of the class runs repository (default: %(default)s)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument(
        "--experiment", action="append", help="only this experiment (repeatable)"
    )
    parser.add_argument("--run-id", action="append", help="only this run (repeatable)")
    parser.add_argument(
        "--include-test-runs",
        action="store_true",
        help="also runs logged by notebooktester",
    )
    parser.add_argument(
        "--list", action="store_true", help="show what would be submitted and stop"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="write and commit, but do not pull or push",
    )
    args = parser.parse_args()

    student = args.student or detect_student()
    if not student:
        sys.exit(
            "cannot tell who you are: pass --student <your GitHub login>, or log in with `gh auth login`"
        )

    client = MlflowClient(tracking_uri=args.tracking_uri)
    runs = select_runs(
        client, args.lesson, args.experiment, args.run_id, args.include_test_runs
    )
    names = {e.experiment_id: e.name for e in client.search_experiments()}
    if not runs:
        sys.exit(
            f"no finished runs to submit for lesson {args.lesson} in {args.tracking_uri}"
        )

    print(f"{len(runs)} run(s) from {args.tracking_uri}, as {student}")
    if args.list:
        print(f"{'run_id':<10} {'experiment':<28} {'role':<11} {'name':<28} metrics")
        for r in runs:
            shown = ", ".join(
                f"{k}={v:.4g}" for k, v in sorted(r.data.metrics.items())[:4]
            )
            print(
                f"{r.info.run_id[:8]:<10} {names[r.info.experiment_id]:<28} "
                f"{r.data.tags.get('role', '-'):<11} {(r.info.run_name or '')[:28]:<28} {shown}"
            )
        return

    repo = args.runs_repo.resolve()
    if git(repo, "rev-parse", "--git-dir", check=False).returncode != 0:
        sys.exit(
            f"{repo} is not a git clone of the class runs repository; clone it there first "
            "(see docs/submit-runs.md) or pass --runs-repo"
        )
    if not args.no_push:
        pull(repo)

    code = course_code()
    now = datetime.now(UTC).replace(microsecond=0)
    written = []
    for run in runs:
        try:
            sub = to_submission(client, run, args.lesson, student, code, names, now)
        except ValidationError as e:
            sys.exit(f"run {run.info.run_id} does not fit the submission format:\n{e}")
        path = repo / sub.relpath()
        if path.exists() and load(path).content_hash() == sub.content_hash():
            continue  # already submitted, unchanged
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sub.dump())
        load(path, repo)  # what was written must read back as valid, at the right place
        written.append(path)

    if not written:
        print("nothing new: every run is already in the runs repository")
        return
    git(repo, "add", "--", *[str(p) for p in written])
    git(
        repo,
        "commit",
        "--quiet",
        "-m",
        f"lesson {args.lesson}: {len(written)} run(s) from {student}",
    )
    print(f"committed {len(written)} new or changed run(s) in {repo}")
    if args.no_push:
        print("not pushed (--no-push)")
        return
    push(repo)
    print("pushed")


if __name__ == "__main__":
    main()
