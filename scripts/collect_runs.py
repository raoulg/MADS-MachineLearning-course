"""Pull the class runs repository and import every submission into a local mlflow store.

    uv run scripts/collect_runs.py --lesson 1 --ui

Runs from the repository's `lesson<N>/<student>/<run_id>.json` files land in the
experiment they were logged under, named `<student>:<run name>` and tagged with
`student`, `lesson`, `source_run_id`, `code_commit` and a content hash. Re-running is
safe: a submission already imported unchanged is skipped, a changed one replaces its
earlier import. Invalid files are reported and skipped.

The class store is separate from your own runs: artefacts/mlflow/class.db by default.
`docs/run-protocol.md` describes the protocol.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# mlflow draws a progress bar per artifact and announces every database check; neither
# says anything useful here.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from run_submission import Submission, explain, load

COURSE_ROOT = Path(__file__).resolve().parents[1]
CLASS_DIR = COURSE_ROOT / "artefacts" / "mlflow"
DEFAULT_TRACKING_URI = f"sqlite:///{CLASS_DIR / 'class.db'}"
DEFAULT_RUNS_REPO = COURSE_ROOT.parent / "mads-runs"
BATCH = 1000  # mlflow's log_batch limit for metrics; params and tags are capped at 100


def experiment_id(client: MlflowClient, name: str, artifact_root: Path) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(
        name, artifact_location=(artifact_root / name).as_uri()
    )


def imported_runs(
    client: MlflowClient, exp_id: str
) -> dict[tuple[str, str], tuple[str, str]]:
    """(student, source_run_id) -> (run_id in this store, content hash)."""
    out, token = {}, None
    while True:
        page = client.search_runs([exp_id], max_results=1000, page_token=token)
        for r in page:
            t = r.data.tags
            if "source_run_id" in t:
                out[(t["student"], t["source_run_id"])] = (
                    r.info.run_id,
                    t.get("submission_hash", ""),
                )
        token = page.token
        if not token:
            return out


def import_run(client: MlflowClient, exp_id: str, sub: Submission, digest: str) -> None:
    run = sub.run
    tags = {
        **run.tags,
        "student": sub.student,
        "lesson": str(sub.lesson),
        "source_run_id": run.run_id,
        "source_run_name": run.run_name,
        "submission_hash": digest,
        "submitted_at": sub.submitted_at.isoformat(),
        "code_commit": sub.code.commit or "",
        "code_dirty": "" if sub.code.dirty is None else str(sub.code.dirty).lower(),
        "code_remote": sub.code.remote or "",
    }
    created = client.create_run(
        exp_id, start_time=run.start_time, run_name=f"{sub.student}:{run.run_name}"
    )
    rid = created.info.run_id

    params = [Param(k, v) for k, v in run.params.items()]
    tag_list = [RunTag(k, v) for k, v in tags.items()]
    metrics = [
        Metric(k, p.value, p.timestamp, p.step)
        for k, points in run.metrics.items()
        for p in points
    ]
    for i in range(0, len(params), 100):
        client.log_batch(rid, params=params[i : i + 100])
    for i in range(0, len(tag_list), 100):
        client.log_batch(rid, tags=tag_list[i : i + 100])
    for i in range(0, len(metrics), BATCH):
        client.log_batch(rid, metrics=metrics[i : i + BATCH])

    for path, content in run.artifacts.items():
        if path.endswith(".json"):
            client.log_dict(rid, content, path)
        else:
            client.log_text(rid, content, path)
    client.set_terminated(rid, status=run.status, end_time=run.end_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-repo",
        type=Path,
        default=Path(os.environ.get("MADS_RUNS_REPO", DEFAULT_RUNS_REPO)),
        help="clone of the class runs repository (default: %(default)s)",
    )
    parser.add_argument(
        "--lesson",
        type=int,
        action="append",
        help="only this lesson (repeatable; default all)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="class store (default: %(default)s)",
    )
    parser.add_argument("--no-pull", action="store_true", help="use the clone as it is")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="start `mlflow ui` on the class store afterwards",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    repo = args.runs_repo.resolve()
    if not (repo / ".git").exists():
        sys.exit(
            f"{repo} is not a clone of the class runs repository (git clone it there, or pass --runs-repo)"
        )
    if not args.no_pull:
        pulled = subprocess.run(
            ["git", "-C", str(repo), "pull", "--ff-only", "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pulled.returncode != 0:
            sys.exit(f"git pull in {repo} failed:\n{pulled.stderr}")

    lessons = args.lesson or sorted(
        {int(p.name.removeprefix("lesson")) for p in repo.glob("lesson*") if p.is_dir()}
    )
    files = [f for n in lessons for f in sorted(repo.glob(f"lesson{n}/*/*.json"))]

    # The architecture files go next to a sqlite store, in class-artifacts/<experiment>.
    artifact_root = CLASS_DIR / "class-artifacts"
    if args.tracking_uri.startswith("sqlite:///"):
        db_dir = Path(args.tracking_uri.removeprefix("sqlite:///")).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = db_dir / "class-artifacts"
    client = MlflowClient(tracking_uri=args.tracking_uri)

    counts: Counter[str] = Counter()
    students: dict[int, set[str]] = {}
    per_student: Counter[tuple[int, str]] = Counter()
    per_experiment: Counter[str] = Counter()
    index: dict[str, dict] = {}
    for path in files:
        rel = path.relative_to(repo).as_posix()
        try:
            sub = load(path, repo)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            print(f"skipped {rel}: " + "; ".join(explain(e)))
            counts["invalid"] += 1
            continue

        exp_id = experiment_id(client, sub.run.experiment, artifact_root)
        if exp_id not in index:
            index[exp_id] = imported_runs(client, exp_id)
        key = (sub.student, sub.run.run_id)
        digest = sub.content_hash()
        existing = index[exp_id].get(key)
        if existing and existing[1] == digest:
            counts["unchanged"] += 1
        else:
            if existing:
                client.delete_run(existing[0])
                counts["updated"] += 1
            else:
                counts["new"] += 1
            import_run(client, exp_id, sub, digest)
            index[exp_id][key] = ("", digest)
        students.setdefault(sub.lesson, set()).add(sub.student)
        per_student[(sub.lesson, sub.student)] += 1
        per_experiment[sub.run.experiment] += 1

    print(
        f"{len(files)} file(s) in {repo}: "
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
    print(f"class store: {args.tracking_uri}")

    ui = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        args.tracking_uri,
        "--port",
        str(args.port),
    ]
    if args.ui:
        print(f"starting mlflow ui on http://127.0.0.1:{args.port} (ctrl-c to stop)")
        os.execvp(ui[0], ui)
    print("browse with: uv run " + " ".join(ui))


if __name__ == "__main__":
    main()
