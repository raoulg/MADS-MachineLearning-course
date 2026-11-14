"""class_analysis.py as a script, on a small class store.

The script is copied into a temporary course root, so its figures land there."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "class_analysis.py"


def imported(client, exp, student, name, score, run_id, **tags) -> str:
    """A run as collect_runs.py imports it; returns its submission_ref."""
    ref = f"lesson{exp[1]}/{student}/{run_id}"
    tags = {
        "student": student,
        "source_run_name": name,
        "source_run_id": run_id,
        "submission_ref": ref,
        "role": "final",
        "origin": "study",
        "dataset": "dtd",
        **tags,
    }
    exp_id = client.get_experiment_by_name(exp[0]).experiment_id
    rid = client.create_run(exp_id, run_name=f"{student}:{name}", tags=tags)
    client.log_metric(rid.info.run_id, "accuracy", score)
    client.set_terminated(rid.info.run_id)
    return ref


def test_who_built_on_whom(tmp_path):
    uri = f"sqlite:///{tmp_path / 'class.db'}"
    client = MlflowClient(tracking_uri=uri)
    for name in ("lesson1-study", "lesson2-study"):
        client.create_experiment(
            name, artifact_location=(tmp_path / "art" / name).as_uri()
        )
    bob = imported(client, ("lesson1-study", 1), "bob", "wide", 0.70, "a" * 32)
    dangling = f"lesson2/carol/{'c' * 32}"
    imported(
        client, ("lesson2-study", 2), "alice", "deeper", 0.75, "b" * 32,
        builds_on=f"{bob},{dangling}",
    )  # fmt: skip

    root = tmp_path / "course"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / SCRIPT.name), "--lesson", "2",
         "--tracking-uri", uri],
        capture_output=True, text=True, env=env, check=True,
    ).stdout  # fmt: skip

    lines = [line.split() for line in out.splitlines()]
    header = ["student", "run", "builds_on", "on_run", "score", "on_score", "diff_pt",
              "target"]  # fmt: skip
    assert header in lines, out
    at = lines.index(header)
    links = lines[at + 1 : at + 3]
    assert lines[at + 3] == []
    assert ["alice", "deeper", "bob", "wide", "0.75", "0.7", "5.0", "found"] in links
    assert [
        "alice", "deeper", "carol", "-", "0.75", "NaN", "NaN", "not", "in", "the",
        "class", "store:", dangling,
    ] in links  # fmt: skip
