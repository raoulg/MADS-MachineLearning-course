import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gitworld import World, make_world  # ty: ignore[import-not-found]  (scripts/ is put on sys.path above)
from run_submission import Submission  # ty: ignore[import-not-found]  (scripts/ is put on sys.path above)


def make_submission(
    student: str, lesson: int = 1, run_id: str | None = None
) -> Submission:
    """A small valid submission, as `submit_runs.to_submission` would build it from a trial;
    every call gives a different run."""
    run_id = run_id or uuid4().hex
    start = int(run_id[:8], 16)
    return Submission.model_validate(
        {
            "schema": "mads-run/2",
            "lesson": lesson,
            "student": student,
            "submitted_at": datetime(2026, 11, 1, tzinfo=UTC),
            "code": {"commit": None, "dirty": None, "remote": None},
            "run": {
                "run_id": run_id,
                "run_name": "baseline",
                "experiment": f"lesson{lesson}-notebook-runs",
                "status": "COMPLETE",
                "start_time": start,
                "end_time": start + 1,
                "params": {"lr": 0.01, "encoder": "dinov2-small"},
                "distributions": {},
                "value": 0.5,
                "intermediate": [{"step": 0, "value": 0.5}],
                "attrs": {"role": "baseline", "dataset": "dtd"},
                "study": {"direction": "maximize", "attrs": {"origin": "notebook"}},
                "artifacts": {},
            },
        }
    )


@pytest.fixture
def submission():
    return make_submission


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    """The git world of `scripts/gitworld.py`: class repository, alice's fork and her clone."""
    return make_world(tmp_path, monkeypatch)
