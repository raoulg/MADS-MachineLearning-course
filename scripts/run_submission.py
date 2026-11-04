"""The run submission format, and the validator for a directory of submissions.

One submitted mlflow run is one JSON file in the class runs repository, at

    lesson<N>/<student>/<run_id>.json

where <student> is the student's GitHub login. The file holds everything needed to
recreate the run in another mlflow store: params, tags, the full history of every
metric, small text/JSON artifacts (the architecture), and where the code came from.

This module depends on pydantic and the standard library only, so the runs repository's
CI can validate a push without installing mlflow:

    uv run --with pydantic python scripts/run_submission.py check <runs-repo> \
        [--actor <login>] [--teachers raoulg] [--files <changed paths>...]

`docs/run-protocol.md` describes the protocol around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA = "mads-run/1"
MAX_FILE_BYTES = 2_000_000
MAX_ARTIFACT_BYTES = 256_000
ARTIFACT_SUFFIXES = (".json", ".txt", ".md", ".yaml", ".yml")
LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")  # GitHub's rule for a login
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
PATH = re.compile(r"^lesson(\d+)/([^/]+)/([^/]+)\.json$")


class MetricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int
    value: float
    timestamp: int


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    run_name: str
    experiment: str
    status: Literal["FINISHED", "FAILED", "KILLED", "RUNNING", "SCHEDULED"]
    start_time: int
    end_time: int | None
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, list[MetricPoint]]
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="artifact path -> parsed JSON for .json files, text otherwise",
    )

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, v: str) -> str:
        if not RUN_ID.match(v):
            raise ValueError("run_id must be a 32-character mlflow run id")
        return v

    @field_validator("artifacts")
    @classmethod
    def _artifacts(cls, v: dict[str, Any]) -> dict[str, Any]:
        for path, content in v.items():
            if not path.endswith(ARTIFACT_SUFFIXES):
                raise ValueError(
                    f"artifact {path!r}: only {ARTIFACT_SUFFIXES} files are submitted"
                )
            size = len(json.dumps(content))
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"artifact {path!r} is {size} bytes, limit {MAX_ARTIFACT_BYTES}"
                )
        return v


class Code(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commit: str | None
    dirty: bool | None
    source: Literal["run", "submit"] = Field(
        description="'run': recorded by mlflow when the run started; "
        "'submit': the course checkout's HEAD when the run was submitted",
    )
    remote: str | None


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: Literal["mads-run/1"] = Field(alias="schema")
    lesson: int = Field(ge=1, le=8)
    student: str
    submitted_at: datetime
    code: Code
    run: Run

    @field_validator("student")
    @classmethod
    def _student(cls, v: str) -> str:
        if not LOGIN.match(v):
            raise ValueError("student must be a GitHub login")
        return v

    def relpath(self) -> str:
        return f"lesson{self.lesson}/{self.student}/{self.run.run_id}.json"

    def dump(self) -> str:
        return (
            json.dumps(
                self.model_dump(mode="json", by_alias=True), indent=1, sort_keys=True
            )
            + "\n"
        )

    def content_hash(self) -> str:
        """Hash of the run itself, independent of when it was submitted."""
        body = json.dumps(self.run.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def load(path: Path, root: Path | None = None) -> Submission:
    """Read and validate one submission file. With `root`, also check that the file sits
    at the path its own content says it belongs at."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{size} bytes, limit {MAX_FILE_BYTES}")
    sub = Submission.model_validate(json.loads(path.read_text()))
    if root is not None:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        if rel != sub.relpath():
            raise ValueError(f"belongs at {sub.relpath()}, found at {rel}")
    return sub


def check(
    root: Path, files: list[str] | None, actor: str | None, teachers: set[str]
) -> list[str]:
    """Every problem with the submissions under `root`, as readable lines.

    `files` restricts the check to those paths (the files a push changed); with them,
    `actor` adds the rule that a student only writes inside their own directory."""
    problems = []
    if files is None:
        paths = sorted(root.glob("lesson*/*/*.json"))
    else:
        # a CI step may pass the changed paths as one newline-separated argument
        paths = [
            root / f for arg in files for f in arg.split() if f.startswith("lesson")
        ]

    for path in paths:
        rel = path.relative_to(root).as_posix()
        m = PATH.match(rel)
        if not m:
            problems.append(f"{rel}: not a lesson<N>/<student>/<run_id>.json path")
            continue
        owner = m.group(2)
        if (
            files is not None
            and actor
            and actor.lower() not in teachers
            and owner.lower() != actor.lower()
        ):
            problems.append(
                f"{rel}: written by {actor}, but this directory belongs to {owner}"
            )
        if not path.exists():
            continue  # a deletion; the ownership rule above still applies
        try:
            load(path, root)
        except (ValueError, json.JSONDecodeError) as e:
            problems.extend(f"{rel}: {line}" for line in explain(e))
    return problems


def explain(error: Exception) -> list[str]:
    """One readable line per problem in a failed `load`."""
    if isinstance(error, ValidationError):
        return [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in error.errors()
        ]
    return [str(error)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate the submissions in a runs repository")
    c.add_argument("root", type=Path)
    c.add_argument("--files", nargs="*", help="only these paths, relative to root")
    c.add_argument(
        "--actor", help="GitHub login that pushed; may only write in its own directory"
    )
    c.add_argument(
        "--teachers", default="", help="comma-separated logins that may write anywhere"
    )
    sub.add_parser("schema", help="print the JSON schema of one submission file")
    args = parser.parse_args()

    if args.cmd == "schema":
        print(json.dumps(Submission.model_json_schema(by_alias=True), indent=2))
        return

    teachers = {t.strip().lower() for t in args.teachers.split(",") if t.strip()}
    problems = check(args.root, args.files, args.actor, teachers)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
