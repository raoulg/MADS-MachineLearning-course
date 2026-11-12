"""The run submission format, and the validator for a directory of submissions.

One submitted mlflow run is one JSON file in the class runs repository, at

    lesson<N>/<student>/<run_id>.json

where <student> is the student's GitHub login. The file holds everything needed to
recreate the run in another mlflow store: params, tags, the full history of every
metric, small text/JSON artifacts (the architecture), and where the code came from.

This module depends on pydantic and the standard library only, so the runs repository's
CI can validate a push or a pull request without installing mlflow:

    uv run --with pydantic python scripts/run_submission.py check <runs-repo> \
        [--actor <login>] [--teachers raoulg] \
        [--files <newline-separated paths>... | --files-from <NUL-separated file>]

With `--files` or `--files-from`, `<runs-repo>` is the repository as it would be with
the change applied, and the paths are every path the change touches, deleted ones
included.

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
MAX_CHANGED_FILES = 200
ARTIFACT_SUFFIXES = (".json", ".txt", ".md", ".yaml", ".yml")
# Every pattern is used with fullmatch, so a trailing newline never slips through.
LOGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", re.ASCII)  # GitHub's login rule
RUN_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
PATH = re.compile(r"lesson(\d+)/([A-Za-z0-9-]+)/([0-9a-f]{32})\.json", re.ASCII)
EXPERIMENT = re.compile(r"lesson\d+-[a-z0-9-]+", re.ASCII)
KEY = re.compile(r"[\w\-. /]{1,250}")  # mlflow's rule for param, tag and metric keys
MAX_PARAM_CHARS = 6000  # mlflow's limits on a value
MAX_TAG_CHARS = 5000
# mlflow's own tags carry local usernames and absolute paths; these are the ones kept.
KEEP_SYSTEM_TAGS = {"mlflow.note.content", "mlflow.parentRunId"}


def shown(path: str) -> str:
    """A path as it is safe to print: quoted when it holds anything unprintable."""
    return path if path.isprintable() else repr(path)


def check_keys(kind: str, keys: Any) -> None:
    for key in keys:
        if not KEY.fullmatch(key):
            raise ValueError(
                f"{kind} key {key!r}: only letters, digits, _ - . / and spaces, "
                "at most 250 characters"
            )


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
        if not RUN_ID.fullmatch(v):
            raise ValueError("run_id must be a 32-character mlflow run id")
        return v

    @field_validator("experiment")
    @classmethod
    def _experiment(cls, v: str) -> str:
        if not EXPERIMENT.fullmatch(v):
            raise ValueError("experiment must be named lesson<N>-<lowercase-name>")
        return v

    @field_validator("params")
    @classmethod
    def _params(cls, v: dict[str, str]) -> dict[str, str]:
        check_keys("param", v)
        for key, value in v.items():
            if len(value) > MAX_PARAM_CHARS:
                raise ValueError(
                    f"param {key!r} is {len(value)} characters, limit {MAX_PARAM_CHARS}"
                )
        return v

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: dict[str, str]) -> dict[str, str]:
        check_keys("tag", v)
        for key, value in v.items():
            if key.startswith("mlflow.") and key not in KEEP_SYSTEM_TAGS:
                raise ValueError(f"tag {key!r}: mlflow's own tags are not submitted")
            if len(value) > MAX_TAG_CHARS:
                raise ValueError(
                    f"tag {key!r} is {len(value)} characters, limit {MAX_TAG_CHARS}"
                )
        return v

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: dict[str, list[MetricPoint]]) -> dict[str, list[MetricPoint]]:
        check_keys("metric", v)
        return v

    @field_validator("artifacts")
    @classmethod
    def _artifacts(cls, v: dict[str, Any]) -> dict[str, Any]:
        for path, content in v.items():
            parts = path.split("/")
            if (
                path.startswith("/")
                or "\\" in path
                or any(p in ("", ".", "..") for p in parts)
            ):
                raise ValueError(
                    f"artifact {path!r}: the path must be relative, without . or .."
                )
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
        if not LOGIN.fullmatch(v):
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

    def copy_hash(self) -> str:
        """Hash of the run without its id, its name and the student: two files with the
        same copy hash hold the same run, however they are labelled."""
        run = self.run.model_dump(mode="json", exclude={"run_id", "run_name"})
        body = json.dumps(run, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def load(path: Path, root: Path | None = None) -> Submission:
    """Read and validate one submission file. With `root`, also check that the file sits
    at the path its own content says it belongs at."""
    if path.is_symlink():
        raise ValueError(
            "a symbolic link, not a file: submit again, do not edit by hand"
        )
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{size} bytes, limit {MAX_FILE_BYTES}")
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"not valid JSON ({e}): submit again, do not edit by hand"
        ) from None
    sub = Submission.model_validate(data)
    if root is not None:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        m = PATH.fullmatch(rel)
        if m and m.group(2) != sub.student:
            raise ValueError(
                f"this file says it is {sub.student}'s run; "
                f"it sits in {m.group(2)}'s directory"
            )
        if rel != sub.relpath():
            raise ValueError(f"belongs at {sub.relpath()}, found at {rel}")
    return sub


def copies(root: Path) -> dict[str, set[str]]:
    """copy hash -> the students whose directories hold a file with it."""
    out: dict[str, set[str]] = {}
    for path in root.glob("lesson*/*/*.json"):
        try:
            sub = load(path)
        except ValueError:
            continue
        out.setdefault(sub.copy_hash(), set()).add(path.parent.name)
    return out


def check(
    root: Path, files: list[str] | None, actor: str | None, teachers: set[str]
) -> list[str]:
    """Every problem with the submissions under `root`, as readable lines.

    `files` restricts the check to those paths (the files a push or pull request
    changed), and then every one of them must be a `lesson<N>/<student>/<run_id>.json`
    path, checked before anything else; with them, `actor` adds the rule that a student
    only writes inside the directory spelled exactly as their login. A run, by id or by
    content, may sit in one student's directory only."""
    problems = []
    if files is None:
        rels = sorted(
            p.relative_to(root).as_posix() for p in root.glob("lesson*/*/*.json")
        )
    else:
        rels = [f for f in files if f]
    hashes: dict[str, set[str]] | None = None

    for rel in rels:
        m = PATH.fullmatch(rel)
        if not m:
            problems.append(
                f"{shown(rel)}: outside lesson<N>/<student>/<run_id>.json; "
                "a submission changes nothing else"
            )
            continue
        path = root / rel
        owner, run_id = m.group(2), m.group(3)
        deleted = not path.exists() and not path.is_symlink()
        if (
            files is not None
            and actor
            and actor.lower() not in teachers
            and owner != actor
        ):
            verb = "deleted" if deleted else "written"
            problems.append(
                f"{rel}: {verb} by {actor}, but this directory belongs to {owner}"
            )
        if deleted:
            continue
        others = sorted(
            {
                p.parent.name
                for p in root.glob(f"lesson*/*/{run_id}.json")
                if p.parent.name != owner
            }
        )
        if others:
            problems.append(
                f"{rel}: this run id is already submitted by {', '.join(others)}"
            )
        try:
            sub = load(path, root)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            problems.extend(f"{rel}: {line}" for line in explain(e))
            continue
        if hashes is None:
            hashes = copies(root)
        copied = sorted(hashes.get(sub.copy_hash(), set()) - {owner, *others})
        if copied:
            problems.append(
                f"{rel}: this run is already submitted by {', '.join(copied)}, "
                "under another run id"
            )
    return problems


def read_files(files: list[str] | None, files_from: Path | None) -> list[str] | None:
    """The changed paths: from a file of NUL-separated paths, or from `--files`
    arguments split on newlines only. None when neither is given."""
    if files_from is not None:
        raw = files_from.read_bytes().decode("utf-8", errors="surrogateescape")
        return [f for f in raw.split("\0") if f]
    if files is None:
        return None
    return [f for arg in files for f in arg.split("\n") if f]


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
    c.add_argument(
        "--files", nargs="*", help="only these paths, relative to root (one per line)"
    )
    c.add_argument(
        "--files-from",
        type=Path,
        help="only the paths in this file, NUL-separated, relative to root",
    )
    c.add_argument(
        "--actor",
        help="GitHub login that pushed or opened the pull request; "
        "may only write in its own directory",
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
    files = read_files(args.files, args.files_from)
    if files is not None and len(files) > MAX_CHANGED_FILES:
        too_many = (
            f"the change touches {len(files)} files, more than {MAX_CHANGED_FILES}"
        )
        problems = [f"{too_many}; submit in smaller parts"]
    else:
        problems = check(args.root, files, args.actor, teachers)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
