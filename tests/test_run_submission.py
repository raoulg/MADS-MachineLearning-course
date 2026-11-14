import json
from pathlib import Path

import pytest
from run_submission import (  # ty: ignore[import-not-found]
    Submission,
    check,
    read_files,
)


def write(root: Path, sub: Submission, rel: str | None = None) -> str:
    rel = rel or sub.relpath()
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sub.dump())
    return rel


def test_valid_file_in_own_directory_passes(tmp_path, submission):
    rel = write(tmp_path, submission("alice"))
    assert check(tmp_path, [rel], "alice", set()) == []


def test_write_into_another_students_directory(tmp_path, submission):
    rel = write(tmp_path, submission("bob"))
    problems = check(tmp_path, [rel], "alice", set())
    assert problems == [f"{rel}: written by alice, but this directory belongs to bob"]


def test_teacher_may_write_anywhere(tmp_path, submission):
    rel = write(tmp_path, submission("bob"))
    assert check(tmp_path, [rel], "raoulg", {"raoulg"}) == []


def test_relabelled_copy_is_caught(tmp_path, submission):
    original = submission("bob")
    write(tmp_path, original)
    copy = original.model_copy(update={"student": "alice"})
    rel = write(tmp_path, copy)
    problems = check(tmp_path, [rel], "alice", set())
    assert problems == [f"{rel}: this run id is already submitted by bob"]


def test_file_naming_another_student(tmp_path, submission):
    sub = submission("bob")
    rel = write(tmp_path, sub, f"lesson1/alice/{sub.run.run_id}.json")
    problems = check(tmp_path, [rel], "alice", set())
    assert problems == [
        f"{rel}: this file says it is bob's run; it sits in alice's directory"
    ]


def test_invalid_json(tmp_path, submission):
    rel = write(tmp_path, submission("alice"))
    (tmp_path / rel).write_text('{"schema": "mads-run/1",')
    [problem] = check(tmp_path, [rel], "alice", set())
    assert problem.startswith(f"{rel}: not valid JSON (")
    assert problem.endswith("): submit again, do not edit by hand")


def test_path_outside_lessons(tmp_path, submission):
    rel = write(tmp_path, submission("alice"))
    (tmp_path / "README.md").write_text("hi\n")
    files = [rel, "README.md", ".github/workflows/validate.yml"]
    problems = check(tmp_path, files, "alice", set())
    rule = (
        "outside lesson<N>/<student>/<run_id>.json; a submission changes nothing else"
    )
    assert problems == [
        f"README.md: {rule}",
        f".github/workflows/validate.yml: {rule}",
    ]


def test_deletion_in_another_students_directory(tmp_path, submission):
    rel = submission("bob").relpath()  # the file is gone from the changed tree
    problems = check(tmp_path, [rel], "alice", set())
    assert problems == [f"{rel}: deleted by alice, but this directory belongs to bob"]


def test_deleting_own_file_passes(tmp_path, submission):
    rel = submission("alice").relpath()
    assert check(tmp_path, [rel], "alice", set()) == []


def test_symlink_is_refused(tmp_path, submission):
    sub = submission("alice")
    (tmp_path / "secret.txt").write_text("x")
    path = tmp_path / sub.relpath()
    path.parent.mkdir(parents=True)
    path.symlink_to(tmp_path / "secret.txt")
    [problem] = check(tmp_path, [sub.relpath()], "alice", set())
    assert "a symbolic link" in problem


def test_whole_repository_without_files(tmp_path, submission):
    write(tmp_path, submission("alice"))
    write(tmp_path, submission("bob"))
    assert check(tmp_path, None, None, set()) == []


@pytest.mark.parametrize(
    "bad",
    [
        "lesson1/alice/{id}.json\u2028README.md",
        "lesson1/alice/{id}.json\r",
        "\nlesson1/alice/{id}.json",
        "lesson1/alice/{id}.json\n",
        "lesson1/alice_/{id}.json",
        "lesson1/alice/{id}.JSON",
    ],
)
def test_odd_paths_are_refused_first(tmp_path, submission, bad):
    sub = submission("alice")
    write(tmp_path, sub)
    rel = bad.format(id=sub.run.run_id)
    [problem] = check(tmp_path, [rel], "alice", set())
    assert "outside lesson<N>/<student>/<run_id>.json" in problem
    assert not any(c in problem for c in "\n\r\u2028")


def test_files_from_is_nul_separated(tmp_path, submission):
    a = submission("alice").relpath()
    listing = tmp_path / "changed.z"
    listing.write_bytes(f"{a}\0README.md\u2028{a}\0".encode())
    assert read_files(None, listing) == [a, f"README.md\u2028{a}"]
    assert read_files([f"{a}\nREADME.md"], None) == [a, "README.md"]
    assert read_files([f"{a}\u2028b"], None) == [f"{a}\u2028b"]


def test_owner_must_match_the_author_exactly(tmp_path, submission):
    rel = write(tmp_path, submission("alice"))
    problems = check(tmp_path, [rel], "Alice", set())
    assert problems == [f"{rel}: written by Alice, but this directory belongs to alice"]


def test_copy_under_a_new_run_id_is_caught(tmp_path, submission):
    original = submission("bob")
    write(tmp_path, original)
    run = original.run.model_copy(update={"run_id": "f" * 32, "run_name": "mine"})
    copy = original.model_copy(update={"student": "alice", "run": run})
    rel = write(tmp_path, copy)
    problems = check(tmp_path, [rel], "alice", set())
    assert problems == [
        f"{rel}: this run is already submitted by bob, under another run id"
    ]


def invalid(tmp_path, sub, **run_fields) -> list[str]:
    data = json.loads(sub.dump())
    data["run"].update(run_fields)
    path = tmp_path / sub.relpath()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return check(tmp_path, [sub.relpath()], sub.student, set())


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("params", {"bad\nkey": "1"}, "param key"),
        ("params", {"k" * 251: "1"}, "param key"),
        ("params", {"lr": "x" * 6001}, "limit 6000"),
        ("tags", {"role": "x" * 5001}, "limit 5000"),
        ("tags", {"mlflow.user": "alice"}, "mlflow's own tags"),
        ("tags", {"a;b": "1"}, "tag key"),
        ("metrics", {"acc\u2028": []}, "metric key"),
        ("experiment", "Lesson1-notebook", "lesson<N>-<lowercase-name>"),
        ("experiment", "lesson1-notebook\n", "lesson<N>-<lowercase-name>"),
        ("artifacts", {"/etc/x.json": {}}, "must be relative"),
        ("artifacts", {"a/../x.json": {}}, "must be relative"),
    ],
)
def test_content_limits(tmp_path, submission, field, value, expected):
    problems = invalid(tmp_path, submission("alice"), **{field: value})
    assert any(expected in p for p in problems), problems


def test_kept_mlflow_tags_pass(tmp_path, submission):
    tags = {"mlflow.note.content": "n", "mlflow.parentRunId": "p", "role": "trial"}
    assert invalid(tmp_path, submission("alice"), tags=tags) == []


def test_builds_on_classmates_runs_passes(tmp_path, submission):
    refs = f"lesson1/bob/{'a' * 32},lesson2/Carol-1/{'b' * 32}"
    tags = {"role": "final", "builds_on": refs}
    assert invalid(tmp_path, submission("alice"), tags=tags) == []


@pytest.mark.parametrize(
    "value",
    [
        f"lesson1/bob/{'a' * 32}.json",
        f"lesson1/bob/{'a' * 32}, lesson1/carol/{'b' * 32}",
        f"lesson1/bob/{'a' * 32},",
        f"lesson1/bob_/{'a' * 32}",
        f"lesson1/bob/{'A' * 32}",
        f"lessonX/bob/{'a' * 32}",
        f"lesson1/bob/{'a' * 32}\n",
        "",
    ],
)
def test_malformed_builds_on_is_refused(tmp_path, submission, value):
    problems = invalid(tmp_path, submission("alice"), tags={"builds_on": value})
    assert any("is not a submission_ref" in p for p in problems), problems


def test_builds_on_itself_is_refused(tmp_path, submission):
    sub = submission("alice")
    tags = {
        "builds_on": f"lesson1/bob/{'a' * 32},{sub.relpath().removesuffix('.json')}"
    }
    [problem] = invalid(tmp_path, sub, tags=tags)
    assert problem.startswith(f"{sub.relpath()}: ")
    assert problem.endswith("tag 'builds_on' names this run itself")


def test_design_script_is_an_allowed_artifact(tmp_path, submission):
    design = "import torch\n\nmodel = torch.nn.Linear(4, 2)\n"
    assert invalid(tmp_path, submission("alice"), artifacts={"design.py": design}) == []
