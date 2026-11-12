from pathlib import Path

from run_submission import Submission, check  # ty: ignore[import-not-found]


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
    files = f"{rel}\nREADME.md\n.github/workflows/validate.yml"
    problems = check(tmp_path, [files], "alice", set())
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
