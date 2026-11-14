"""collect_runs against the git world of `gitworld.py`: it reads the class repository's
main from a git ref, whatever branch the clone is on."""

import sys
from pathlib import Path

import collect_runs  # ty: ignore[import-not-found]
import pytest
import submit_runs  # ty: ignore[import-not-found]
from gitworld import run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def collect(monkeypatch, repo: Path, uri: str, *extra: str) -> None:
    argv = ["collect_runs.py", "--runs-repo", str(repo), "--tracking-uri", uri]
    monkeypatch.setattr(sys, "argv", [*argv, *extra])
    collect_runs.main()


def imported(uri: str) -> dict[str, dict[str, str]]:
    """submission_ref -> tags of every run in the class store."""
    client = MlflowClient(tracking_uri=uri)
    runs = client.search_runs([e.experiment_id for e in client.search_experiments()])
    return {r.data.tags["submission_ref"]: r.data.tags for r in runs}


def ref(sub) -> str:
    return sub.relpath().removesuffix(".json")


def state(repo: Path) -> tuple[str, str, str]:
    """The branch, the commit and the working tree's status of a clone."""
    return (
        run("git", "branch", "--show-current", cwd=repo),
        run("git", "rev-parse", "HEAD", cwd=repo),
        run("git", "status", "--porcelain", "--untracked-files=all", cwd=repo),
    )


def teacher_clone(world) -> Path:
    """The teacher's clone of the class repository: only `origin`."""
    where = world.tmp / "class-clone"
    run("git", "clone", "--quiet", str(world.upstream), str(where))
    return where


def test_a_students_clone_on_its_lesson_branch_reads_upstream_main(
    world, submission, tmp_path, monkeypatch
):
    merged, pending = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [merged])
    world.squash_merge()
    bob = submission("bob")
    world.teacher_push(bob)
    submit_runs.submit(world.clone, 1, "alice", [merged, pending])
    carol = submission("carol")
    world.teacher_push(carol)  # after the clone last fetched: collect must fetch it
    (world.clone / "notes.txt").write_text("mine\n")
    before = state(world.clone)
    assert before[0] == "lesson1"

    uri = f"sqlite:///{tmp_path / 'class.db'}"
    collect(monkeypatch, world.clone, uri, "--lesson", "1")
    # the fork's main has no runs and lesson1 holds the pending one: neither is read
    assert sorted(imported(uri)) == sorted([ref(merged), ref(bob), ref(carol)])
    assert imported(uri)[ref(bob)]["student"] == "bob"
    assert state(world.clone) == before


def test_the_teachers_clone_with_only_origin(world, submission, tmp_path, monkeypatch):
    first = submission("alice")
    world.teacher_push(first)
    clone = teacher_clone(world)
    run("git", "switch", "--quiet", "-c", "notes", cwd=clone)
    later = submission("bob")
    world.teacher_push(later)
    before = state(clone)

    uri = f"sqlite:///{tmp_path / 'class.db'}"
    collect(monkeypatch, clone, uri, "--no-pull")
    assert sorted(imported(uri)) == [ref(first)]
    collect(monkeypatch, clone, uri)
    assert sorted(imported(uri)) == sorted([ref(first), ref(later)])
    assert state(clone) == before
    assert not (clone / later.relpath()).exists()


def test_the_design_script_lands_as_design_py(world, submission, tmp_path, monkeypatch):
    sub = submission("alice")
    design = "import torch\n\nmodel = torch.nn.Linear(4, 2)\n"
    run_ = sub.run.model_copy(update={"artifacts": {"design.py": design}})
    sub = sub.model_copy(update={"run": run_})
    world.teacher_push(sub)

    uri = f"sqlite:///{tmp_path / 'class.db'}"
    collect(monkeypatch, teacher_clone(world), uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("lesson1-notebook")
    assert exp is not None
    [only] = client.search_runs([exp.experiment_id])
    assert only.data.tags["submission_ref"] == ref(sub)
    run_id = only.info.run_id
    [artifact] = client.list_artifacts(run_id)
    assert artifact.path == "design.py"
    local = client.download_artifacts(run_id, "design.py", str(tmp_path))
    assert Path(local).read_text() == design


def test_a_failed_import_is_reported_removed_and_fails_the_run(
    world, submission, tmp_path, monkeypatch, capsys
):
    good, bad = submission("alice"), submission("bob")
    world.teacher_push(good)
    world.teacher_push(bad)

    fill = collect_runs.fill_run

    def fail_for_bob(client, rid, sub, tags):
        if sub.student == "bob":
            raise MlflowException("disk full")
        fill(client, rid, sub, tags)

    monkeypatch.setattr(collect_runs, "fill_run", fail_for_bob)
    uri = f"sqlite:///{tmp_path / 'class.db'}"
    with pytest.raises(SystemExit) as e:
        collect(monkeypatch, teacher_clone(world), uri)
    assert e.value.code != 0
    out = capsys.readouterr().out
    assert f"failed {bad.relpath()}: disk full" in out
    assert "1 failed" in out and "1 new" in out

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("lesson1-notebook")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert [r.data.tags["student"] for r in runs] == ["alice"]


def test_an_import_without_submission_ref_gets_it_unchanged(
    world, submission, tmp_path, monkeypatch, capsys
):
    bob = submission("bob")
    world.teacher_push(bob)
    clone = teacher_clone(world)
    uri = f"sqlite:///{tmp_path / 'class.db'}"
    collect(monkeypatch, clone, uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("lesson1-notebook")
    assert exp is not None
    [before] = client.search_runs([exp.experiment_id])
    client.delete_tag(before.info.run_id, "submission_ref")
    capsys.readouterr()

    collect(monkeypatch, clone, uri)
    assert "1 unchanged" in capsys.readouterr().out
    [after] = client.search_runs([exp.experiment_id])
    assert after.info.run_id == before.info.run_id
    assert after.data.tags["submission_ref"] == ref(bob)


def test_only_the_lessons_asked_for(world, submission, tmp_path, monkeypatch):
    one, two = submission("alice", lesson=1), submission("bob", lesson=2)
    world.teacher_push(one)
    world.teacher_push(two)
    uri = f"sqlite:///{tmp_path / 'class.db'}"
    collect(monkeypatch, teacher_clone(world), uri, "--lesson", "1")
    assert sorted(imported(uri)) == [ref(one)]


def test_submit_after_collect_on_a_fork_clone(world, submission, tmp_path, monkeypatch):
    first, second = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [first])
    world.squash_merge()
    world.teacher_push(submission("bob"))
    collect(monkeypatch, world.clone, f"sqlite:///{tmp_path / 'class.db'}")

    submit_runs.submit(world.clone, 1, "alice", [first, second])
    assert second.relpath() in world.files_in_pr()
    assert run("git", "branch", "--show-current", cwd=world.clone) == "lesson1"
