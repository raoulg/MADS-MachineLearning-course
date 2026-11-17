"""The plan, the edit of study.py and the guards of `scripts/class_rehearsal.py`; nothing
here trains or downloads."""

import dataclasses
import importlib.util
import subprocess
from pathlib import Path

import class_rehearsal as rehearsal  # ty: ignore[import-not-found]  (scripts/ is on sys.path, see conftest.py)
import pytest
from gitworld import FAKE_GH, gitconfig  # ty: ignore[import-not-found]

from hpo.search import Choice, Float, Int

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / rehearsal.STUDY_FILE


def test_the_plan_is_a_dag_whose_rounds_name_earlier_parents():
    plan = rehearsal.PLAN
    rehearsal.check_plan(plan)
    by_key = {p.key: p for p in plan}
    assert sorted(by_key) == ["a1", "b1", "b2", "c1", "c2", "d1", "h0"]
    assert sorted({p.student for p in plan if p.round > 0}) == ["a", "b", "c", "d"]
    edges = sorted((parent, p.key) for p in plan for parent in p.builds_on)
    assert edges == [
        ("a1", "b1"),
        ("b1", "c2"),
        ("b2", "c1"),
        ("c1", "c2"),
        ("c1", "d1"),
        ("h0", "a1"),
        ("h0", "b2"),
    ]
    for p in plan:
        assert all(by_key[parent].round < p.round for parent in p.builds_on)


def test_round_0_is_the_teacher_publishing_what_the_lesson_hands_out():
    [h0] = [p for p in rehearsal.PLAN if p.round == 0]
    assert h0.handed_out and h0.student == rehearsal.TEACHER and not h0.builds_on
    assert h0.hands_in == "lesson1-notebook*"
    assert {p.hands_in for p in rehearsal.PLAN if p.round > 0} == {"lesson1-study*"}
    assert [p.key for p in rehearsal.PLAN if p.round == 1] == ["a1", "b2"]
    assert all(p.builds_on == ("h0",) for p in rehearsal.PLAN if p.round == 1)


@pytest.mark.parametrize(
    ("key", "change", "says"),
    [
        ("b1", {"builds_on": ("c1",)}, "a parent is of an earlier round"),  # same round
        ("a1", {"builds_on": ("d1",)}, "a parent is of an earlier round"),  # later
        ("b1", {"builds_on": ("zz",)}, "not planned"),
        ("a1", {"builds_on": ()}, "builds on nothing"),
        ("h0", {"builds_on": ("a1",)}, "round 0 is the teacher's"),
        ("a1", {"round": 0}, "round 0 is the teacher's"),
    ],
)
def test_a_plan_that_names_no_earlier_study_is_refused(key, change, says):
    plan = tuple(
        dataclasses.replace(p, **change) if p.key == key else p for p in rehearsal.PLAN
    )
    with pytest.raises(rehearsal.Stop, match=says):
        rehearsal.check_plan(plan)


def test_a_child_searches_what_its_parents_changed_and_its_own():
    handed_out = rehearsal.handed_out_space(STUDY.read_text())
    c2 = rehearsal.space_of("c2", rehearsal.PLAN, handed_out)
    assert c2["n_hidden_layers"] == "Int(1, 3)"  # from a1, through b1
    assert c2["dropout"] == "Float(0.0, 0.7)"  # from c1
    assert (c2["skip"], c2["gate"]) == ("True", "True")
    assert c2["epochs"] == "EPOCHS"  # as handed out
    for p in rehearsal.PLAN:
        assert rehearsal.space_of(p.key, rehearsal.PLAN, handed_out).keys() == (
            handed_out.keys()
        )


def load_study(path: Path, monkeypatch):
    # the placeholder login of notebooktester, so importing never asks gh
    monkeypatch.setenv("NOTEBOOKTESTER_RUNNING", "1")
    monkeypatch.syspath_prepend(str(STUDY.parent))
    spec = importlib.util.spec_from_file_location("rehearsal_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "builds_on", [(), ("blissful-bee-18",), ("blissful-bee-18", "zany-heron-41")]
)
def test_the_edit_of_study_py_imports_and_carries_the_answers(
    builds_on, tmp_path, monkeypatch
):
    text = STUDY.read_text()
    space = rehearsal.handed_out_space(text) | {
        "hidden_dim": "Int(256, 2048, log=True)",
        "dropout": "Float(0.0, 0.7)",
        "norm": "Choice(['none', 'layer'])",
        "gate": "True",
    }
    answers = rehearsal.Answers(
        study="c-skip-and-gate",
        idea="skip and gate together on a wide head",
        builds_on=builds_on,
        expects="both together beat either alone",
        space=space,
        minutes=1.5,
    )
    path = tmp_path / "study.py"
    path.write_text(rehearsal.edit_study(text, answers))
    study = load_study(path, monkeypatch)

    assert study.STUDY == "c-skip-and-gate"
    assert study.IDEA.words == "skip and gate together on a wide head"
    handed_out = study.Idea.handed_out(1).name
    assert study.IDEA.parents == (tuple(sorted(builds_on)) or (handed_out,))
    assert study.EXPECTS == "both together beat either alone"
    assert study.MINUTES == 1.5
    assert study.SPACE["hidden_dim"] == Int(256, 2048, log=True)
    assert study.SPACE["dropout"] == Float(0.0, 0.7)
    assert study.SPACE["norm"] == Choice(["none", "layer"])
    assert study.SPACE["gate"] is True
    assert list(study.SPACE) == list(space)
    # the questions above the answers stay
    assert "# What do you expect to happen, and why?" in path.read_text()


def test_an_edit_of_a_study_py_without_an_answer_is_refused():
    text = STUDY.read_text().replace("MINUTES = ", "BUDGET = ")
    answers = rehearsal.Answers("a-x", "words of an idea", (), "expects", {}, 1)
    with pytest.raises(rehearsal.Stop, match="MINUTES"):
        rehearsal.edit_study(text, answers)


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def test_the_guard_refuses_a_real_gh(tmp_path):
    fake = executable(tmp_path / "sandbox/bin/gh", FAKE_GH)
    real = executable(tmp_path / "elsewhere/gh", "#!/bin/sh\necho real\n")
    rehearsal.check_gh(f"{fake.parent}:{real.parent}", fake)
    with pytest.raises(rehearsal.Stop, match="not the fake"):
        rehearsal.check_gh(f"{real.parent}:{fake.parent}", fake)
    with pytest.raises(rehearsal.Stop, match="not the fake"):
        rehearsal.check_gh(str(tmp_path / "empty"), fake)


def git_env(config: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(config.parent),
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def test_the_guard_refuses_a_rewrite_outside_the_sandbox(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "gitconfig").write_text(gitconfig(sandbox))
    rehearsal.check_rewrite(git_env(sandbox / "gitconfig"), sandbox)

    (tmp_path / "gitconfig").write_text(gitconfig(sandbox))
    with pytest.raises(rehearsal.Stop, match="not a file inside"):
        rehearsal.check_rewrite(git_env(tmp_path / "gitconfig"), sandbox)

    (sandbox / "gitconfig").write_text("[user]\n  name = x\n")
    with pytest.raises(rehearsal.Stop, match="does not rewrite"):
        rehearsal.check_rewrite(git_env(sandbox / "gitconfig"), sandbox)


def test_the_guard_refuses_a_remote_on_github(tmp_path):
    sandbox = tmp_path / "sandbox"
    repo = sandbox / "students/a/mads-runs"
    repo.mkdir(parents=True)
    config = sandbox / "gitconfig"
    config.write_text("[user]\n  name = x\n")
    env = git_env(config)
    subprocess.run(["git", "init", "--quiet", str(repo)], env=env, check=True)
    url = "https://github.com/a/mads-ml-runs-2026-11.git"
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", url], env=env, check=True
    )
    assert rehearsal.repositories(sandbox) == [repo]
    with pytest.raises(rehearsal.Stop, match="resolves to https://github.com"):
        rehearsal.check_remotes([repo], env)

    config.write_text(gitconfig(sandbox))
    rehearsal.check_remotes([repo], env)


def test_a_step_output_is_read_as_github_writes_it(tmp_path):
    output = tmp_path / "out"
    output.write_text("problems<<EOF_1\nline one\nline=two\nEOF_1\nstatus=1\n")
    assert rehearsal.outputs_of(output) == {
        "problems": "line one\nline=two",
        "status": "1",
    }
