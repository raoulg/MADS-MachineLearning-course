"""The plan, the edit of study.py and the guards of `scripts/class_rehearsal.py`; nothing
here trains or downloads."""

import dataclasses
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import class_rehearsal as rehearsal  # ty: ignore[import-not-found]  (scripts/ is on sys.path, see conftest.py)
import pytest
import submit_runs  # ty: ignore[import-not-found]
from gitworld import FAKE_GH, gitconfig, make_world  # ty: ignore[import-not-found]
from run_submission import (  # ty: ignore[import-not-found]
    MAX_CHANGED_FILES,
    idea_id,
    idea_name,
)

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
    text = rehearsal.sandbox_gitconfig(sandbox, ["a", "b"])
    (sandbox / "gitconfig").write_text(text)
    rehearsal.check_rewrite(git_env(sandbox / "gitconfig"), sandbox)

    with_system = git_env(sandbox / "gitconfig")
    del with_system["GIT_CONFIG_NOSYSTEM"]
    with pytest.raises(rehearsal.Stop, match="GIT_CONFIG_NOSYSTEM"):
        rehearsal.check_rewrite(with_system, sandbox)

    (tmp_path / "gitconfig").write_text(text)
    with pytest.raises(rehearsal.Stop, match="not a file inside"):
        rehearsal.check_rewrite(git_env(tmp_path / "gitconfig"), sandbox)

    (sandbox / "gitconfig").write_text(gitconfig(sandbox))  # rewrites, allows https
    with pytest.raises(rehearsal.Stop, match="lets git use https"):
        rehearsal.check_rewrite(git_env(sandbox / "gitconfig"), sandbox)

    (sandbox / "gitconfig").write_text("[user]\n  name = x\n")
    with pytest.raises(rehearsal.Stop, match="does not rewrite"):
        rehearsal.check_rewrite(git_env(sandbox / "gitconfig"), sandbox)


def test_a_remote_on_github_passes_only_where_git_cannot_fetch_it(tmp_path):
    sandbox = tmp_path / "sandbox"
    repo = sandbox / "students/a/course"
    repo.mkdir(parents=True)
    config = sandbox / "gitconfig"
    config.write_text("[user]\n  name = x\n")
    env = git_env(config)
    subprocess.run(["git", "init", "--quiet", str(repo)], env=env, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", rehearsal.COURSE_REPO],
        env=env, check=True,
    )  # fmt: skip
    assert rehearsal.repositories(sandbox) == [repo]
    with pytest.raises(rehearsal.Stop, match="lets git use https"):
        rehearsal.check_remotes([repo], env)

    config.write_text(rehearsal.sandbox_gitconfig(sandbox, ["a"]))
    rehearsal.check_remotes([repo], env)
    # a normal shell reads none of the sandbox's config, until the clone holds it
    with pytest.raises(rehearsal.Stop, match="lets git use https"):
        rehearsal.check_remotes([repo], rehearsal.plain_env(env))
    rehearsal.pin(repo, config, env)
    rehearsal.check_remotes([repo], rehearsal.plain_env(env))


def test_a_pinned_fork_clone_reaches_the_sandbox_in_a_plain_shell(tmp_path):
    sandbox = tmp_path / "sandbox"
    repo = sandbox / "students/a/mads-runs"
    repo.mkdir(parents=True)
    config = sandbox / "gitconfig"
    config.write_text(rehearsal.sandbox_gitconfig(sandbox, ["a"]))
    env = git_env(config)
    subprocess.run(["git", "init", "--quiet", str(repo)], env=env, check=True)
    url = "https://github.com/a/mads-ml-runs-2026-11.git"
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", url], env=env, check=True
    )
    rehearsal.pin(repo, config, env)
    plain = rehearsal.plain_env(env)
    got = rehearsal.git_out(["ls-remote", "--get-url", "origin"], repo, plain)
    assert got == f"file://{sandbox}/gh-root/a/mads-ml-runs-2026-11.git"


def test_the_sandbox_env_finds_the_fake_gh_before_a_real_one(tmp_path, monkeypatch):
    real = executable(tmp_path / "elsewhere/gh", "#!/bin/sh\necho real\n")
    monkeypatch.setenv("PATH", f"{real.parent}:/usr/bin:/bin")
    r = rehearsal.Rehearsal(tmp_path / "sandbox", tmp_path / "studies.db", 1)
    executable(r.fake, FAKE_GH)
    rehearsal.check_gh(r.env("a")["PATH"], r.fake)


def with_idea_on(sub, parent: str):
    """`sub` with a study idea of its student that builds on `parent`."""
    words = "an idea on something unpublished"
    id_ = idea_id(sub.student, words, [parent])
    attrs = {
        **sub.run.study.attrs,
        "idea": words, "idea_author": sub.student, "idea_parents": parent,
        "idea_id": id_, "idea_name": idea_name(id_),
    }  # fmt: skip
    study = sub.run.study.model_copy(update={"attrs": attrs})
    return sub.model_copy(update={"run": sub.run.model_copy(update={"study": study})})


@pytest.fixture
def rehearsed(tmp_path, monkeypatch):
    """A rehearsal over a git world whose class repository is the rehearsal's, its
    validator and workflow those of this checkout's commit."""
    world = make_world(tmp_path, monkeypatch, upstream=rehearsal.CLASS_REPO)
    r = rehearsal.Rehearsal(tmp_path, tmp_path / "studies.db", 1)
    r.world = world
    r.head = rehearsal.git_out(["rev-parse", "HEAD"], ROOT, dict(os.environ))
    return r


def pull_requests(r) -> list[dict]:
    return json.loads((r.sandbox / "gh-prs.json").read_text())


def test_the_check_merges_a_valid_pull_request_and_stops_an_invalid_one(
    rehearsed, submission
):
    world = rehearsed.world
    good = submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [good])
    [pr] = pull_requests(rehearsed)
    rehearsed.check_and_merge(pr["url"], "good")
    assert world.on_upstream() == [good.relpath()]
    assert pull_requests(rehearsed)[0]["state"] == "merged"

    bad = with_idea_on(submission("alice"), "missing-idea-01")
    submit_runs.submit(world.clone, 1, "alice", [good, bad])
    url = pull_requests(rehearsed)[-1]["url"]
    with pytest.raises(rehearsal.Stop, match="missing-idea-01, which is not in the"):
        rehearsed.check_and_merge(url, "bad")
    assert world.on_upstream() == [good.relpath()]
    assert pull_requests(rehearsed)[-1]["state"] == "open"


@pytest.mark.parametrize(
    "change", [{"base": "develop"}, {"repo": "someone/mads-ml-runs-2026-11"}]
)
def test_a_pull_request_off_main_of_the_class_repository_is_refused(
    rehearsed, submission, change
):
    submit_runs.submit(rehearsed.world.clone, 1, "alice", [submission("alice")])
    [pr] = pull_requests(rehearsed)
    (rehearsed.sandbox / "gh-prs.json").write_text(json.dumps([{**pr, **change}]))
    with pytest.raises(rehearsal.Stop, match="not raoulg/mads-ml-runs-2026-11:main"):
        rehearsed.check_and_merge(pr["url"], "x")
    assert rehearsed.world.on_upstream() == []


def outcome(key: str, idea: str, parents: tuple[str, ...]):
    planned = next(p for p in rehearsal.PLAN if p.key == key)
    return rehearsal.Outcome(planned, parents, idea)


def test_the_closing_check_stops_when_the_parents_differ_from_the_plan():
    outcomes = {
        "h0": outcome("h0", "root-idea-01", ()),
        "a1": outcome("a1", "first-idea-02", ("root-idea-01",)),
    }
    held = [
        ("root-idea-01", "course", "lesson 1 as handed out", []),
        ("first-idea-02", "a", "words", ["root-idea-01"]),
    ]
    rehearsal.check_ideas(outcomes, held)
    with pytest.raises(rehearsal.Stop, match="differ from the plan"):
        rehearsal.check_ideas(outcomes, [held[0], (*held[1][:3], [])])
    with pytest.raises(rehearsal.Stop, match="differ from the plan"):
        rehearsal.check_ideas(outcomes, held[:1])


def test_a_submitted_path_of_this_machine_is_found():
    texts = {
        "lesson1/a/x.json": '{"remote": "https://github.com/raoulg/x.git"}',
        "lesson1/a/y.json": f'{{"remote": "{Path.home()}/code/course"}}',
        "lesson1/a/z.json": '{"remote": "/Users/someone/course"}',
    }
    assert rehearsal.local_paths(texts) == ["lesson1/a/y.json", "lesson1/a/z.json"]


def test_a_study_planned_over_the_check_s_file_limit_is_warned_about():
    seconds, repeats = 3.25, 3
    short = rehearsal.run_files(1, seconds, repeats)
    long = rehearsal.run_files(15, seconds, repeats)
    assert short < MAX_CHANGED_FILES < long
    assert rehearsal.file_warnings(rehearsal.PLAN, short, 128) == {}
    warned = rehearsal.file_warnings(rehearsal.PLAN, long, 128)
    assert sorted(warned) == ["a1", "b1", "b2", "c1", "c2", "d1"]
    assert str(MAX_CHANGED_FILES) in warned["a1"]
    assert list(rehearsal.file_warnings(rehearsal.PLAN, short, 500)) == ["h0"]


def test_a_step_output_is_read_as_github_writes_it(tmp_path):
    output = tmp_path / "out"
    output.write_text("problems<<EOF_1\nline one\nline=two\nEOF_1\nstatus=1\n")
    assert rehearsal.outputs_of(output) == {
        "problems": "line one\nline=two",
        "status": "1",
    }


def test_the_teachers_class_page_links_the_course_start_page(tmp_path):
    page = tmp_path / "dashboard" / "index.html"
    command = rehearsal.teacher_page_command(page)
    assert command[:3] == ["uv", "run", "scripts/class_analysis.py"]
    assert command[command.index("--out") + 1] == str(page)
    start = command[command.index("--course-link") + 1]
    assert start == rehearsal.COURSE_START
    assert (ROOT / "presentations" / start.rsplit("/", 1)[-1]).exists()
