import json
from pathlib import Path

import pytest
import run_submission  # ty: ignore[import-not-found]
from pydantic import ValidationError
from run_submission import (  # ty: ignore[import-not-found]
    ADJECTIVES,
    ANIMALS,
    IDEA_NAME,
    Submission,
    check,
    explain,
    idea_id,
    idea_name,
    idea_words,
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
    (tmp_path / rel).write_text('{"schema": "mads-run/2",')
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


FLOAT = {
    "name": "FloatDistribution",
    "attributes": {"low": 0.0001, "high": 0.01, "log": True, "step": None},
}


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("params", {"bad\nkey": 1}, "param key"),
        ("params", {"k" * 251: 1}, "param key"),
        ("params", {"lr": "x" * 20_001}, "limit 20000"),
        ("params", {"lr": [1, 2]}, "params.lr"),
        ("attrs", {"role": "x" * 20_001}, "limit 20000"),
        ("attrs", {"a;b": 1}, "attr key"),
        ("experiment", "Lesson1-notebook", "lesson<N>-<lowercase-name>"),
        ("experiment", "lesson1-notebook\n", "lesson<N>-<lowercase-name>"),
        ("status", "FINISHED", "status"),
        ("value", None, "a COMPLETE trial has a value"),
        ("run_id", "abc", "32-character"),
        ("distributions", {"zz": FLOAT}, "distributions for params"),
        ("params", {"lr": 0.01, "n": 1}, None),
        ("artifacts", {"/etc/x.json": {}}, "must be relative"),
        ("artifacts", {"a/../x.json": {}}, "must be relative"),
        ("artifacts", {"weights.pt": "x"}, "only"),
    ],
)
def test_content_limits(tmp_path, submission, field, value, expected):
    problems = invalid(tmp_path, submission("alice"), **{field: value})
    if expected is None:
        assert problems == []
    else:
        assert any(expected in p for p in problems), problems


def test_a_param_with_its_distribution_passes(tmp_path, submission):
    run = {"params": {"lr": 0.003}, "distributions": {"lr": FLOAT}}
    assert invalid(tmp_path, submission("alice"), **run) == []


@pytest.mark.parametrize(
    "dist",
    [
        {"name": "NormalDistribution", "attributes": {}},
        {"name": "FloatDistribution", "attributes": {"low": 0, "high": 1}},
        {"name": "CategoricalDistribution", "attributes": {"choices": ["a"], "x": 1}},
    ],
)
def test_a_malformed_distribution_is_refused(tmp_path, submission, dist):
    run = {"params": {"lr": 0.003}, "distributions": {"lr": dist}}
    assert invalid(tmp_path, submission("alice"), **run) != []


def test_a_pruned_trial_without_a_value_passes(tmp_path, submission):
    run = {"status": "PRUNED", "value": None}
    assert invalid(tmp_path, submission("alice"), **run) == []


def test_builds_on_classmates_runs_passes(tmp_path, submission):
    refs = f"lesson1/bob/{'a' * 32},lesson2/Carol-1/{'b' * 32}"
    attrs = {"role": "final", "builds_on": refs}
    assert invalid(tmp_path, submission("alice"), attrs=attrs) == []


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
    problems = invalid(tmp_path, submission("alice"), attrs={"builds_on": value})
    assert any("is not a submission_ref" in p for p in problems), problems


def test_builds_on_itself_is_refused(tmp_path, submission):
    sub = submission("alice")
    attrs = {
        "builds_on": f"lesson1/bob/{'a' * 32},{sub.relpath().removesuffix('.json')}"
    }
    [problem] = invalid(tmp_path, sub, attrs=attrs)
    assert problem.startswith(f"{sub.relpath()}: ")
    assert problem.endswith("attr 'builds_on' names this run itself")


def test_builds_on_itself_in_another_case_is_refused(tmp_path, submission):
    sub = submission("alice")
    attrs = {"builds_on": f"lesson1/Alice/{sub.run.run_id}"}
    [problem] = invalid(tmp_path, sub, attrs=attrs)
    assert problem.endswith("attr 'builds_on' names this run itself")


def test_explain_gives_a_model_level_error_without_a_location(submission):
    data = json.loads(submission("alice").dump())
    data["run"]["attrs"]["builds_on"] = f"lesson1/alice/{data['run']['run_id']}"
    with pytest.raises(ValidationError) as e:
        Submission.model_validate(data)
    assert explain(e.value) == ["Value error, attr 'builds_on' names this run itself"]


def test_design_script_is_an_allowed_artifact(tmp_path, submission):
    design = "import torch\n\nmodel = torch.nn.Linear(4, 2)\n"
    assert invalid(tmp_path, submission("alice"), artifacts={"design.py": design}) == []


def with_idea(sub, words, parents=(), author=None, experiment=None):
    """`sub` with a study that carries the idea `words`, built on `parents`."""
    author = author or sub.student.lower()
    id_ = idea_id(author, words, list(parents))
    data = json.loads(sub.dump())
    data["run"]["study"]["attrs"].update(
        idea=idea_words(words),
        idea_author=author,
        idea_parents=",".join(sorted(parents)),
        idea_id=id_,
        idea_name=run_submission.idea_name(id_),
    )
    if experiment:
        data["run"]["experiment"] = experiment
    return Submission.model_validate(data)


def study_attrs(tmp_path, sub, **attrs) -> list[str]:
    """The problems of `sub` after its study attrs are changed by hand."""
    data = json.loads(sub.dump())
    for key, value in attrs.items():
        if value is None:
            del data["run"]["study"]["attrs"][key]
        else:
            data["run"]["study"]["attrs"][key] = value
    path = tmp_path / sub.relpath()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return check(tmp_path, [sub.relpath()], sub.student, set())


def test_idea_id_holds_author_words_and_parents():
    base = idea_id("alice", "does depth help on eurosat", [])
    assert base == idea_id("Alice", "Does  depth help\non EuroSAT", [])
    assert base != idea_id("bob", "does depth help on eurosat", [])
    assert base != idea_id("alice", "does width help on eurosat", [])
    assert base != idea_id("alice", "does depth help on eurosat", ["zany-heron-41"])
    assert idea_id("alice", "x", ["b-b-02", "a-a-01"]) == idea_id(
        "alice", "x", ["a-a-01", "b-b-02"]
    )


def test_idea_name_reads_three_bytes_of_the_id():
    assert idea_name("000000" + "f" * 58) == f"{ADJECTIVES[0]}-{ANIMALS[0]}-00"
    assert idea_name("01ffc8" + "0" * 58) == f"{ADJECTIVES[1]}-{ANIMALS[255]}-00"
    assert len(set(ADJECTIVES)) == 256 and len(set(ANIMALS)) == 256
    for word in (*ADJECTIVES, *ANIMALS):
        assert IDEA_NAME.fullmatch(f"{word}-{word}-07")


def test_a_study_with_its_idea_passes(tmp_path, submission):
    rel = write(tmp_path, with_idea(submission("Alice"), "does depth help on eurosat"))
    assert check(tmp_path, [rel], "Alice", set()) == []


def test_an_idea_builds_on_a_published_idea(tmp_path, submission):
    parent = with_idea(submission("bob"), "norm in the hidden block")
    write(tmp_path, parent)
    child = with_idea(
        submission("alice"), "depth and norm together", [parent.idea()[0]]
    )
    rel = write(tmp_path, child)
    assert check(tmp_path, [rel], "alice", set()) == []


def test_an_idea_on_two_parents_in_one_pull_request(tmp_path, submission):
    first = with_idea(submission("alice"), "depth on eurosat")
    second = with_idea(submission("alice"), "norm in the hidden block")
    names = [first.idea()[0], second.idea()[0]]
    merged = with_idea(submission("alice"), "depth and norm together", names)
    rels = [write(tmp_path, sub) for sub in (first, second, merged)]
    assert check(tmp_path, rels, "alice", set()) == []


def test_an_unpublished_parent_is_refused(tmp_path, submission):
    child = with_idea(submission("alice"), "depth and norm together", ["zany-heron-41"])
    rel = write(tmp_path, child)
    problems = check(tmp_path, [rel], "alice", set())
    assert len(problems) == 1
    assert "builds on the idea zany-heron-41, which is not in the class" in problems[0]


def test_another_students_idea_is_refused(submission):
    with pytest.raises(ValidationError, match="'idea_author' is 'bob'"):
        with_idea(submission("alice"), "does depth help on eurosat", author="bob")


def test_the_course_is_the_author_of_a_handed_out_study_only(submission):
    handed_out = with_idea(
        submission("alice"), "lesson 1 as handed out", author="course"
    )
    assert handed_out.run.experiment == "lesson1-notebook-runs"
    with pytest.raises(ValidationError, match="'idea_author' is 'course'"):
        with_idea(
            submission("alice"),
            "lesson 1 as handed out",
            author="course",
            experiment="lesson1-study-alice-depth",
        )


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({"idea": "does width help on eurosat"}, "'idea_id' is not the id"),
        ({"idea_parents": "zany-heron-41"}, "'idea_id' is not the id"),
        ({"idea_name": "zany-heron-41"}, "'idea_name' is not the name"),
        ({"idea_id": None}, "missing ['idea_id']"),
        ({"idea": "Does depth help"}, "in lowercase with single spaces"),
        ({"idea": "short"}, "8 to 120 characters"),
        ({"idea_parents": "Zany-Heron-41"}, "is not the name of an idea"),
        ({"idea_parents": "zany-heron-41\n"}, "is not the name of an idea"),
        ({"idea_parents": "b-b-02,a-a-01"}, "sorted, each name once"),
        ({"idea_author": "Alice"}, "a GitHub login in lowercase"),
        ({"idea_name": 7}, "are text"),
    ],
)
def test_an_idea_edited_by_hand_is_refused(tmp_path, submission, attrs, expected):
    sub = with_idea(submission("alice"), "does depth help on eurosat")
    problems = study_attrs(tmp_path, sub, **attrs)
    assert len(problems) == 1 and expected in problems[0]


def test_one_name_stands_for_one_idea(tmp_path, submission, monkeypatch):
    monkeypatch.setattr("run_submission.idea_name", lambda id_: "zany-heron-41")
    write(tmp_path, with_idea(submission("bob"), "norm in the hidden block"))
    rel = write(tmp_path, with_idea(submission("alice"), "depth on eurosat"))
    problems = check(tmp_path, [rel], "alice", set())
    assert len(problems) == 1
    assert "zany-heron-41 already stands for another idea" in problems[0]


LOG_RANGE = {"low": 0.001, "high": 0.1, "log": True, "step": None}


@pytest.mark.parametrize(
    "run_fields, expected",
    [
        ({"value": float("nan")}, "value"),
        ({"value": float("inf")}, "value"),
        ({"intermediate": [{"step": 10**30, "value": 0.5}]}, "intermediate.0.step"),
        ({"intermediate": [{"step": 0, "value": float("nan")}]}, "intermediate.0"),
        ({"start_time": -1}, "start_time"),
        ({"end_time": 2**63}, "end_time"),
        ({"params": {"lr": 10**400}}, "param 'lr' is a finite number"),
        ({"artifacts": {"notes.txt": {"a": 1}}}, "a file that is not .json is text"),
        ({"artifacts": {"C:evil.txt": "x"}}, "only letters, digits"),
        ({"artifacts": {"a b.txt": "x"}}, "only letters, digits"),
        ({"study": {"direction": "minimize", "attrs": {}}}, "maximizes"),
        (
            {
                "distributions": {
                    "lr": {"name": "FloatDistribution", "attributes": LOG_RANGE}
                }
            },
            "",
        ),
        (
            {
                "params": {"lr": 5.0},
                "distributions": {
                    "lr": {"name": "FloatDistribution", "attributes": LOG_RANGE}
                },
            },
            "param 'lr' is not a value of the distribution",
        ),
        (
            {
                "params": {"norm": "group"},
                "distributions": {
                    "norm": {
                        "name": "CategoricalDistribution",
                        "attributes": {"choices": ["none", "batch"]},
                    }
                },
            },
            "param 'norm' is not a value of the distribution",
        ),
    ],
)
def test_what_optuna_would_refuse_is_refused(
    tmp_path, submission, run_fields, expected
):
    problems = invalid(tmp_path, submission("alice"), **run_fields)
    if not expected:
        assert problems == []
    else:
        assert problems and all(expected in p for p in problems), problems


@pytest.mark.parametrize(
    "attributes",
    [
        {"low": "x", "high": [1], "log": "no", "step": {}},
        {"low": 0.1, "high": 0.001, "log": False, "step": None},
        {"low": 0.0, "high": 1.0, "log": True, "step": None},
        {"low": 0.1, "high": 1.0, "log": True, "step": 0.1},
        {"low": 0.1, "high": float("inf"), "log": False, "step": None},
        {"low": 0.1, "high": 1.0, "log": False, "step": 0},
    ],
)
def test_a_range_optuna_cannot_build_is_refused(tmp_path, submission, attributes):
    dist = {"lr": {"name": "FloatDistribution", "attributes": attributes}}
    assert invalid(tmp_path, submission("alice"), distributions=dist)


def test_an_int_param_follows_its_step(tmp_path, submission):
    attributes = {"low": 32, "high": 128, "log": False, "step": 32}
    dist = {"width": {"name": "IntDistribution", "attributes": attributes}}
    sub = submission("alice")
    assert invalid(tmp_path, sub, params={"width": 96}, distributions=dist) == []
    assert invalid(tmp_path, sub, params={"width": 100}, distributions=dist)
    assert invalid(tmp_path, sub, params={"width": 96.0}, distributions=dist)


def raw_problems(tmp_path, sub, text: str) -> list[str]:
    path = tmp_path / sub.relpath()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return check(tmp_path, [sub.relpath()], sub.student, set())


def test_a_key_that_is_there_twice_is_refused(tmp_path, submission):
    sub = submission("alice")
    text = sub.dump().replace('"lesson": 1,', '"lesson": 1, "lesson": 1,', 1)
    problems = raw_problems(tmp_path, sub, text)
    assert len(problems) == 1 and "a key is there twice" in problems[0]


def test_a_lone_surrogate_is_refused(tmp_path, submission):
    sub = submission("alice")
    text = sub.dump().replace('"baseline"', '"\\udfff"', 1)
    problems = raw_problems(tmp_path, sub, text)
    assert len(problems) == 1 and "not valid JSON" in problems[0]


@pytest.mark.parametrize("depth", [300, 100_000])
def test_deep_nesting_is_a_problem_not_a_crash(tmp_path, submission, depth):
    sub = submission("alice")
    nested = "[" * depth + "]" * depth
    text = sub.dump().replace('"dataset": "dtd"', f'"dataset": {nested}', 1)
    problems = raw_problems(tmp_path, sub, text)
    assert len(problems) == 1 and "nested too deep" in problems[0]


def test_a_key_outside_ascii_is_refused(tmp_path, submission):
    problems = invalid(tmp_path, submission("alice"), attrs={"аccuracy": 1})
    assert len(problems) == 1 and "attr key" in problems[0]


def test_a_newline_in_a_key_stays_on_its_line(tmp_path, submission):
    key = "x\n::warning::injected"
    problems = invalid(tmp_path, submission("alice"), params={key: [1]})
    assert problems and all("\n" not in p for p in problems)
