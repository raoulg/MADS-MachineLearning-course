"""The git part of submit_runs, against the git world of `gitworld.py`."""

import json
import sys
from datetime import UTC, datetime

import pytest
import submit_runs  # ty: ignore[import-not-found]
from gitworld import gitconfig, run
from mlflow.tracking import MlflowClient


def test_submit_cycle(world, submission, capsys):
    first = [submission("alice"), submission("alice")]

    submit_runs.submit(world.clone, 1, "alice", first)
    [create] = world.creates()
    assert create[create.index("--repo") + 1] == "class/runs"
    assert create[create.index("--base") + 1] == "main"
    assert create[create.index("--head") + 1] == "alice:lesson1"
    assert create[create.index("--title") + 1] == "lesson 1: alice"
    assert world.files_in_pr() == sorted(s.relpath() for s in first)
    assert (
        "pull request: https://github.com/class/runs/pull/1" in capsys.readouterr().out
    )

    tip = world.fork_tip()
    submit_runs.submit(world.clone, 1, "alice", first)
    assert world.fork_tip() == tip
    assert len(world.creates()) == 1

    second = [*first, submission("alice")]
    submit_runs.submit(world.clone, 1, "alice", second)
    changed = run(
        "git", "--git-dir", str(world.fork), "diff", "--name-only", tip, "lesson1"
    ).split()
    assert changed == [second[-1].relpath()]
    assert len(world.creates()) == 1

    world.squash_merge()
    submit_runs.submit(world.clone, 1, "alice", second)
    assert "nothing new" in capsys.readouterr().out
    assert len(world.creates()) == 1

    third = [*second, submission("alice")]
    submit_runs.submit(world.clone, 1, "alice", third)
    assert world.files_in_pr() == [third[-1].relpath()]
    assert len(world.creates()) == 2


def test_already_on_upstream_is_not_written_again(world, submission):
    mine = submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [mine])
    # the same run reaches main another way, submitted at another moment
    elsewhere = submission("alice")
    earlier = datetime(2026, 10, 1, tzinfo=UTC)
    world.teacher_push(elsewhere.model_copy(update={"submitted_at": earlier}))
    tip = world.fork_tip()

    submit_runs.submit(world.clone, 1, "alice", [mine, elsewhere])
    assert world.fork_tip() == tip
    assert world.files_in_pr() == [mine.relpath()]


def test_two_lessons_get_two_pull_requests(world, submission):
    submit_runs.submit(world.clone, 1, "alice", [submission("alice", lesson=1)])
    submit_runs.submit(world.clone, 2, "alice", [submission("alice", lesson=2)])
    heads = [c[c.index("--head") + 1] for c in world.creates()]
    assert heads == ["alice:lesson1", "alice:lesson2"]
    assert len(world.files_in_pr("lesson2")) == 1


def test_a_pull_request_from_someone_elses_branch_is_not_ours(world, submission):
    world.state.write_text(
        json.dumps(
            [
                {
                    "url": "https://github.com/class/runs/pull/7",
                    "repo": "class/runs",
                    "base": "main",
                    "head_owner": "class",
                    "head_branch": "lesson1",
                    "author": "alice",
                    "state": "open",
                }
            ]
        )
    )
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    assert len(world.creates()) == 1


def test_a_second_clone_and_a_lagging_branch(world, submission, capsys):
    x, y, z = submission("alice"), submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [x])

    again = world.clone_fork(world.tmp / "re-clone")  # the student clones the fork anew
    submit_runs.submit(again, 1, "alice", [x, y])
    assert len(world.creates()) == 1
    assert (
        "pull request: https://github.com/class/runs/pull/1" in capsys.readouterr().out
    )

    # the first clone's lesson1 lacks y; the fork's commits are merged in, not lost
    submit_runs.submit(world.clone, 1, "alice", [x, y, z])
    assert world.files_in_pr() == sorted(s.relpath() for s in (x, y, z))
    assert len(world.creates()) == 1


def test_fork_branch_deleted_after_the_merge(world, submission):
    x, y = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [x])
    world.squash_merge()
    run("git", "--git-dir", str(world.fork), "branch", "-D", "lesson1")

    submit_runs.submit(world.clone, 1, "alice", [x, y])
    assert world.files_in_pr() == [y.relpath()]
    assert len(world.creates()) == 2


def test_a_run_deleted_locally_leaves_the_pull_request(world, submission, capsys):
    x, y = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [x, y])
    submit_runs.submit(world.clone, 1, "alice", [x], prune=True)
    assert world.files_in_pr() == [x.relpath()]
    out = capsys.readouterr().out
    assert "removed 1 run(s)" in out and y.relpath() in out

    world.squash_merge()
    z = submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [z], prune=True)  # x is on main
    assert world.files_in_pr() == [z.relpath()]
    assert world.on_upstream() == [x.relpath()]


def test_without_prune_nothing_is_removed(world, submission):
    x, y = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [x, y])
    submit_runs.submit(world.clone, 1, "alice", [x])
    assert world.files_in_pr() == sorted([x.relpath(), y.relpath()])


def test_commit_without_a_git_identity(world, submission, tmp_path, monkeypatch):
    bare = tmp_path / "gitconfig-no-user"
    bare.write_text(gitconfig(tmp_path, user=False))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(bare))
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    author = run("git", "--git-dir", str(world.fork), "log", "-1", "--format=%an <%ae>",
                 "lesson1")  # fmt: skip
    assert author == "alice <alice@users.noreply.github.com>"


def test_untracked_files_outside_lessons_do_not_block(world, submission):
    (world.clone / "notes.txt").write_text("mine\n")
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    assert len(world.creates()) == 1

    stray = world.clone / "lesson1/alice" / ("e" * 32 + ".json")
    stray.write_text("{}")
    with pytest.raises(SystemExit) as e:
        submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    assert "uncommitted changes" in str(e.value)


def test_no_pr_pushes_and_stops(world, submission, capsys):
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")], open_pr=False)
    assert world.fork_tip()
    assert world.gh_calls() == []
    assert (
        "https://github.com/class/runs/compare/main...alice:lesson1"
        in capsys.readouterr().out
    )


def test_without_gh_it_pushes_and_gives_the_compare_url(
    world, submission, capsys, monkeypatch
):
    monkeypatch.setattr(submit_runs, "gh_available", lambda: False)
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    assert world.fork_tip()
    assert world.gh_calls() == []
    out = capsys.readouterr().out
    assert "`gh` is not installed" in out
    assert "https://github.com/class/runs/compare/main...alice:lesson1" in out


def test_clone_without_upstream(world, submission):
    run("git", "remote", "remove", "upstream", cwd=world.clone)
    with pytest.raises(SystemExit) as e:
        submit_runs.submit(world.clone, 1, "alice", [submission("alice")])
    assert "remote add upstream" in str(e.value)
    assert "fetch upstream" in str(e.value)


def test_logins_that_disagree_stop(world):
    with pytest.raises(SystemExit) as e:
        submit_runs.resolve_login("bob", world.clone)
    assert "--student says bob" in str(e.value)
    assert "gh api user says alice" in str(e.value)


def test_login_takes_gh_spelling(world, monkeypatch):
    monkeypatch.setenv("FAKE_GH_LOGIN", "Alice")
    assert submit_runs.resolve_login("alice", world.clone) == "Alice"
    assert submit_runs.resolve_login(None, world.clone) == "Alice"


def test_login_from_origin_without_gh(world, monkeypatch):
    monkeypatch.setattr(submit_runs, "gh", lambda *args: None)
    assert submit_runs.resolve_login(None, world.clone) == "alice"


def test_github_slug():
    assert submit_runs.github_slug("git@github.com:alice/mads-runs.git") == (
        "alice/mads-runs"
    )
    assert submit_runs.github_slug("https://tok@github.com/alice/mads-runs") == (
        "alice/mads-runs"
    )
    assert submit_runs.github_slug("/tmp/fork.git") is None


def test_list_folds_seed_series():
    line = submit_runs.summarize_metrics(
        {
            "accuracy": 0.8123,
            "accuracy_std": 0.0123,
            "accuracy_per_seed": 0.83,
            "n_seeds": 5.0,
            "n_params": 1000.0,
        }
    )
    assert line == "accuracy=0.8123 ± 0.012 (n=5), n_params=1000"


def local_store(tmp_path, runs) -> tuple[str, dict[str, str]]:
    """A student's mlflow store with `runs`, (experiment, name, tags) each, finished,
    with an accuracy and a design script; the URI and run id by name."""
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    client = MlflowClient(tracking_uri=uri)
    ids = {}
    for experiment, name, tags in runs:
        exp = client.get_experiment_by_name(experiment)
        exp_id = (
            exp.experiment_id
            if exp
            else client.create_experiment(
                experiment, artifact_location=(tmp_path / "artifacts").as_uri()
            )
        )
        rid = client.create_run(exp_id, run_name=name, tags=tags).info.run_id
        ids[name] = rid
        client.log_metric(rid, "accuracy", 0.5)
        client.log_text(rid, "model = 1\n", "design.py")
        client.set_terminated(rid)
    return uri, ids


def submit_main(monkeypatch, world, uri: str, *extra: str) -> list[str]:
    """`submit_runs.py --lesson 1` without pushing; the files it wrote on lesson1."""
    argv = ["submit_runs.py", "--lesson", "1", "--student", "alice", "--no-push"]
    argv += ["--runs-repo", str(world.clone), "--tracking-uri", uri, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    submit_runs.main()
    return run("git", "ls-files", "lesson1", cwd=world.clone).split()


def test_runs_imported_from_the_class_are_not_submitted(
    world, tmp_path, monkeypatch, capsys
):
    uri, ids = local_store(
        tmp_path,
        [
            ("lesson1-notebook", "mine", {"role": "final"}),
            ("lesson1-notebook", "bob:baseline",
             {"submission_ref": f"lesson1/bob/{'a' * 32}"}),
            ("lesson1-notebook", "carol:baseline", {"source_run_id": "b" * 32}),
        ],
    )  # fmt: skip
    written = submit_main(monkeypatch, world, uri)

    out = capsys.readouterr().out
    assert "not submitted: 2 run(s) imported from the class repository" in out
    assert written == [f"lesson1/alice/{ids['mine']}.json"]
    data = json.loads((world.clone / written[0]).read_text())
    assert data["run"]["artifacts"] == {"design.py": "model = 1\n"}


def test_a_run_that_does_not_fit_is_skipped_and_the_rest_submitted(
    world, tmp_path, monkeypatch, capsys
):
    uri, ids = local_store(
        tmp_path,
        [
            ("lesson1-notebook", "good", {"role": "final"}),
            ("lesson1-notebook", "bad", {"builds_on": "lesson1/bob"}),
        ],
    )
    written = submit_main(monkeypatch, world, uri)
    assert written == [f"lesson1/alice/{ids['good']}.json"]
    [line] = [x for x in capsys.readouterr().out.splitlines() if "skipped" in x]
    assert line.startswith(f"skipped run {ids['bad']} (bad), it does not fit")
    assert "tag 'builds_on': 'lesson1/bob' is not a submission_ref" in line


def test_an_artifact_over_the_limit_is_left_out_with_a_line(tmp_path, capsys):
    uri, ids = local_store(tmp_path, [("lesson1-notebook", "big", {})])
    client = MlflowClient(tracking_uri=uri)
    client.log_text(ids["big"], "x" * 256_001, "notes.txt")
    assert submit_runs.read_artifacts(client, ids["big"]) == {
        "design.py": "model = 1\n"
    }
    out = capsys.readouterr().out
    assert (
        f"left out of {ids['big'][:8]}: the artifact notes.txt is 256001 bytes" in out
    )


def test_by_default_only_the_notebook_and_the_study(world, tmp_path, monkeypatch):
    uri, ids = local_store(
        tmp_path,
        [
            ("lesson1-notebook", "handed-out", {}),
            ("lesson1-study", "mine", {}),
            ("lesson1-search", "notebook-04", {}),
        ],
    )
    path = {name: f"lesson1/alice/{rid}.json" for name, rid in ids.items()}
    written = submit_main(monkeypatch, world, uri)
    assert written == sorted([path["handed-out"], path["mine"]])

    written = submit_main(monkeypatch, world, uri, "--experiment", "lesson1-search")
    assert path["notebook-04"] in written
