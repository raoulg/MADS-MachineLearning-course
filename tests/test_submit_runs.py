"""The git part of submit_runs, against local bare repositories and a fake `gh`.

`https://github.com/<owner>/<repo>.git` is rewritten by git's `insteadOf` to a bare
repository under `gh-root/`: `class/runs.git` plays the class repository, `alice/runs`
the student's fork. A fake `gh` on PATH records every call and keeps the pull requests
in a JSON file, filtering them the way `gh pr list` does.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import submit_runs  # ty: ignore[import-not-found]

FAKE_GH = f"""#!{sys.executable}
import json, os, re, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
path = os.environ["FAKE_GH_STATE"]
prs = json.load(open(path)) if os.path.exists(path) else []

def opt(name):
    return args[args.index(name) + 1] if name in args else None

if args[:2] == ["api", "user"]:
    print(os.environ["FAKE_GH_LOGIN"])
    sys.exit(0)
repo = opt("--repo")
if not repo or not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+", repo):
    sys.exit(f"fake gh: --repo must be OWNER/REPO, got {{repo!r}}")
if args[:2] == ["pr", "list"]:
    hits = [
        p for p in prs
        if p["repo"] == repo
        and (opt("--head") is None or p["head_branch"] == opt("--head"))
        and (opt("--author") is None or p["author"] == opt("--author"))
        and (opt("--state") is None or p["state"] == opt("--state"))
    ]
    print(json.dumps([
        {{"url": p["url"], "headRepositoryOwner": {{"login": p["head_owner"]}}}}
        for p in hits
    ]))
elif args[:2] == ["pr", "create"]:
    owner, branch = opt("--head").split(":")
    url = f"https://github.com/{{repo}}/pull/{{len(prs) + 1}}"
    prs.append({{
        "url": url, "repo": repo, "base": opt("--base"), "head_owner": owner,
        "head_branch": branch, "author": os.environ["FAKE_GH_LOGIN"], "state": "open",
    }})
    json.dump(prs, open(path, "w"))
    print(url)
else:
    sys.exit(f"fake gh does not know {{args}}")
"""


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class World:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root = tmp / "gh-root"
        self.upstream = self.root / "class/runs.git"
        self.fork = self.root / "alice/runs.git"
        self.clone = tmp / "mads-runs"
        self.log = tmp / "gh-log.jsonl"
        self.state = tmp / "gh-prs.json"

        seed = tmp / "seed"
        run("git", "init", "--quiet", "-b", "main", str(seed))
        (seed / "README.md").write_text("class runs\n")
        run("git", "add", "README.md", cwd=seed)
        run("git", "commit", "--quiet", "-m", "start", cwd=seed)
        run("git", "clone", "--quiet", "--bare", str(seed), str(self.upstream))
        run("git", "clone", "--quiet", "--bare", str(self.upstream), str(self.fork))
        shutil.rmtree(seed)
        self.clone_fork(self.clone)

    def clone_fork(self, where: Path) -> Path:
        run("git", "clone", "--quiet", "https://github.com/alice/runs.git", str(where))
        run(
            "git", "remote", "add", "upstream", "https://github.com/class/runs.git",
            cwd=where,
        )  # fmt: skip
        return where

    def gh_calls(self, *prefix: str) -> list[list[str]]:
        if not self.log.exists():
            return []
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [c for c in calls if c[: len(prefix)] == list(prefix)]

    def creates(self) -> list[list[str]]:
        return self.gh_calls("pr", "create")

    def fork_tip(self, branch: str = "lesson1") -> str:
        return run("git", "--git-dir", str(self.fork), "rev-parse", branch)

    def files_in_pr(self, branch: str = "lesson1") -> list[str]:
        """What GitHub shows in the pull request: the diff from the merge base."""
        check = self.tmp / "check"
        if not check.exists():
            run("git", "clone", "--quiet", str(self.upstream), str(check))
            run("git", "remote", "add", "fork", str(self.fork), cwd=check)
        run("git", "fetch", "--quiet", "--prune", "origin", cwd=check)
        run("git", "fetch", "--quiet", "--prune", "fork", cwd=check)
        out = run(
            "git", "diff", "--name-only", f"origin/main...fork/{branch}", cwd=check
        )
        return sorted(out.split())

    def on_upstream(self) -> list[str]:
        out = run("git", "--git-dir", str(self.upstream), "ls-tree", "-r",
                  "--name-only", "main")  # fmt: skip
        return sorted(p for p in out.split() if p.startswith("lesson"))

    def teacher_push(self, sub) -> None:
        """The teacher writes a file straight into the class repository's main."""
        work = self.tmp / "teacher"
        shutil.rmtree(work, ignore_errors=True)
        run("git", "clone", "--quiet", str(self.upstream), str(work))
        path = work / sub.relpath()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sub.dump())
        run("git", "add", "-A", cwd=work)
        run("git", "commit", "--quiet", "-m", "teacher", cwd=work)
        run("git", "push", "--quiet", "origin", "main", cwd=work)

    def squash_merge(self, branch: str = "lesson1") -> None:
        """What the workflow's `gh pr merge --squash` does to the class repository."""
        work = self.tmp / "merge"
        shutil.rmtree(work, ignore_errors=True)
        run("git", "clone", "--quiet", str(self.upstream), str(work))
        run("git", "fetch", "--quiet", str(self.fork), branch, cwd=work)
        run("git", "merge", "--quiet", "--squash", "FETCH_HEAD", cwd=work)
        run("git", "commit", "--quiet", "-m", "squash", cwd=work)
        run("git", "push", "--quiet", "origin", "main", cwd=work)
        prs = json.loads(self.state.read_text())
        for pr in prs:
            if pr["head_branch"] == branch:
                pr["state"] = "merged"
        self.state.write_text(json.dumps(prs))


def gitconfig(tmp: Path, user: bool = True) -> str:
    text = f'[url "file://{tmp / "gh-root"}/"]\n  insteadOf = https://github.com/\n'
    text += "[init]\n  defaultBranch = main\n"
    if user:
        text += "[user]\n  name = Student\n  email = student@example.invalid\n"
    return text


@pytest.fixture
def world(tmp_path, monkeypatch):
    config = tmp_path / "gitconfig"
    config.write_text(gitconfig(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gh"
    fake.write_text(FAKE_GH)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    w = World(tmp_path)
    monkeypatch.setenv("FAKE_GH_LOG", str(w.log))
    monkeypatch.setenv("FAKE_GH_STATE", str(w.state))
    monkeypatch.setenv("FAKE_GH_LOGIN", "alice")
    return w


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
