"""The git part of submit_runs, against two local bare repositories and a fake `gh`.

`upstream.git` plays the class repository, `fork.git` the student's fork, and a fake
`gh` on PATH records every call and keeps a list of pull requests in a JSON file.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import submit_runs  # ty: ignore[import-not-found]

FAKE_GH = f"""#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
state_path = os.environ["FAKE_GH_STATE"]
prs = json.load(open(state_path)) if os.path.exists(state_path) else []
if args[:2] == ["api", "user"]:
    print(os.environ["FAKE_GH_LOGIN"])
elif args[:2] == ["pr", "list"]:
    print(json.dumps([p for p in prs if p["state"] == "open"]))
elif args[:2] == ["pr", "create"]:
    owner = args[args.index("--head") + 1].split(":")[0]
    url = f"https://github.com/class/runs/pull/{{len(prs) + 1}}"
    prs.append({{"url": url, "headRepositoryOwner": {{"login": owner}}, "state": "open"}})
    json.dump(prs, open(state_path, "w"))
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
        self.upstream = tmp / "upstream.git"
        self.fork = tmp / "fork.git"
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
        run("git", "clone", "--quiet", str(self.fork), str(self.clone))
        run("git", "remote", "add", "upstream", str(self.upstream), cwd=self.clone)

    def gh_calls(self, *prefix: str) -> list[list[str]]:
        if not self.log.exists():
            return []
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [c for c in calls if c[: len(prefix)] == list(prefix)]

    def fork_tip(self, branch: str = "lesson1") -> str:
        return run("git", "--git-dir", str(self.fork), "rev-parse", branch)

    def files_in_pr(self, branch: str = "lesson1") -> list[str]:
        """What GitHub shows in the pull request: the diff from the merge base."""
        check = self.tmp / "check"
        if not check.exists():
            run("git", "clone", "--quiet", str(self.upstream), str(check))
            run("git", "remote", "add", "fork", str(self.fork), cwd=check)
        run("git", "fetch", "--quiet", "origin", cwd=check)
        run("git", "fetch", "--quiet", "fork", cwd=check)
        out = run(
            "git", "diff", "--name-only", f"origin/main...fork/{branch}", cwd=check
        )
        return out.split()

    def squash_merge(self, branch: str = "lesson1") -> None:
        """What the workflow's `gh pr merge --squash` does to the class repository."""
        work = self.tmp / "merge"
        run("git", "clone", "--quiet", str(self.upstream), str(work))
        run("git", "fetch", "--quiet", str(self.fork), branch, cwd=work)
        run("git", "merge", "--quiet", "--squash", "FETCH_HEAD", cwd=work)
        run("git", "commit", "--quiet", "-m", "squash", cwd=work)
        run("git", "push", "--quiet", "origin", "main", cwd=work)
        prs = json.loads(self.state.read_text())
        for pr in prs:
            pr["state"] = "merged"
        self.state.write_text(json.dumps(prs))
        subprocess.run(["rm", "-rf", str(work)], check=True)


@pytest.fixture
def world(tmp_path, monkeypatch):
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        "[user]\n  name = Student\n  email = student@example.invalid\n"
        "[init]\n  defaultBranch = main\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
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
    assert len(world.gh_calls("pr", "create")) == 1
    create = world.gh_calls("pr", "create")[0]
    assert create[create.index("--title") + 1] == "lesson 1: alice"
    assert create[create.index("--head") + 1] == "alice:lesson1"
    assert sorted(world.files_in_pr()) == sorted(s.relpath() for s in first)
    assert (
        "pull request: https://github.com/class/runs/pull/1" in capsys.readouterr().out
    )

    tip = world.fork_tip()
    submit_runs.submit(world.clone, 1, "alice", first)
    assert world.fork_tip() == tip
    assert len(world.gh_calls("pr", "create")) == 1

    second = [*first, submission("alice")]
    submit_runs.submit(world.clone, 1, "alice", second)
    changed = run(
        "git", "--git-dir", str(world.fork), "diff", "--name-only", tip, "lesson1"
    ).split()
    assert changed == [second[-1].relpath()]
    assert len(world.gh_calls("pr", "create")) == 1

    world.squash_merge()
    submit_runs.submit(world.clone, 1, "alice", second)
    assert "nothing new" in capsys.readouterr().out
    assert len(world.gh_calls("pr", "create")) == 1

    third = [*second, submission("alice")]
    submit_runs.submit(world.clone, 1, "alice", third)
    assert world.files_in_pr() == [third[-1].relpath()]
    assert len(world.gh_calls("pr", "create")) == 2


def test_no_pr_pushes_and_stops(world, submission, capsys):
    submit_runs.submit(world.clone, 1, "alice", [submission("alice")], open_pr=False)
    assert world.fork_tip()
    assert world.gh_calls() == []
    assert "Open it in the browser" in capsys.readouterr().out


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


def test_login_from_gh(world):
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
