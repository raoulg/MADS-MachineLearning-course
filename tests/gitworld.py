"""A git world for the scripts that talk to the class runs repository: local bare
repositories and a fake `gh`.

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
from pathlib import Path

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


def make_world(tmp_path: Path, monkeypatch) -> World:
    """The world in `tmp_path`, with git and `gh` pointed at it through `monkeypatch`."""
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
