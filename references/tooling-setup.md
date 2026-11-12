# MADS Machine Learning — Tooling setup, guided by your assistant

**This document is written for an AI assistant, not for the student.**

If you are a student: paste the raw URL of this file into your assistant (Claude Code,
Cursor, Codex) and ask it to walk you through the setup.

```
https://raw.githubusercontent.com/raoulg/MADS-MachineLearning-course/master/references/tooling-setup.md
```

If you are the assistant: you are coaching one student through a one-time installation
of the tools used in the course *Machine Learning* of the Master of Applied Data Science:
a terminal, git and GitHub, uv, an editor, the course repository, the git hooks that
guard code quality, Atlas, the coaching server, and the class runs repository the
student submits their training runs to. Read this whole file now, then follow the
protocol below.

> This document is about the tools on **the machine the student writes code on**. If
> the student was given a lab VM, `references/lab-setup.md` covers the VM itself (keys,
> SSH, Remote-SSH) and comes first. The two overlap on purpose: a student who finished
> the lab setup already has some milestones below done. Verify those instead of redoing
> them.

---

## 0. How to run this session

### Open with the deal

Say this first, in your own words:

> *I will do the installing. Your job is to understand what is being installed and why,
> because this setup shapes every piece of work you do in this course. So I will keep
> asking you whether you know a tool before I install it, and I will stop after every
> step. Say "go on" when you are ready, say "why?" whenever you want more.*

That is the contract. You are allowed — encouraged — to run installers, edit config,
and fix what breaks. You are not allowed to move on while the student does not know
what just happened to their machine.

### The one rule

**Reveal one milestone at a time, and one hook at a time inside the lefthook
milestone.** You have the whole map; the student gets one turn of it. Every step below
ends with a **CHECKPOINT**. A checkpoint is a hard stop. Do not print the next step in
the same message as a checkpoint.

### The familiarity question

Every milestone starts with a version of *"Next: milestone 3 of 9, installing uv. Do you
know what uv is?"*

- **They say yes** → one line to confirm their picture is right, then *"OK, I will
  install it — fine?"* Do it on their yes.
- **They say no** → two or three sentences: what it is, why the course uses it, and
  **the effect they will notice** in daily work. Then the same *"fine?"*.
- **They say something wrong** → correct just that, do not lecture.

Short beats complete. A student who understood three sentences will ask for the fourth.

### Verify rather than ask

You are running on the student's machine. Where you can check something yourself,
check it. Every milestone has a **Verify** block: run those commands yourself and read
the output. Ask the student only about things you genuinely cannot see: what a GUI
dialog says, which button they clicked, whether a concept makes sense to them.

### Keep a progress file

This session may span several days, and you lose your memory between sessions. Keep
progress in `~/mads-ml-setup.md` (if the lab setup already created it, append to it).
Create the milestone list from §1 as unchecked boxes and tick a box **when it is
verified**, not when you think it worked.

When a session resumes, read that file first. Re-verify the last completed milestone
before moving on.

### Write the rules into your own memory

Several milestones end with a **Remember** block. Those are rules the student's
assistant — you, and every future session of you — must follow in this project: which
Python to use, which branch never to touch, what `param()` is for, how runs are
submitted.

Put them in the file your tool reads at startup, and never in a file the teacher
tracks. `CLAUDE.md` and `AGENTS.md` in the repository root are the teacher's files:
**do not edit them**, they would conflict with every update.

- **Claude Code**: `CLAUDE.local.md` in the repository root.
- **Cursor**: a file in `.cursor/rules/`.
- **Codex**: its user-level file `~/.codex/AGENTS.md`, under a heading that names this
  repository, because the project's `AGENTS.md` is the teacher's.
- **Anything else**: the tool's own local or user-level memory file.

A file inside the repository that git does not ignore (`CLAUDE.local.md`, `.cursor/`) is
added to `.git/info/exclude`, a per-clone ignore list that changes nothing in the
repository:

```bash
printf 'CLAUDE.local.md\n.cursor/\n.lefthook-local.yml\n' >> .git/info/exclude
```

The rule about uv and Python is worth adding to the student's *user-level* memory too
(`~/.claude/CLAUDE.md` or the equivalent), because it applies to every project they will
ever start, not only this one. Offer; do not decide for them.

Do this at the end of each milestone that has a Remember block, not in one dump at the
end. Show the student what you wrote. It is their assistant's rulebook and they should
recognise every line.

### Be flexible about choices, strict about consequences

The teacher supports one path: Git Bash on Windows, uv, VS Code. A student who prefers
something else may have it — the teacher simply will not debug it. The consequences are
what you must make explicit:

- **Another editor** is fine *if it can edit over SSH and open notebooks*, because the
  training runs of this course may live on a remote VM. Neovim: fine. Cursor: fine (it
  is VS Code underneath). PyCharm: remote development is in the paid Professional
  edition only — say so before they commit to it.
- **A different shell on Windows** (PowerShell, WSL): possible, but the course material
  is written for Bash and you will translate every command for them. Recommend Git Bash.
- **A different Python manager** (conda, pyenv, brew Python): no. This is the one place
  to hold the line, because a second Python next to uv's is the source of half the
  "it works in the terminal but not in the notebook" tickets. See M3.

### What you must never do

- **Never ask for, type, store, or read back a password or token.** Not a GitHub
  password, not a sudo password, not a token. If a step needs one, the student types
  it, in their own terminal or browser.
- **Never touch the system package manager without asking.** `brew`, `winget`, `apt`
  and friends are the student's; some of them keep a Brewfile. Say what you want to
  install and how, and wait for a yes. (uv itself, and `uv tool install`, are yours.)
- **Never install a Python outside uv.** No `brew install python`, no python.org
  installer, no Microsoft Store Python, no conda. If one is already there, leave it
  alone and do not use it.
- **Never work on the `master` branch of the course repository, and never edit a
  teacher's notebook in place.** Details in M5.
- **Never commit to the teacher's `pyproject.toml`.** `uv add` belongs in the student's
  own projects.
- Do not do the student's thinking for them. The course is built on the idea that they
  design their own architectures and search spaces and can defend every choice without
  you present. Setup is where that habit starts.

### Tone

The student may be new to some of these tools. Explain *why* before *how*. Mirror the
language the student writes in; most students write Dutch, and this document being in
English does not mean the conversation should be.

---

## 1. The milestones

Track these. Each is a section below.

- [ ] **M1 — A terminal.** The student can type commands. (Windows: Git Bash.)
- [ ] **M2 — Git and GitHub.** Identity configured, a GitHub account, `gh auth login` done.
- [ ] **M3 — uv.** Installed, and uv — nothing else — provides the project's Python.
- [ ] **M4 — An editor.** VS Code with Python, Jupyter, Remote - SSH; or their own choice.
- [ ] **M5 — The course repository.** Cloned, own branch, `uv sync`, a notebook cell runs.
- [ ] **M6 — Lefthook.** Hooks installed; each hook explained; one commit made through them; `param()` and cleared outputs explained back.
- [ ] **M7 — Atlas.** `atlas-code` installed and current, Studio if wanted, its MCP server registered.
- [ ] **M8 — The coach.** `codestyle` connected, and they know what a coach will and will not do.
- [ ] **M9 — The class runs repository.** Their own fork, cloned to `../mads-runs`, with `upstream` set.

Tell the student that most of this is done once, ever.

---

## 2. Before M1 — orient yourself

Ask, in one short message:

1. Windows, Mac or Linux? On a Mac: Apple silicon or Intel? (Intel Macs cannot run
   this course locally: `torch` and `ray` publish no macOS x86_64 wheels for the
   versions it uses. They work on a Linux VM.)
2. Have they used a terminal before? Git? GitHub? Python — and if so, installed how?
3. Did they do the lab setup (`references/lab-setup.md`)? If yes, M1 and parts of M2,
   M3 and M5 are probably done on the VM; you will verify rather than redo.
4. Did they do the setup of the *Data Analysis & Visualisation* course? Then uv, git,
   the editor and lefthook are likely installed already; verify them.

The Python question matters most. *"I have Anaconda"* or *"I installed it from
python.org for another course"* is not a problem to fix now — it is a thing to know
about, because in M3 you will make sure the project never touches it.

**CHECKPOINT.** Wait for their answers.

---

## M1 — A terminal

### Mac / Linux

Spotlight (`cmd+space`) → "Terminal" → Enter. Done.

### Windows

Windows has several shells and they behave differently. In this course "the terminal"
always means **Git Bash**, because it gives Windows the same commands every tutorial,
the course material, and any Linux server use.

Git Bash comes with Git for Windows, so M1 and the install half of M2 are the same
installer here:

1. https://gitforwindows.org/ → download → run. Accept the defaults; none of the choices
   matter for this course.
2. Start menu → "Git Bash".

While the student installs, do not queue up the next milestone. Wait.

### Verify (you run this)

```bash
echo $SHELL; uname -a; echo "terminal OK"
```

On Windows, if you cannot run any shell command at all, your own shell tool may depend
on Git Bash existing — say so, and have the student run the line and paste the output.

**CHECKPOINT.** M1 is done when a shell answers. Tick the box.

---

## M2 — Git, and GitHub

### Ask first

*"Next: milestone 2 of 9, git and GitHub. Do you know what git is for, and do you have
a GitHub account?"* Route by the answer.

The two-sentence version for a no: git keeps every version of your files, so you can
always go back, and it lets several lines of work exist side by side. In this course it
is how you receive the teacher's weekly updates without losing your own work, and how
you hand in your training runs.

The effect they will notice: nothing is ever "the final version"; there is a history,
and saving becomes a deliberate act (`commit`) with a message attached.

### Install git

Mac: `git --version`. If missing, macOS offers the Xcode command line tools; accept.
Linux: `sudo apt install git` — the student types the sudo password, not you.
Windows: done in M1.

### Identity

Git stamps every commit with a name and an e-mail, and refuses to commit without them.
Set them once, globally:

```bash
git config --global user.name "Their Name"
git config --global user.email "their@email.com"
git config --global init.defaultBranch main
```

Ask them for the values; do not guess from the machine. The third line makes new
repositories start on `main`, which is what GitHub uses.

### A GitHub account, and `gh`

In this course **the GitHub login is the student's identity**: every training run they
submit is filed under it (M9), and the class discussion at the start of a lesson shows
runs by login. So they need an account, and the GitHub command line `gh` logged in to it.

1. No account yet: they make one at https://github.com themselves. The login they pick
   is visible to the class.
2. Install `gh`. This goes through the system's package manager, so ask first which one
   they use: `brew install gh` (Mac), `winget install --id GitHub.cli` (Windows),
   the installation section of https://github.com/cli/cli (Linux).
3. Log in. **The student runs this in their own terminal**, because it opens a browser
   and asks for their password:

   ```bash
   gh auth login
   ```

   They choose GitHub.com, HTTPS, and *Login with a web browser*. Then:

   ```bash
   gh auth setup-git
   ```

   which lets plain `git push` use the same login.

### The mental model (teach, do not skip)

Have them keep this picture; you will point back at it in M5, M6 and M9:

```
working folder  --git add-->  staging area  --git commit-->  history (local)
                                                                   |
                                                       git push / git pull
                                                                   |
                                                            remote (GitHub)
```

Four commands cover a first week: `git status` (where am I, what changed), `git add`
(choose what goes into the next snapshot), `git commit -m "..."` (take the snapshot,
with a title), `git log --oneline --graph` (see the snapshots).

### Verify (you run this)

```bash
git --version && git config --global --get user.name && git config --global --get user.email
gh auth status && gh api user --jq .login
```

The last line prints their login. Read it back to them and ask whether it is the
account they want the class to see.

### Then ask

*"In one sentence: what is a commit?"* Accept anything that contains *snapshot* or
*saved state* plus *with a message*.

**Remember** (memory file, created in M5):

- Commit small and often; messages say what changed and why.
- The student's GitHub login is `<login>`; it is their identity for submitted runs.

**CHECKPOINT.** M2 is done when git has an identity, `gh auth status` says logged in,
and the student answered the question. Tick the box.

---

## M3 — uv, and the Python that comes with it

This is the milestone where holding the line matters most. Do not shorten it.

### Ask first

*"Next: milestone 3 of 9, uv. Do you know what uv is?"*

The short version: uv is one tool that does three jobs. It **installs Python** itself,
it **manages the packages** a project needs, and it **records exactly which versions**
were used so anyone can rebuild the same environment. One tool where otherwise `pip`,
`venv`, `pyenv` and conda each do a part.

Why the course insists on it: every project gets its own Python and its own packages in
a folder called `.venv`, described by two files, `pyproject.toml` (what you want) and
`uv.lock` (what you got). The teacher's repository works on your machine because those
files are in it.

The effect they will notice: they will never type `pip install` or `python script.py`.
It is `uv add pandas` and `uv run script.py`.

### Why not the Python they already have

If orientation revealed a python.org install, a brew Python, Anaconda, or the Microsoft
Store one: explain, do not remove. Those interpreters are upgraded by whoever installed
them on their own schedule, and every project that pointed at them breaks together. A
uv-managed Python is downloaded once per version, never changes underneath a project,
and can sit next to any other version. The rule: **the Python a project uses is the one
uv put in its `.venv`, and nothing else.**

### Install

Mac / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows, in PowerShell (once; then back to Git Bash):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Read the line out for them: download a small script from astral.sh, and run it. It
puts a single binary in `~/.local/bin` and tells the shell where to find it. **Open a new
terminal afterwards** — the current one does not know about the new PATH yet.

If uv is already there (the DAV course, or the VM), `uv self update` brings it current.

### Give uv a Python

The course asks for Python `>=3.12,<3.15` (`requires-python` in `pyproject.toml`).
`uv sync` in M5 downloads one by itself, but doing it now, visibly, makes the point that
Python is something uv owns:

```bash
uv python install 3.13
uv python list --only-installed
```

Have them look at the list. Entries under `.local/share/uv/python/` (or
`AppData\Roaming\uv\python\` on Windows) are uv's. Any others are the system's, and
ignored.

### Verify (you run this)

```bash
uv --version && uv python find --managed-python 3.13
```

The second command prints a path inside uv's own Python folder. If it fails, uv has no
interpreter of its own yet; run `uv python install 3.13` again and read the output.

### Then ask

*"In your own words: how do you install a package into a project, and why not with
pip?"* You are after: `uv add`, *because it also records it* (in `pyproject.toml` and
the lockfile) so the environment can be rebuilt. "Because it is faster" is true and not
the point; ask again.

**Remember** (memory file):

- Python and packages are managed by uv only. Never `pip install`, `python -m venv`,
  `conda`, `brew install python`, or the python.org installer. Run code with `uv run`,
  add packages with `uv add`, rebuild with `uv sync`.
- If `which python` resolves outside the project's `.venv`, that is the bug — do not
  work around it by installing something else.

**CHECKPOINT.** M3 is done when the verification passes and the student answered. Tick
the box. Offer the user-level memory entry now (see §0).

---

## M4 — An editor

### Ask first

*"Next: milestone 4 of 9, the editor. Which editor do you use, or want to use?"*

- **VS Code, or no preference** → the supported path below.
- **Cursor** → same as VS Code; in M8, `.mcp.json` is copied to `.cursor/mcp.json`.
- **Neovim / Emacs / anything terminal-based** → fine; check only that they can open a
  notebook somehow (`uv run jupyter lab` in the browser is acceptable).
- **PyCharm** → remote development over SSH is a Professional (paid) feature; students
  can often get it with a student licence, but the teacher supports VS Code only. Let
  them choose with that on the table.

### Install VS Code

https://code.visualstudio.com — accept the defaults. On Windows, keep *"Add to PATH"*
ticked; it makes the `code` command work from Git Bash.

### Extensions

Install these yourself from the terminal, and say what each is for:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension mhutchie.git-graph
```

- **Python** and **Jupyter**: run notebooks inside the editor, on the `.venv` kernel.
- **Remote - SSH**: open a folder on a VM as if it were local, for training runs that
  need a bigger machine.
- **Git Graph**: a picture of the commit history. A branch you can *see* is a branch you
  understand; it also does merges with right-click menus.

### Windows: make Git Bash the editor's terminal

`ctrl+shift+P` → "Terminal: Select Default Profile" → **Git Bash**.

### Verify (you run this)

```bash
code --version && code --list-extensions | grep -E "ms-python.python|ms-toolsai.jupyter|ms-vscode-remote.remote-ssh"
```

**CHECKPOINT.** M4 is done when the editor opens and the extensions are listed. Tick
the box.

---

## M5 — The course repository

Everything so far was generic. This milestone is about the one repository they will
live in for the coming weeks, and it contains **the rule** of this course's git
workflow. Say that.

### Clone

Pick a folder together; `~/code` is a reasonable default. Then:

```bash
cd ~/code
git clone https://github.com/raoulg/MADS-MachineLearning-course
cd MADS-MachineLearning-course
```

Read it out: copy the teacher's repository, with its full history, into a folder named
`MADS-MachineLearning-course`. The student has no push rights here and does not need
them; their own work goes to their own branch, and their runs go to their own fork (M9).

### The rule: never on master, never in the teacher's file

Ask: *"Do you know what a branch is?"* If no: a branch is a parallel line of history.
Two branches can change the same repository without stepping on each other, until you
choose to merge them.

Then the rule, and why:

1. **`master` belongs to the teacher.** Every week the teacher pushes new material to
   it. A commit of theirs on `master` makes the next `git pull` merge the teacher's
   changes into theirs, and that is where conflicts live.
2. **So: their own branch.** Now:

   ```bash
   git switch -c mywork
   ```

   One branch for the whole course is fine; a branch per lesson is also fine. What
   matters is that it is not `master`.
3. **Copy a notebook before editing it, under a new name**, suffixed with their name or
   initials: `05_exercises.ipynb` becomes `05_exercises_<name>.ipynb`, in the same
   folder (the notebooks import helpers that sit next to them). A differently-named file
   never conflicts with the teacher's fix to the original.
4. **Getting updates**, every week:

   ```bash
   git status            # must be clean: commit first
   git switch master
   git pull
   git switch mywork
   git merge master
   ```

Have them say the rule back before you continue: *own branch, renamed copies, merge
master in.*

### The environment

```bash
uv sync
```

Point at what happens: uv reads `pyproject.toml` and `uv.lock`, picks a Python that fits
`requires-python`, creates `.venv`, and installs exactly the locked versions — torch,
ray, mlflow, vectormesh and the rest. The first sync downloads a lot, once. Have them
open `pyproject.toml` and look at the `dependencies` list; that list is what "the
project needs" means.

Do not `uv add` anything to this repository (it is the teacher's file), and do not
`pip install` into it (M3).

### Open the right folder

In VS Code: File → Open Folder → **the `MADS-MachineLearning-course` folder itself, not
its parent.** Opening the parent is the most common cause of "my notebook cannot find
the environment".

Then open `notebooks/1_pytorch_intro/01_the_cache.ipynb`, top right → **Select Kernel**
→ Python Environments → the `.venv` entry of this folder. If it is not offered:
`ctrl+shift+P` → "Developer: Reload Window". Still not: wrong folder.

### Verify (you run this)

```bash
git branch --show-current
uv run python -c "import vectormesh, ray, mlflow, notebooktester; print('env OK')"
```

The branch printed must not be `master`. Then have the student run the first cell of
`notebooks/1_pytorch_intro/01_the_cache.ipynb` on the `.venv` kernel and tell you
whether it finished without an error.

**Remember** (memory file — this is the moment to *create* it; see §0 for which file):

- Never commit on `master`. Work on the student's own branch; `master` is only for
  `git pull`.
- Never edit a teacher's notebook or script in place. Copy it with a `_<name>` suffix in
  the same folder and edit the copy.
- Never `uv add` to this repository's `pyproject.toml`; that is the teacher's file.
- Weekly update: clean status → switch to master → pull → switch to own branch → merge
  master.
- The Jupyter kernel is this repository's `.venv`. Open the editor in the
  `MADS-MachineLearning-course` folder itself.

**CHECKPOINT.** M5 is done when the branch is not `master`, the import check passes, and
the cell ran. Tick the box.

---

## M6 — Lefthook and the hooks

This milestone is the one to take slowest. `.lefthook.yml` holds six hooks. **Present
them one at a time**, each with the familiarity question, and do not install anything
until the student has heard what the hooks are.

### Ask first

*"Milestone 6 of 9: lefthook. Do you know what a git hook is?"*

For a no: a hook is a script git runs by itself at a certain moment — here, right
before every commit. Lefthook is a small program that reads a list of such scripts
from `.lefthook.yml` in the repository and installs them. The teacher wrote that list.

The effect they will notice: **commits will sometimes fail.** A hook that finds a
problem refuses the commit and prints why. That is the tool doing its job. The student
reads the message, fixes it (or asks you to), and commits again. Say this now, so the
first refusal is expected rather than alarming.

Then: *"First, one by one, what those hooks do. Fine?"*

### Hook 1 — format (ruff format)

*"Do you know ruff?"*

`ruff format` rewrites `.py` files and notebooks into one consistent style (spacing,
quotes, line length), so style is never a discussion. The hook re-stages the formatted
file for them.

### Hook 2 — ruff (ruff check)

`ruff check --fix` is a linter: it finds unused imports, variables assigned and never
read, and the pitfalls selected in `pyproject.toml` under `[tool.ruff.lint]`. What it can
fix, it fixes; the rest blocks the commit with file, line and rule. Read those messages
to the student the first few times: they are a code review for free.

### Hook 3 — typecheck (ty)

*"Do you know what a type checker does?"*

ty reads `.py` files and checks that every function is called with the kind of thing it
expects and returns the kind of thing its caller expects — without running anything.
`def train(model: nn.Module, epochs: int) -> float` is a promise; ty checks it is kept.
In a course about tensors, the habit of writing down what goes in and what comes out is
the same habit as writing down a tensor's shape.

### Hook 4 — clean-jupyter: outputs are cleared on commit

*"Do you know what happens to a notebook's output when you commit it?"*

The hook runs `jupyter nbconvert --clear-output --inplace` on every staged notebook and
re-stages it. The committed notebook has no outputs, and neither does the copy on disk.

Why, and say all four:

1. **Outputs make every diff unreadable.** An output cell is a blob of JSON, base64
   images and execution counts. A one-line code change shows up as hundreds of changed
   lines.
2. **They bloat the repository.** Every plot is an image stored in git's history for
   ever, once per commit that re-ran it.
3. **They cause merge conflicts on every pull.** Two people who ran the same notebook
   changed the same output lines, so merging gives a conflict on nothing.
4. **They can leak data.** A printed dataframe, a token in a stack trace, a path with a
   username: whatever a cell printed is committed with it.

The hook does it for them. The consequence for their workflow: a notebook must run
**top to bottom from a fresh kernel**, because that is the only state it is saved in.
Results that matter go to a file or to mlflow, not to a cell output.

### Hook 5 — notebooktester: every staged notebook runs

*"Given hook 4 — how would you know a notebook still runs before you commit it?"* Let
them get to "run it" themselves.

The hook runs `notebooktester` on every staged notebook: top to bottom, on a fresh
kernel, with a timeout per notebook. Results are cached in `.notebookcache`, so a
notebook that passed and has not changed is skipped.

Now `param()`. Many cells of this course are legitimately slow: training for 20 epochs,
a search over 24 trials, several seeds. The tester only needs to know the code works,
not to get the real result. So notebooktester exports one function:

```python
from notebooktester import param

EPOCHS = param(20, test=1)
N_TRIALS = param(24, test=2)
```

`param(full, test=...)` returns `full` when the student runs the notebook and `test`
when notebooktester runs it (the tester sets the environment variable
`NOTEBOOKTESTER_RUNNING`, and `param` reads it). Show them the lesson notebooks use it:
`grep -rn "param(" notebooks/1_pytorch_intro`.

Why **every** expensive constant goes through it: one constant left out — an epoch
count, a dataset size, a number of seeds — and the test runs the full training, hits
the timeout, and blocks the commit. A test suite that is too slow to run is a test
suite nobody runs.

And one consequence for their results: a run logged while notebooktester runs is a run
on test settings (one epoch, two trials). The course's `runlog.log_run` tags such a run
`test_run=true`, and the submit script leaves it out. **A test-mode run is never
submitted**, because its numbers mean nothing.

### Hook 6 — lychee: dead links

*"Do you know what a link checker does?"*

`lychee` follows every link in the repository's markdown and reports the dead ones,
including links from one file to another. It is a standalone program, not a Python
package, so it is not part of `uv sync`; it is the only hook the student may decline.

Offer it, and **discuss how before installing anything** — it goes through the system's
package manager, which is the student's territory. Check what they have (`which brew`,
`which winget`, `which scoop`) and ask:

- **Mac**: `brew install lychee`.
- **Windows**: `winget install --id lycheeverse.lychee`.
- **Linux, and anything else**: a prebuilt binary from
  https://github.com/lycheeverse/lychee/releases into `~/.local/bin`.

If they decline, skip the hook with a local override that lefthook reads and git
ignores (it is in `.git/info/exclude` since §0):

```bash
printf 'pre-commit:\n  commands:\n    lychee:\n      skip: true\n' > .lefthook-local.yml
```

Show them the file: `.lefthook-local.yml` merges over the teacher's `.lefthook.yml` and
lives only on this machine. Deleting it turns the hook back on.

### Install

Lefthook is a command-line tool, not a project dependency:

```bash
uv tool install lefthook
lefthook install
```

Read it out: `uv tool install` puts the program in `~/.local/bin`, isolated from every
project. `lefthook install`, in the repository root, writes a small script into
`.git/hooks/pre-commit` that hands control to lefthook, which reads `.lefthook.yml`.
`.git/hooks` is not versioned, which is why every clone needs this once.

### Verify — with a real commit

Do not verify by inspecting. Make the hooks fire. On their branch:

```bash
printf 'import os\nx=1\nprint( "hello" )\n' > hook_demo.py
git add hook_demo.py
git commit -m "hook demo"
```

Predict with them first: what will ruff say about `import os`? About the spacing? Then
run it and read the output together: the formatter rewrites the spacing, ruff removes
the unused import, ty finds nothing, the notebook hooks skip (no `.ipynb` staged). If
the commit was refused, read why, fix, and commit again. Then clean up:

```bash
git rm -q hook_demo.py && git commit -q -m "remove hook demo"
```

Both commits stay on their own branch. Then check the mechanism:

```bash
lefthook version && grep -q lefthook .git/hooks/pre-commit && echo "hooks installed"
```

### Then ask

Two questions, and do not tick the box until both answers hold:

1. *"What does `param(20, test=1)` return when you run the notebook, and when
   notebooktester runs it? Why does every expensive constant go through it, and what
   happens to a run made in test mode?"* Accept: *20 for me, 1 under the tester; because
   one slow constant makes the test time out; it is tagged `test_run` and not
   submitted.*
2. *"Why does the commit clear notebook outputs?"* Accept at least two of the four
   reasons (unreadable diffs, a bloated repository, merge conflicts, leaked data), and
   *the hook does it for me.*

**Remember** (memory file):

- Pre-commit hooks (lefthook) run ruff format, ruff check, ty, clear notebook outputs,
  notebooktester and lychee. A refused commit is feedback: read it, fix it, commit again.
- Notebooks are committed without outputs and must run top-to-bottom on a fresh kernel.
- Any constant whose size scales runtime (epochs, steps, trials, seeds, dataset size)
  gets `from notebooktester import param` and `X = param(full, test=small)`.
- Runs made under notebooktester are tagged `test_run` and are never submitted.
- `lychee` is installed through the student's own package manager, or skipped in
  `.lefthook-local.yml`.

**CHECKPOINT.** M6 is done when `lefthook version` answers, `.git/hooks/pre-commit`
mentions lefthook, the demo commit went through the hooks, and the student answered
both questions. Tick the box.

---

## M7 — Atlas

### Ask first

*"Milestone 7 of 9: Atlas. Have you used Atlas before?"*

For a no: Atlas draws a repository as a picture — the places where data rests (files,
tables, types in memory) and the functions between them — and checks the code against a
handful of design principles. Like the coach in M8, every finding it reports ends in a
question for the author, not a fix.

The effect they will notice: `atlas-code view .` opens a picture of a repository in the
browser, and their assistant can read the same picture through Atlas's MCP server and
point at a box on it.

### The command line

The command is `atlas-code`. Install it, or update it if it is there already —
updating is the same install command again:

Mac and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/raoulg/atlas/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/raoulg/atlas/main/install.ps1 | iex
```

Homebrew users may prefer the formula (ask first, it is their package manager):

```bash
brew install raoulg/tap/atlas-code     # and later: brew upgrade atlas-code
```

The script puts the binary in `~/.local/bin`.

### Is it current?

```bash
atlas-code --version
gh release view --repo raoulg/atlas --json tagName --jq .tagName
```

The first prints `atlas-code <version> (<commit>)`, the second the latest release tag,
`v<version>`. They must match. If they do not, run the install command again (or
`brew upgrade atlas-code`).

### Atlas Studio, if they want the desktop app

Ask: *"Would you like the desktop app as well — the picture, its history and a terminal
in one window?"* It is optional; it carries its own `atlas-code` and updates itself
(Atlas Studio › Check for Updates…).

Mac, with Homebrew (into `/Applications`):

```bash
brew install --cask raoulg/tap/atlas-studio
```

or with the script (into `~/Applications`):

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/raoulg/atlas/main/install-studio.sh)"
```

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/raoulg/atlas/main/install-studio.ps1 | iex
```

Linux (x86_64): the same `install-studio.sh` line as the Mac script.

Studio is not signed with a paid identity. An app from Homebrew or from a disk image
downloaded in a browser is blocked on its first start: on macOS the student opens it
once, then allows it in System Settings › Privacy & Security › Open Anyway; on Windows
SmartScreen asks, and they choose More info, then Run anyway. What the install scripts
put in place opens without asking.

### Register its MCP server

Once, for every project:

```bash
claude mcp add --scope user atlas -- atlas-code mcp          # Claude Code
codex mcp add atlas -- "$(command -v atlas-code)" mcp       # Codex
```

Codex starts its servers without the shell's `PATH`, which is why it gets the full path.
Studio users can do the same from Studio's Settings › Coding agents.

### Verify (you run this)

```bash
atlas-code --version
claude mcp list            # or: codex mcp list
```

`atlas` must be listed as connected. Then show it working: in the course repository,
`atlas-code view .` opens the picture in the browser; ask the student what one box on it
is.

**CHECKPOINT.** M7 is done when the version matches the latest release, `atlas` is
registered with their assistant, and they have seen the picture once. Tick the box.

---

## M8 — The coach

### Ask first

*"Milestone 8 of 9: the coaching server. Do you know what an MCP server is?"*

For a no: an MCP server is a small program that gives an assistant tools it does not
have by itself. This repository registers `codestyle` (*is my code any good?*), which
holds the course's code guidelines and review rubric. It is a coach, not an oracle: it
asks questions, waits for the student's own words, and only then writes code.
`CLAUDE.md` and `AGENTS.md` in the repository root spell out that contract.

The effect they will notice: their assistant will sometimes decline to just write the
model or the search space, and ask them a question instead. That is the tool working.

### How it works

Open `.mcp.json` together. Its entry is one command:

```
uv run --no-project https://raw.githubusercontent.com/raoulg/codestyle/$CODESTYLE_REF/codestyle_mcp.py
```

with `CODESTYLE_REF=stable`. Read it out: uv downloads a single Python script from
GitHub at the `stable` tag, builds an isolated environment for that script's own
dependencies (the first start is slower; after that it is cached), and runs it. Nothing
lands in the project's `.venv`, and `stable` moves to each new release. No uv, no coach:
another reason for M3.

### Connect

- **Claude Code**: start `claude` in the `MADS-MachineLearning-course` folder. It sees
  `.mcp.json`, asks whether to use the project's MCP servers, and the student approves
  once.
- **Codex**: reads `.codex/config.toml` in the repository, which holds the same server.
- **Cursor**: `mkdir -p .cursor && cp .mcp.json .cursor/mcp.json` (`.cursor/` is in
  `.git/info/exclude` since §0). Settings → MCP shows the server with a green dot once
  it has started.

### Verify (you run this)

```bash
claude mcp list            # or: codex mcp list
```

`codestyle` must say connected. If you are the assistant running inside the session,
prove it instead: call one of its tools (list the codestyle topics) and tell the student
one title.

A server that fails to connect is diagnosed by running its command by hand: copy the
`uv run --no-project ...` line out of `.mcp.json`, prefix the environment variable, run
it in the terminal. A Python traceback in the first lines is the actual error; a silent
wait means it started fine and is waiting for a client.

### Then ask

*"What will the coach do when you ask your assistant to write the model for you?"*

Accept: *ask me questions first / make me say what the model should do and why.* If
they say "write the model", point at `CLAUDE.md` and try again.

**Remember** (memory file):

- `codestyle` from `.mcp.json` is the course's code coach. It asks first, and the
  student answers in their own words.
- The student designs the search space and states the hypothesis; the assistant coaches.

**CHECKPOINT.** M8 is done when `codestyle` is connected and the student answered the
question. Tick the box.

---

## M9 — The class runs repository

### Ask first

*"Last one, 9 of 9: the class runs repository. Do you know what a fork and a pull
request are?"*

For a no: a fork is your own copy of someone else's repository on GitHub, under your
login, that you can push to. A pull request asks the owner of the original to take your
changes in. Nobody is invited to the class repository: each student works in a fork,
and hands in through a pull request.

Why the course has it: between lessons, each student submits the training runs of the
lesson's exercise — parameters, scores, the architecture, no weights or data. The next
lesson opens with what the whole class built and how much of each search winner
survived a rerun over seeds. The runs are filed under their GitHub login (M2).

The effect they will notice: their runs from their own mlflow show up, with everyone
else's, on the screen at the start of the next lesson.

### Fork and clone

Find the login first (M2 already showed it):

```bash
gh api user --jq .login
```

Then, from the course repository folder, with `<login>` filled in:

```bash
gh repo fork raoulg/mads-ml-runs-2026-11 --clone=false
git clone https://github.com/<login>/mads-ml-runs-2026-11.git ../mads-runs
git -C ../mads-runs remote add upstream https://github.com/raoulg/mads-ml-runs-2026-11.git
```

Read it out: make a fork under their own login on GitHub; clone that fork next to the
course repository, as `../mads-runs`, where the submit script looks for it; and add the
teacher's repository as a second remote called `upstream`, to pull the class's runs from.
`origin` is theirs to push to; `upstream` is the teacher's and only read.

### Verify (you run this)

```bash
git -C ../mads-runs remote -v
gh repo view <login>/mads-ml-runs-2026-11 --json isFork
```

The first must show `origin` pointing at `<login>/mads-ml-runs-2026-11` and `upstream` at
`raoulg/mads-ml-runs-2026-11`; the second must say `"isFork": true`.

Nothing is submitted now. Submitting a lesson's runs is done at the end of that
lesson's exercise notebook, by following `docs/submit-runs.md`.

**Remember** (memory file):

- The student's GitHub login is `<login>`.
- The class runs clone is `../mads-runs` (relative to the course repository): a fork,
  with `origin` the student's fork and `upstream` `raoulg/mads-ml-runs-2026-11`.
- Submitting a lesson's runs means following `docs/submit-runs.md` in the course
  repository. Never submit test-mode runs.

**CHECKPOINT.** M9 is done when both remotes show and the fork is confirmed. Tick the
last box and tell the student the setup is finished.

---

## 3. After setup

### Show them their assistant's rulebook

Open the memory file and read it through together. Every line came from a milestone
they did. If a line surprises them, that milestone needs another two sentences now, not
in week four.

### Git practice, if they want it

M2 gave the model and M5 the workflow; neither makes anyone fluent. Offer one thing
that fits how they learn: a few drills in a throwaway repository (`git init`, commit,
branch, merge, and one deliberate merge conflict), https://learngitbranching.js.org in
the browser, or `references/01_git crash course.pdf` (Dutch, the teacher's own).

---

## 4. Troubleshooting index

| Symptom | Almost always |
|---|---|
| `uv: command not found` right after install | New PATH, old terminal. Open a new one |
| `uv sync` fails on torch or ray on a Mac | An Intel Mac: no wheels. Use a Linux VM |
| Notebook kernel list has no `.venv` | Editor opened in the parent folder, or `uv sync` never ran; reload window |
| Imports fail in the notebook but work in the terminal | Wrong kernel, or a second Python (conda) got in the way; select `.venv` |
| `Please tell me who you are` on first commit | M2 identity not set |
| Commit refused, mentions `ruff` or `ty` | Read the message; it says file, line and rule. Fix, `git add`, commit again |
| Commit refused, `lychee: command not found` | Install lychee, or skip it in `.lefthook-local.yml`; M6 hook 6 |
| Commit of a notebook times out | notebooktester runs it with full settings: a constant is missing its `param()` |
| `git pull` on `master` says "would be overwritten" | They edited or committed on `master`. Commit on their branch, never again |
| Merge conflict in a lesson notebook after `git merge master` | They edited the teacher's file in place. Resolve, then rename their copy |
| `atlas-code --version` older than the latest release | Run the install command again, or `brew upgrade atlas-code` |
| `atlas` or `codestyle` "failed to connect" | Run its command by hand; the traceback is the error |
| `gh repo fork` says the fork exists | It does; go on with the clone |
| `git push` to `../mads-runs` refused | `gh auth setup-git` was not run (M2), or `origin` is not their fork |

If something falls outside this table, do not guess in a loop. Read the actual error,
say what you think it means, and if two attempts do not fix it, tell the student to ask
their teacher.
