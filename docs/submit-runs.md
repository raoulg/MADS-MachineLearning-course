# Submitting your runs: instructions for a coding agent

A student pointed you here to submit their Optuna runs for one lesson of the MADS Machine
Learning course. Follow these steps in order. Run every command from the root of the
course repository (the directory with `pyproject.toml` and `scripts/`).

The script does the mechanical work and validates it. Your job is to set things up once,
show the student what will be submitted, and handle the errors described below.

## What you are submitting

Each lesson's notebooks log their runs to the student's local Optuna store,
`artefacts/optuna/studies.db`, through `notebooks/1_pytorch_intro/optunalog.py`: a run is
one trial of a study, and every trial carries a `run_id`. The lesson's exercise logs to
studies whose names start with `lesson<N>-notebook` (the part of the exercise as handed
out: `lesson1-notebook` for its search, `lesson1-notebook-runs` for the baseline, the
winner and the ablation) and `lesson<N>-study` (the student's own study:
`lesson1-study-<name>` for its search, `lesson1-study-<name>-runs` for its baseline,
winner and arms). Those are what `scripts/submit_runs.py --lesson <N>` submits. The
lesson's other notebooks log to studies of their own (`lesson1-search`,
`lesson1-representation-quality`, ...), which are not submitted unless named with
`--experiment`. The script turns each finished or pruned trial into one small JSON file
holding its params with their distributions, its value and its value per epoch (or per
repeat), its attrs, the attrs of its study, and its architecture and design script, at
`lesson<N>/<github-login>/<run_id>.json`. No model weights and no data are submitted. A
trial without a `run_id` was not logged through `optunalog`; it is skipped, with one line
saying how many.

The files reach the class runs repository through a pull request from the student's own
fork. The student needs no access to the class repository: a check on the pull request
validates the files and merges it.

## 1. Find the student's GitHub login

```bash
gh api user --jq .login
```

If `gh` is not installed or not logged in, ask the student for their GitHub login. Do not
guess it. The script also reads it from the fork's URL and stops when the sources
disagree.

## 2. One time: set up git and gh, fork the runs repository, clone the fork

git must be able to push to GitHub as the student, and commits need an author. Check
both, and set up what is missing:

```bash
gh auth status                          # logged in? If not:
gh auth login                           # the student logs in, in the browser
gh auth setup-git                       # git pushes to GitHub with gh's login
git config --global user.name           # empty? Then set both, with the student:
git config --global user.name "<the student's name>"
git config --global user.email "<login>@users.noreply.github.com"
```

The `users.noreply.github.com` address keeps the student's own e-mail address out of
the public runs repository. Without an identity the script commits as
`<login> <<login>@users.noreply.github.com>` by itself. Without `gh`, git needs another
way to authenticate to GitHub (a credential helper or SSH); the student sets that up.

The runs repository of the current cohort (November 2026) is
https://github.com/raoulg/mads-ml-runs-2026-11. It is public. The student forks it, and
the clone of the fork, next to the course repository, has two remotes: `origin` (the
fork) and `upstream` (the class repository).

```bash
gh repo fork raoulg/mads-ml-runs-2026-11 --clone=false
git clone https://github.com/<login>/mads-ml-runs-2026-11.git ../mads-runs
git -C ../mads-runs remote add upstream https://github.com/raoulg/mads-ml-runs-2026-11.git
git -C ../mads-runs fetch upstream
```

Without `gh`, the student forks it in the browser (the Fork button on the repository's
page) and you run the last three commands.

Skip this step if `../mads-runs` already exists with both remotes
(`git -C ../mads-runs remote -v`). If the student keeps the clone elsewhere, pass
`--runs-repo <path>` to every command below.

## 3. Show the student what will be submitted

```bash
uv run scripts/submit_runs.py --lesson <N> --student <login> --list
```

This changes nothing. Show the student the list: one line per run, with its study, its
role (baseline, trial, final, ablation, hypothesis), its name and its outcome; a run
repeated a few times shows as `accuracy=<mean> ± <spread between runs> (n=<repeats>)`, a
trial of a search as `accuracy=<value>`, a pruned trial as `pruned after <n> epochs`. By
default every finished or pruned trial of the studies `lesson<N>-notebook*` and
`lesson<N>-study*` is included. Trials made while the notebook tester runs go to a store
of their own, `artefacts/optuna/test.db`, which is never read. That default is what the
teacher wants. Change it only if the student asks: `--run-id <id>` (repeatable) for single
runs, or `--experiment <study>` (repeatable, a name or a pattern such as
`'lesson1-search*'`) for exactly the studies named, in place of the two patterns, for
example `--experiment lesson1-search --experiment lesson1-search-runs`. `--store <file>`
reads another Optuna store than `artefacts/optuna/studies.db`.

If the list is empty, the notebook has not been run yet, or it was run from another
checkout (each checkout has its own store). Tell the student; do not fabricate runs.

## 4. Submit

```bash
uv run scripts/submit_runs.py --lesson <N> --student <login>
```

A student who submits by hand types `make share LESSON=<N>`: the same script, with the
login from `gh` or from the owner of the fork (`make share LESSON=<N> ARGS=--list` shows the list of
step 3).

It fetches the class repository, works on the branch `lesson<N>` of the clone, writes and
validates the files, commits, pushes the branch to the fork, and opens a pull request
against the class repository titled `lesson <N>: <login>`. Its last line is the pull
request's URL; show it to the student.

Running it again later is safe. A run already in the class repository or on the branch is
skipped. New runs are added to the branch, and so to the open pull request; once that
pull request is merged, the next submission opens a new one.

A run that is gone from the student's store (its study was deleted with
`uv run optuna delete-study`) is also removed from the branch, if it is not in the class
repository yet; the script lists what it removed. That is how a pull request held up by a
bad run gets out: delete its study and run the notebook again. This happens only
without `--run-id` and `--experiment`, when the script sees all of the lesson's runs.

Without `gh`, or with `--no-pr`, the script pushes the branch and prints the compare URL
where the student opens the pull request in the browser. With `--no-push` it writes and
commits on the branch and stops there.

## 5. The check on the pull request

The pull request runs the check `validate`. It reads every file the pull request changes
and accepts only valid submission files in `lesson<N>/<login>/`, where `<login>` is the
author of the pull request. When it passes, the pull request is merged and the runs are
in the class repository. When it fails, it posts the problems as a comment on the pull
request. Read them with

```bash
gh pr view <url> --comments
```

and act on them with the table below. Fixing a run and submitting again pushes to the
same pull request, and the check runs again.

## 6. Look at the class's runs, and build on one

The runs the class repository holds (its `main`: every merged pull request) can be
imported into a separate local Optuna store, `artefacts/optuna/class.db`, and looked at:

```bash
make dashboard LESSON=<N>
```

That writes the class page, `artefacts/class-analysis/lesson<N>/index.html` (the class's
runs in pictures), opens it in the browser, and then runs
`uv run scripts/collect_runs.py --lesson <N> --no-pull --ui`, which rebuilds `class.db`
and opens optuna-dashboard on it (`http://127.0.0.1:8089`). The page fetches `main` from
the clone's `upstream`, and the import reads the files from there, so it works on any
branch the clone is on and changes nothing in it. The class store is rebuilt from `main`
on every run, so it holds exactly what `main` holds. Without `--ui`, `collect_runs.py`
only imports. To read one student's own runs next to the class's, run
`make optuna-dashboard` (your own store, port 8088) in a second terminal.

In the class store the handed-out studies of the exercise (`lesson<N>-notebook` and
`lesson<N>-notebook-runs`) hold every student's trials; every other study is kept per
student, as `<study>@<login>`. Each imported trial carries the user attr `submission_ref`,
for example `lesson1/bob/0f3c...` (shown on the trial's page in optuna-dashboard), and
`student`. When the student builds on a classmate's run, copy that attr from the trial in
optuna-dashboard and pass it to `log_run`; the student names the run, you do not pick one
for them:

```python
log_run("deeper-head", "final", scores={"accuracy": per_repeat},
        builds_on="lesson1/bob/0f3c...", design="my_design.py")
```

`builds_on` takes one `submission_ref` or a list of them and is written as the user attr
`builds_on`; an entry that is not a `submission_ref` raises an error before anything is
written. `design` is the path of the student's design script, the file that defines their
architectures; it is submitted with the run as the artifact `design.py`. Nothing runs it:
it is there to be read. `record` (the call a search's objective makes on its trial) takes
the same two arguments.

The imported runs live in `class.db`, never in the student's own store, and the submit
script never submits them: a trial that carries `submission_ref` or `source_run_id` came
from the class, and is left out with one line saying so.

## Rules

- Do not edit, rename or delete files in the runs clone by hand. They are generated; fix
  the run (rerun the notebook) and submit again.
- Write only in `lesson<N>/<login>/`. Never touch another student's directory, and never
  change anything outside `lesson<N>/`. The check fails any pull request that does.
- Never force-push, and never rewrite the history of the fork or the class repository.
- Do not point `--store` at `artefacts/optuna/test.db`. Those runs come from automated
  notebook tests with shortened settings and mean nothing to the class.
- Do not add model weights, datasets or other files to the runs repository.

## When something fails

| Message | What to do |
|---|---|
| `cannot tell who you are` | Pass `--student <login>` (step 1). |
| `these disagree about who you are` | `--student`, the login of `gh` and the owner of `origin` must be one account. Pass the student's own login, log `gh` in as the student (`gh auth login`), or point `origin` at the student's fork (`git -C ../mads-runs remote set-url origin https://github.com/<login>/mads-ml-runs-2026-11.git`). |
| `... is not a git clone of your fork` | Do step 2, or pass `--runs-repo`. |
| `... has no upstream remote` | Run the two commands the message names (the last two of step 2). |
| `... has uncommitted changes` | Someone edited the clone by hand: a tracked file, or a new file under `lesson*/` (other new files do not count). Show `git -C ../mads-runs status` to the student; do not discard their work without asking. |
| `removed N run(s) that are no longer in your Optuna store` | Not an error: those runs are gone from the store (their study was deleted), so they leave the branch and the pull request. If that was not meant, run the notebook again and submit again. |
| `` `gh` is not installed, so the pull request is not opened for you `` | The branch is pushed. Open the pull request at the URL printed next. |
| `lesson<N> holds no runs that the class repository lacks` | Everything on the branch is in the class repository already. A pull request still open from the branch has nothing left to merge; the student may close it. |
| `git fetch ... failed` / `Repository not found` / `403` | The fork does not exist yet, `origin` points somewhere else, or git is not authenticated for GitHub. `gh auth login` followed by `gh auth setup-git` fixes the last. |
| `git push origin lesson<N> failed` | Show the git error to the student. Do not force-push. |
| `merging ... into your branch failed` | Show it to the teacher; do not resolve it by hand. |
| `pushed, but opening the pull request failed` | Open the pull request in the browser at the URL the message gives. |
| `no study matches ['lesson<N>-notebook*', 'lesson<N>-study*']; this store has [...]` | The exercise has not logged anything yet, or in another checkout; the message lists the studies the store does hold. |
| `no Optuna store at ...` | The lesson's exercise notebook has not run yet in this checkout, or `--store` names a file that does not exist. |
| `skipped run <id> (<name>), it does not fit the submission format: ...` | That run is left out and the others are submitted. The message says what the format does not accept; tell the student, and do not work around it. Fix the cause in the notebook, run it again, and submit again. |
| `not submitted: N run(s) imported from the class repository` | Not an error: trials imported by `collect_runs.py` ended up in the student's store (for example with `--store` pointing at `class.db`). They are a classmate's runs and are left out; the student's own runs are submitted. |
| `not submitted: N trial(s) without a run_id` | Not an error: those trials were not logged through `optunalog`, so they have no id to submit under. Running the notebook again logs them with one. |
| `... has no upstream/main` | `collect_runs.py` with `--no-pull` on a clone that never fetched; leave out `--no-pull`. |

The check's comment on the pull request names each file with one of these problems:

| Problem in the comment | What it means |
|---|---|
| `outside lesson<N>/<student>/<run_id>.json; a submission changes nothing else` | The pull request changes another file (a README, `.github/`, ...), or a path with characters a submission never has (shown quoted). Take that change out of the branch. |
| `written by <a>, but this directory belongs to <b>` | A file in someone else's directory. The pull request must hold only the author's own runs. The directory must be spelled exactly as the login, upper and lower case included; the script uses `gh`'s spelling. |
| `deleted by <a>, but this directory belongs to <b>` | The branch removes another student's file. Files are never deleted by hand; show this to the teacher. |
| `this run id is already submitted by <other>` | The run is another student's. Submit only runs the student made. |
| `this run is already submitted by <other>, under another run id` | The run's content is another student's, relabelled. Submit only runs the student made. |
| `not a regular file (mode ...)` | The branch holds a link or something else instead of a file. Submit again with the script. |
| `the change touches N files, more than 200` | Too much in one pull request. Submit per study (`--experiment`), one pull request at a time. |
| `param key ...`, `attr key ...`, `study attr key ...`, `distribution key ...` | A name with characters the format does not accept (letters, digits, `_ - . /` and spaces only), or longer than 250. Rename it in the notebook and log the run again. |
| `... is N bytes, limit 20000` (a param, attr or study attr) / `limit 256000` (an artifact) | A value larger than a submission carries. Log a smaller one; for a design script, keep only the architectures in it. |
| `only ('.json', '.txt', ...) files are submitted` | An artifact of another kind. Weights and data are never submitted; log only text artifacts. |
| `experiment must be named lesson<N>-<lowercase-name>` | The study's name does not follow the convention. Name it like `lesson1-notebook` or `lesson1-study-<name>`. |
| `a COMPLETE trial has a value` | A finished trial without a value. Run the trial again. |
| `attr 'builds_on': ... is not a submission_ref` | `builds_on` names a run that is not written as `lesson<N>/<login>/<run_id>`. Copy the `submission_ref` attr from the classmate's trial in the class store and log the run again with it (below). |
| `attr 'builds_on' names this run itself` | A run cannot build on itself. Log it with the classmate's run it builds on (below). |
| `the path must be relative, without . or ..` | An artifact path that points outside the run. Log the artifact under a plain relative name. |
| `this file says it is <a>'s run; it sits in <b>'s directory` | The file was copied or edited by hand. Submit again with the script. |
| `not valid JSON (...): submit again, do not edit by hand` | The file was edited by hand. Submit again with the script. |
| `a symbolic link, not a file` | The branch holds a link instead of a file. Submit again with the script. |
| any other field message (`run.params...`, `student: ...`) | The file does not fit the format; show it to the student. |

A trial that is already logged cannot be changed in Optuna. `log_run` checks `builds_on` and
`design` before it writes anything, so a wrong one never reaches the store; when a run
needs another `builds_on`, the student logs it again through the notebook with the right
one, and submits again.
