# Submitting your runs: instructions for a coding agent

A student pointed you here to submit their mlflow runs for one lesson of the MADS Machine
Learning course. Follow these steps in order. Run every command from the root of the
course repository (the directory with `pyproject.toml` and `scripts/`).

The script does the mechanical work and validates it. Your job is to set things up once,
show the student what will be submitted, and handle the errors described below.

## What you are submitting

Each lesson's notebooks log their runs to the student's local mlflow store,
`artefacts/mlflow/mlruns.db`, in experiments named `lesson<N>-notebook` (the notebook as
handed out) and `lesson<N>-study` (the student's own research). `scripts/submit_runs.py`
turns each finished run into one small JSON file holding its params, metrics, tags and
architecture, at `lesson<N>/<github-login>/<run_id>.json`. No model weights and no data
are submitted.

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

## 2. One time: fork the runs repository and clone the fork

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

This changes nothing. Show the student the list: one line per run, with its role
(baseline, trial, final, ablation, hypothesis), its study tag if it has one, and its main
metrics; a score over seeds shows as `accuracy=<mean> ± <std> (n=<seeds>)`. By default
every finished run of the lesson's experiments is included, and runs from automated
notebook tests are left out. That default is what the teacher wants. Narrow it only if
the student asks, with `--run-id <id>` (repeatable) or `--experiment <name>`.

If the list is empty, the notebook has not been run yet, or it was run with a different
`MLFLOW_TRACKING_URI`. Tell the student; do not fabricate runs.

## 4. Submit

```bash
uv run scripts/submit_runs.py --lesson <N> --student <login>
```

It fetches the class repository, works on the branch `lesson<N>` of the clone, writes and
validates the files, commits, pushes the branch to the fork, and opens a pull request
against the class repository titled `lesson <N>: <login>`. Its last line is the pull
request's URL; show it to the student.

Running it again later is safe. A run already in the class repository or on the branch is
skipped. New runs are added to the branch, and so to the open pull request; once that
pull request is merged, the next submission opens a new one.

Without `gh`, add `--no-pr`: the script pushes the branch and prints the compare URL where
the student opens the pull request in the browser.

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

## Rules

- Do not edit, rename or delete files in the runs clone by hand. They are generated; fix
  the run in mlflow (or rerun the notebook) and submit again.
- Write only in `lesson<N>/<login>/`. Never touch another student's directory, and never
  change anything outside `lesson<N>/`. The check fails any pull request that does.
- Never force-push, and never rewrite the history of the fork or the class repository.
- Do not pass `--include-test-runs`. Those runs come from automated notebook tests with
  shortened settings and mean nothing to the class.
- Do not add model weights, datasets or other files to the runs repository.

## When something fails

| Message | What to do |
|---|---|
| `cannot tell who you are` | Pass `--student <login>` (step 1). |
| `these disagree about who you are` | `--student`, the login of `gh` and the owner of `origin` must be one account. Pass the student's own login, log `gh` in as the student (`gh auth login`), or point `origin` at the student's fork (`git -C ../mads-runs remote set-url origin https://github.com/<login>/mads-ml-runs-2026-11.git`). |
| `... is not a git clone of your fork` | Do step 2, or pass `--runs-repo`. |
| `... has no upstream remote` | Run the two commands the message names (the last two of step 2). |
| `... has uncommitted changes` | Someone edited the clone by hand. Show `git -C ../mads-runs status` to the student; do not discard their work without asking. |
| `git fetch ... failed` / `Repository not found` / `403` | The fork does not exist yet, `origin` points somewhere else, or git is not authenticated for GitHub. `gh auth login` followed by `gh auth setup-git` fixes the last. |
| `git push origin lesson<N> failed` | Show the git error to the student. Do not force-push. |
| `merging ... into your branch failed` | Show it to the teacher; do not resolve it by hand. |
| `pushed, but opening the pull request failed` | Open the pull request in the browser at the URL the message gives. |
| `no experiment matches lesson<N>-*` | The notebook has not logged anything yet; it lists the experiments it did find. |
| `does not fit the submission format` | A run has something the format does not accept (the message says what). Tell the student; do not work around it. |

The check's comment on the pull request names each file with one of these problems:

| Problem in the comment | What it means |
|---|---|
| `outside lesson<N>/<student>/<run_id>.json; a submission changes nothing else` | The pull request changes another file (a README, `.github/`, ...). Take that change out of the branch. |
| `written by <a>, but this directory belongs to <b>` | A file in someone else's directory. The pull request must hold only the author's own runs. |
| `deleted by <a>, but this directory belongs to <b>` | The branch removes another student's file. Files are never deleted by hand; show this to the teacher. |
| `this run id is already submitted by <other>` | The run is another student's. Submit only runs the student made. |
| `this file says it is <a>'s run; it sits in <b>'s directory` | The file was copied or edited by hand. Submit again with the script. |
| `not valid JSON (...): submit again, do not edit by hand` | The file was edited by hand. Submit again with the script. |
| `a symbolic link, not a file` | The branch holds a link instead of a file. Submit again with the script. |
| any other field message (`run.metrics...`, `student: ...`) | The file does not fit the format; show it to the student. |
