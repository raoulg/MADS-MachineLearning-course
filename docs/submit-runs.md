# Submitting your runs: instructions for a coding agent

A student pointed you here to submit their mlflow runs for one lesson of the MADS Machine
Learning course. Follow these steps in order. Run every command from the root of the
course repository (the directory with `pyproject.toml` and `scripts/`).

The script does the mechanical work and validates it. Your job is to set things up once,
show the student what will be submitted, and handle the errors described below.

## What you are submitting

Each lesson's exercise notebook logs its runs to the student's local mlflow store,
`artefacts/mlflow/mlruns.db`, in an experiment named `lesson<N>-...` (for lesson 1:
`lesson1-exercises`). `scripts/submit_runs.py` turns each finished run into one small JSON
file holding its params, metrics, tags and architecture, and pushes it to the class runs
repository at `lesson<N>/<github-login>/<run_id>.json`. No model weights and no data are
submitted.

## 1. Find the student's GitHub login

```bash
gh api user --jq .login
```

If `gh` is not installed or not logged in, ask the student for their GitHub login. Do not
guess it. It must be the account that was invited to the runs repository.

## 2. One time: clone the runs repository

The runs repository of the current cohort (November 2026) is
https://github.com/raoulg/mads-ml-runs-2026-11. It is public, so anyone can read it; pushing needs Write access, which
the student gets by accepting the GitHub invitation from the teacher. If the push is
refused, ask the student whether they accepted it.

Clone it next to the course repository, where the script looks for it:

```bash
git clone https://github.com/raoulg/mads-ml-runs-2026-11.git ../mads-runs
```

Skip this step if `../mads-runs` already exists. If the student keeps the clone elsewhere,
pass `--runs-repo <path>` to every command below.

## 3. Show the student what will be submitted

```bash
uv run scripts/submit_runs.py --lesson <N> --student <login> --list
```

This changes nothing. Show the student the list: one line per run, with its role
(baseline, trial, final, ablation, hypothesis) and its main metrics. By default every
finished run of the lesson's experiments is included, and runs from automated notebook
tests are left out. That default is what the teacher wants. Narrow it only if the student
asks, with `--run-id <id>` (repeatable) or `--experiment <name>`.

If the list is empty, the notebook has not been run yet, or it was run with a different
`MLFLOW_TRACKING_URI`. Tell the student; do not fabricate runs.

## 4. Submit

```bash
uv run scripts/submit_runs.py --lesson <N> --student <login>
```

It pulls the runs repository, writes and validates the files, commits, and pushes. If
another student pushed in the meantime, it rebases and retries by itself. Report the last
lines of its output to the student. Running it again later is safe: runs already
submitted are skipped, and new runs are added.

## Rules

- Do not edit, rename or delete files in the runs repository by hand. They are generated;
  fix the run in mlflow (or rerun the notebook) and submit again.
- Write only in `lesson<N>/<login>/`. Never touch another student's directory. The
  repository's check fails any push that does.
- Never force-push, and never rewrite the runs repository's history.
- Do not pass `--include-test-runs`. Those runs come from automated notebook tests with
  shortened settings and mean nothing to the class.
- Do not add model weights, datasets or other files to the runs repository.

## When something fails

| Message | What to do |
|---|---|
| `cannot tell who you are` | Pass `--student <login>` (step 1). |
| `... is not a git clone of the class runs repository` | Do step 2, or pass `--runs-repo`. |
| `git pull ... failed` / `Repository not found` / `403` | The student has not accepted the invitation, or git is not authenticated for GitHub. `gh auth login` followed by `gh auth setup-git` usually fixes the second. |
| `no experiment matches lesson<N>-*` | The notebook has not logged anything yet; it lists the experiments it did find. |
| `does not fit the submission format` | A run has something the format does not accept (the message says what). Tell the student; do not work around it. |
| `push failed after 4 attempts` | Show the git error to the student. Do not force-push. |
| The check on GitHub fails after a push | Open the failed run's log (`gh run list -R raoulg/mads-ml-runs-2026-11`, `gh run view <id> --log-failed`). It names the file and the problem. |
