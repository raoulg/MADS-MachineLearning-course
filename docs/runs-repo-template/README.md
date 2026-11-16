# MADS Machine Learning: class runs

The runs everyone in the class submitted, one JSON file per Optuna trial (schema `mads-run/2`):

    lesson<N>/<github-login>/<run_id>.json

## You are a student: point your coding agent at the instructions

In your clone of the [course repository](https://github.com/raoulg/MADS-MachineLearning-course),
tell your coding agent:

> follow docs/submit-runs.md for lesson 1

[`docs/submit-runs.md`](https://github.com/raoulg/MADS-MachineLearning-course/blob/master/docs/submit-runs.md)
is written for the agent. It walks it through the one-time setup (your own fork of this
repository, and a clone of that fork) and through the submission, and it says what every
message of the check means. Without an agent, the same two commands from the root of the
course repository:

    make share LESSON=1 ARGS=--list    # what would be submitted; changes nothing
    make share LESSON=1                # submit, as a pull request from your fork
    make dashboard LESSON=1            # the class's runs, as one page and in optuna-dashboard

Never edit a file here by hand. The files are generated from your Optuna store; a file
that was edited does not pass the check.

## What happens to a pull request

The check `validate` reads every file a pull request changes. It merges a pull request
whose files are valid runs in its author's own directory, and comments on one that is
not: a file in someone else's directory, a run another student already submitted, a file
that does not fit the format, or any change outside `lesson<N>/`. You need no access to
this repository: the pull request comes from your fork.

## Reading the class's runs

The class's runs are what `main` holds. `scripts/collect_runs.py` in the course
repository reads them from `main` (from the remote `upstream` in a clone of a fork) and
imports them into a local Optuna store, `artefacts/optuna/class.db`, for student and
teacher alike; `scripts/class_analysis.py` shows them as one page of pictures. Both run
behind `make dashboard`.

The format: `scripts/run_submission.py` in the course repository.
