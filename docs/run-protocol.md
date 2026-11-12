# The class run protocol

Students submit their training runs between lessons. Each lesson opens with mlflow showing
what the whole class did: which architectures were built, what they scored, and how much
of each search winner survived being rerun over seeds. There is no server. A GitHub
repository carries the runs; for the November 2026 cohort that is
https://github.com/raoulg/mads-ml-runs-2026-11 (public).

## How several people write to one mlflow without a server

mlflow normally needs a tracking server, because a shared database needs one process that
receives every write. This protocol has no shared database, so it needs no server:

1. **Every student keeps their own local mlflow store** (`artefacts/mlflow/mlruns.db` in
   their checkout of this repository). The notebooks log to it as usual, offline.
2. **A submission is a set of files in a pull request.** Each student forks the runs
   repository once. `scripts/submit_runs.py` writes each run as one JSON file into the
   student's clone of that fork, at `lesson<N>/<github-login>/<run_id>.json`, commits on
   the branch `lesson<N>`, pushes it to the fork, and opens a pull request against the
   class repository. Every student writes only into their own directory, so two pull
   requests never touch the same file and merge in any order.
3. **A check merges what is valid.** The runs repository's workflow validates every pull
   request and merges it when its files are valid runs in its author's own directory.
   Students need no access to the class repository, and the teacher merges nothing by
   hand.
4. **The teacher's collect step is the only writer to the class mlflow store.**
   `scripts/collect_runs.py` pulls the repository and imports every file into a separate
   local store (`artefacts/mlflow/class.db`). It is idempotent: a run already imported
   unchanged is skipped, and a run whose file changed replaces its earlier import. Run it
   as often as you like.

Git and GitHub do the work a tracking server would do: they receive writes
asynchronously, keep them in order, and authenticate who wrote what (the author of the
pull request). `mlflow ui` only ever reads a local
sqlite file.

```
 student laptop                  GitHub                                 your laptop
 notebook --log_run--> mlruns.db
 submit_runs.py --push--> alice/mads-ml-runs (fork), branch lesson1
                            |  pull request, checked and merged by validate.yml
                            v
                          raoulg/mads-ml-runs: lesson1/alice/<run>.json --pull--> collect_runs.py
                                               lesson1/bob/<run>.json             --> class.db
                                                                                  --> mlflow ui
                                                                                  --> class_analysis.py
```

## The pieces

| File | Who | What |
|---|---|---|
| `notebooks/<lesson>/runlog.py` | notebooks | `use_experiment` + `log_run`: logs a run in the protocol's shape |
| `scripts/run_submission.py` | both, and CI | the file format (pydantic) and the validator |
| `scripts/submit_runs.py` | student | local mlflow → JSON files → commit on `lesson<N>` → push to the fork → pull request |
| `docs/submit-runs.md` | student's coding agent | step-by-step instructions, including the one-time setup |
| `scripts/collect_runs.py` | teacher | pull → validate → import into `class.db` → optionally `mlflow ui` |
| `scripts/class_analysis.py` | teacher | the opening analysis: tables and figures for one lesson |
| `docs/runs-repo-template/` | teacher, once | README and the workflow that validates and merges pull requests |

## What a submission holds

One file per run, about 5 KB, schema `mads-run/1` (`uv run python
scripts/run_submission.py schema` prints the JSON schema):

- **lesson**, **student** (GitHub login), **submitted_at**.
- **code**: the git commit of the student's course checkout, whether it had uncommitted
  changes, and its remote. `source` says whether mlflow recorded the commit when the run
  started (`run`) or the script read it at submission (`submit`; notebooks do not record
  it, so this is the usual case).
- **run**: run id, name, experiment, status, start and end time, params, tags, the full
  history of every metric (so per-epoch curves survive), and the small text/JSON
  artifacts: in practice `architecture.json`, the layer table with input and output shapes.

mlflow's own system tags are dropped, because they carry the student's local username and
absolute paths. Artifacts other than `.json`/`.txt`/`.md`/`.yaml` files under 256 KB are
never submitted, so weights cannot end up in git.

## Logging conventions

The analysis reads what `runlog.log_run` writes. A notebook that follows the protocol logs
every run worth discussing through it, for example:

```python
from runlog import log_run, use_experiment

use_experiment("lesson1-notebook")
log_run("winner", "final", params=best, scores={"accuracy": per_seed_scores},
        model=model, example=one_batch[:2])
```

- **experiment**: `lesson<N>-notebook` for the runs of the notebook as handed out,
  `lesson<N>-study` for the student's own research. The submit script takes every
  experiment with the prefix `lesson<N>-`.
- **origin** tag: `notebook` (the notebook as handed out) or `study` (the student's own
  research).
- **dataset** tag, on every run: the dataset the run was trained and scored on.
- **role** tag: `baseline`, `trial` (one search trial), `final` (the result a student
  stands behind, rerun over seeds), `ablation`, `hypothesis`.
- **a study** carries, on each of its runs, the tags `study` (its name), `claim` (what the
  student set out to show), `search_space` (the space searched), `space_is_default`
  (`true` when that is the notebook's space, unchanged) and `budget` (the number of
  trials), and the artifact `search_space.json` with the space in full.
- **scores**: per-seed values become `<metric>` (the mean), `<metric>_std`,
  `<metric>_per_seed` and `n_seeds`. `submit_runs.py --list` shows them as
  `accuracy=<mean> ± <std> (n=<seeds>)`.
- **curves**: per-epoch values, logged with step = epoch.
- **model** + **example**: `n_params`, the one-line `architecture` tag
  (`GaussianNoise > Linear(384->128) > GELU > Linear(128->47)`) and `architecture.json`,
  traced by torchinfo through one example batch.
- Runs logged while notebooktester runs get `test_run=true` and are not submitted.

`runlog.py` sits in the lesson folder, as the course's other per-lesson helpers do. To
bring a lesson into the protocol, copy it into that folder, log the exercise's runs
through `log_run`, and end the exercise notebook with the submission section (lesson 1's
exercise notebook is the example).

## Setting up the runs repository (once per cohort)

1. Create a public repository per cohort, for example `raoulg/mads-ml-runs-2026-11`, with
   a `main` branch. Copy in `docs/runs-repo-template/` (README and
   `.github/workflows/validate.yml`); set `TEACHERS` in the workflow to the logins allowed
   to write anywhere.
2. In the repository's settings:
   - **Actions → General → Workflow permissions**: *Read and write permissions*, so the
     workflow's token can merge. The workflow itself declares, per job, the least it
     needs.
   - **Rules → Rulesets**: a ruleset on `main` that **blocks force pushes and
     deletions**, so history stays intact.
   - **Settings → Variables → Actions**: the variable `AUTO_MERGE`. Unset or anything but
     `false`, a pull request that passes the check is merged by the workflow; `false`
     leaves the passing ones for you to merge.
3. Add no collaborators: students submit from their own forks, and their GitHub login (the
   author of the pull request) is their identity in the protocol.
4. Put the repository URL in the course announcement. `docs/submit-runs.md` tells the
   student's agent to fork it and clone the fork to `../mads-runs`, next to the course
   repository.
5. Clone the class repository itself to `../mads-runs` (or set `MADS_RUNS_REPO`); that
   clone is what `collect_runs.py` pulls.

The workflow, `validate.yml`, runs on every pull request against `main` and on every push
to `main`:

- **A pull request** runs under `pull_request_target`: the workflow as it is on `main`,
  with a token that can write. Nothing from the pull request runs as code. The workflow
  lists the changed paths through the API, fetches the pull request's head into a
  separate directory without credentials, lays its submission files over a checkout of
  `main`, and runs the validator with `--actor <author of the pull request>`. Every
  changed path must be a valid `lesson<N>/<author>/<run_id>.json`, and no run id may sit
  in another student's directory. A failure is posted as one comment on the pull request
  and fails the check; a pass is squash-merged, unless `AUTO_MERGE` is `false`.
- **A push to `main`** (your own) is validated as a set of changed files under
  `lesson*`, with `github.actor` as the actor.

The validator, `scripts/run_submission.py`, is taken from the course repository's
default branch, so that file must be on `master`.

## Opening a lesson

Set a deadline before the lesson (the evening before works). Then:

```bash
uv run scripts/collect_runs.py --lesson 1          # pull and import; prints runs per student and per experiment
uv run scripts/class_analysis.py --lesson 1        # tables in the terminal, figures in artefacts/class-analysis/lesson1/
uv run scripts/collect_runs.py --lesson 1 --ui     # import again (a no-op) and open mlflow ui on the class store
```

`class_analysis.py` also runs cell by cell (`# %%`) in VS Code or Jupyter, which is the
easier way to walk through it in front of the class. It shows:

- **who submitted what**: runs per student and role;
- **baseline against final, per student**: the gain, the gain in pooled seed sigmas, and
  how much of the best single-seed trial survived the rerun over seeds;
- **score against parameters** for every run, with the baselines' two-sigma band;
- **which architectures appear**: each distinct `architecture` signature, how many
  students built it, best and median score;
- **which choices the searches explored**: per search axis, how often each value was
  tried and what it scored.

In mlflow ui itself, useful views on the class store:

- Filter `tags.role = "final"` and show columns `tags.student`, `metrics.accuracy`,
  `metrics.accuracy_std`, `metrics.n_params`, `tags.architecture`.
- Chart view, scatter `n_params` against `accuracy` over all trials.
- Select one student's trials and compare their `accuracy` curves per epoch.
- Every run's Artifacts tab has its `architecture.json` with the shape at each layer.

## Why not mlflow-export-import, or a tracking server

`mlflow-export-import` is the standard tool for copying runs between tracking servers. It
does not fit this job. `import-run` always creates a new run, so importing twice
duplicates everything, and it would need the same index this protocol keeps anyway. It
exports every artifact, which would put model weights in git. Its `run.json` carries the
exporter's OS username, the local tracking URI and all `mlflow.*` source tags with their
absolute paths. It has no field for the student or the lesson and no validator for CI.
Its last PyPI release is from 2023, and it depends on `mlflow-skinny[databricks]`, which
this course does not otherwise need. mlflow itself has no run export and import beyond
`mlflow experiments csv`, a summary table without metric history or artifacts.

The format here keeps what that tool's `run.json` keeps (info, params, metric history,
tags), drops what is private, and adds identity and validation. Import uses mlflow's own
client API (`create_run`, `log_batch`, `log_dict`), and it is only a few lines.

A hosted tracking server would make students write to one database live. It needs
hosting, credentials for every student, and a network during the exercise. The git
transport needs none of those, and it leaves a history of who submitted what and when.
