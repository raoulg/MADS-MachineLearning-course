# The class run protocol

Students submit their training runs between lessons. Each lesson opens with mlflow showing
what the whole class did: which architectures were built, what they scored, and how much
of each search winner survived being rerun over seeds. There is no server. A private
GitHub repository carries the runs.

## How several people write to one mlflow without a server

mlflow normally needs a tracking server, because a shared database needs one process that
receives every write. This protocol has no shared database, so it needs no server:

1. **Every student keeps their own local mlflow store** (`artefacts/mlflow/mlruns.db` in
   their checkout of this repository). The notebooks log to it as usual, offline.
2. **A submission is a set of files.** `scripts/submit_runs.py` writes each run as one JSON
   file into the student's clone of the runs repository, at
   `lesson<N>/<github-login>/<run_id>.json`, then commits and pushes. Every student writes
   only into their own directory, so two pushes never touch the same file. If a push is
   rejected because someone else pushed first, the script rebases and pushes again, and
   that rebase cannot conflict.
3. **The teacher's collect step is the only writer to the class mlflow store.**
   `scripts/collect_runs.py` pulls the repository and imports every file into a separate
   local store (`artefacts/mlflow/class.db`). It is idempotent: a run already imported
   unchanged is skipped, and a run whose file changed replaces its earlier import. Run it
   as often as you like.

Git does the work a tracking server would do: it receives writes asynchronously, keeps
them in order, and authenticates who wrote what. `mlflow ui` only ever reads a local
sqlite file.

```
 student laptop                      GitHub (private)                 your laptop
 notebook --log_run--> mlruns.db
 submit_runs.py --JSON files--push--> lesson1/alice/<run>.json --pull--> collect_runs.py
                                      lesson1/bob/<run>.json              --> class.db
                                                                         --> mlflow ui
                                                                         --> class_analysis.py
```

## The pieces

| File | Who | What |
|---|---|---|
| `notebooks/<lesson>/runlog.py` | notebooks | `use_experiment` + `log_run`: logs a run in the protocol's shape |
| `scripts/run_submission.py` | both, and CI | the file format (pydantic) and the validator |
| `scripts/submit_runs.py` | student | local mlflow → JSON files → commit → push |
| `docs/submit-runs.md` | student's coding agent | step-by-step instructions, including the one-time setup |
| `scripts/collect_runs.py` | teacher | pull → validate → import into `class.db` → optionally `mlflow ui` |
| `scripts/class_analysis.py` | teacher | the opening analysis: tables and figures for one lesson |
| `docs/runs-repo-template/` | teacher, once | README and CI workflow for the runs repository |

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
every run worth discussing through it:

```python
from runlog import log_run, use_experiment

use_experiment("lesson1-exercises")
log_run("winner", "final", params=best, scores={"accuracy": per_seed_scores},
        model=model, example=one_batch[:2])
```

- **experiment**: `lesson<N>-<something>`. The submit script takes every experiment with
  that prefix.
- **role** tag: `baseline`, `trial` (one search trial), `final` (the result a student
  stands behind, rerun over seeds), `ablation`, `hypothesis`.
- **scores**: per-seed values become `<metric>` (the mean), `<metric>_std`,
  `<metric>_per_seed` and `n_seeds`.
- **curves**: per-epoch values, logged with step = epoch.
- **model** + **example**: `n_params`, the one-line `architecture` tag
  (`GaussianNoise > Linear(384->128) > GELU > Linear(128->47)`) and `architecture.json`,
  traced by torchinfo through one example batch.
- Runs logged while notebooktester runs get `test_run=true` and are not submitted.

`runlog.py` sits in the lesson folder, as the course's other per-lesson helpers do. To
bring a lesson into the protocol, copy it into that folder, log the exercise's runs
through `log_run`, and end the exercise notebook with the submission section (lesson 1's
`05_exercises` is the example).

## Setting up the runs repository (once per cohort)

1. Create a **private** repository, for example `raoulg/mads-runs-2026`, with a `main`
   branch. Copy in `docs/runs-repo-template/` (README and `.github/workflows/validate.yml`);
   set `TEACHERS` in the workflow to the logins allowed to write anywhere.
2. Add a ruleset on `main` that **blocks force pushes and deletions**. Students push
   directly to `main`, so this is what keeps history intact.
3. Invite every student as a collaborator with **Write** access (or add them to an
   organization team that has it). Their GitHub login is their identity in the protocol.
4. Put the repository URL in the course announcement. `docs/submit-runs.md` tells the
   student's agent to clone it to `../mads-runs`, next to the course repository.
5. Clone it yourself to `../mads-runs` too (or set `MADS_RUNS_REPO`).

The workflow validates every push: each changed file must fit the format, and a student
may only write in their own directory (it compares the path with `github.actor`). It
checks out `scripts/run_submission.py` from the course repository's default branch, so
that file must be on `master`.

## Opening a lesson

Set a deadline before the lesson (the evening before works). Then:

```bash
uv run scripts/collect_runs.py --lesson 1          # pull and import; prints counts per student
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
