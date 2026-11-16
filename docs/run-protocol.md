# The class run protocol

Students submit their training runs between lessons. Each lesson opens with the whole
class's runs in front of us: which architectures were built, what they scored, and how much
of each search winner survived being rerun a few times. There is no server. A GitHub
repository carries the runs; for the November 2026 cohort that is
https://github.com/raoulg/mads-ml-runs-2026-11 (public).

## How several people write to one store without a server

Optuna keeps its studies in a database, and a database that several people write to live
needs one process that receives every write. This protocol has no shared database, so it
needs no such process:

1. **Every student keeps their own local Optuna store** (`artefacts/optuna/studies.db` in
   their checkout of this repository). The notebooks and exercise scripts of every lesson
   log to it through `hpo.optunalog`, offline. A run is one trial of a study.
2. **A submission is a set of files in a pull request.** Each student forks the runs
   repository once. `scripts/submit_runs.py` writes each trial as one JSON file into the
   student's clone of that fork, at `lesson<N>/<github-login>/<run_id>.json`, commits on
   the branch `lesson<N>`, pushes it to the fork, and opens a pull request against the
   class repository. Every student writes only into their own directory, so two pull
   requests never touch the same file and merge in any order.
3. **A check merges what is valid.** The runs repository's workflow validates every pull
   request and merges it when its files are valid runs in its author's own directory.
   Students need no access to the class repository, and the teacher merges nothing by
   hand.
4. **Collecting is for student and teacher alike, each into their own class store.**
   `scripts/collect_runs.py` fetches the class repository's `main` and rebuilds a separate
   local store from every file on it (`artefacts/optuna/class.db`), the only thing that
   writes there. It reads the files from the git ref `<remote>/main` (`upstream` in a
   student's clone of their fork, `origin` in the teacher's clone of the class
   repository), never from the working tree, so it works on any branch and leaves the
   clone as it was. The store is rebuilt on every run, so it holds exactly what `main`
   holds; run it as often as you like.

Git and GitHub do the work a tracking server would do: they receive writes
asynchronously, keep them in order, and authenticate who wrote what (the author of the
pull request). optuna-dashboard only ever reads a local sqlite file.

```
 student laptop                  GitHub                                 your laptop
 notebook --log_run--> studies.db
 submit_runs.py --push--> alice/mads-ml-runs (fork), branch lesson1
                            |  pull request, checked and merged by validate.yml
                            v
                          raoulg/mads-ml-runs: lesson1/alice/<run>.json --fetch--> collect_runs.py
                                               lesson1/bob/<run>.json             --> class.db
                                                                                  --> optuna-dashboard
                                                                                  --> class_analysis.py
```

## The pieces

| File | Who | What |
|---|---|---|
| `hpo/optunalog.py` | notebooks, scripts | `search_study` + `draw` + `record` log a search, `use_study` + `log_run` log a run that is given its params: a run in the protocol's shape |
| `hpo/metrics.py` | notebooks, scripts | every metric of a classification run, from one confusion matrix over the whole test set |
| `hpo/designs.py`, `hpo/exercise.py` | exercise scripts | `@option` and the space check; the study of an exercise (`run_study`), given a lesson's harness |
| `scripts/run_submission.py` | both, and CI | the file format (pydantic) and the validator |
| `scripts/submit_runs.py` | student | local Optuna store → JSON files → commit on `lesson<N>` → push to the fork → pull request |
| `docs/submit-runs.md` | student's coding agent | step-by-step instructions, including the one-time setup |
| `scripts/collect_runs.py` | both | fetch `main` → validate → rebuild `class.db` → optionally optuna-dashboard |
| `scripts/class_analysis.py` | both | the class page: the class's runs of one lesson in pictures, as one HTML file |
| `Makefile` | both | `make share LESSON=<N>`, `make dashboard LESSON=<N>`, `make optuna-dashboard` |
| `docs/runs-repo-template/` | teacher, once | README and the workflow that validates and merges pull requests |

`make share LESSON=<N>` is `submit_runs.py --lesson <N>` (`ARGS=--list` shows what would be
submitted). `make dashboard LESSON=<N>` writes and opens the class page, then rebuilds
`class.db` and opens optuna-dashboard on it. `make optuna-dashboard` opens optuna-dashboard
on your own store.

## What a submission holds

One file per trial, schema `mads-run/2` (`uv run python scripts/run_submission.py schema`
prints the JSON schema):

- **lesson**, **student** (GitHub login), **submitted_at**.
- **code**: the git commit of the student's course checkout when the run was submitted,
  whether it had uncommitted changes, and its remote.
- **run**:
  - `run_id` and `run_name`: the id the trial got when it was recorded (32 hex
    characters), and its name;
  - `experiment`: the name of the study the trial belongs to;
  - `status`: `COMPLETE`, `PRUNED` (a pruner stopped it early) or `FAIL`; a `COMPLETE`
    trial has a value;
  - `start_time` and `end_time`;
  - `params`, and per param its `distributions`: what it was drawn from (a float, an
    integer or a categorical distribution, with its range or choices). A param that was
    given rather than drawn has none;
  - `value`: the trial's final value, the accuracy; and `intermediate`: the value after
    every epoch of a search trial, or after every repeat of a run that was rerun a few
    times (step = repeat);
  - `attrs`: the trial's user attrs (below);
  - `study`: the study's direction and its own attrs, among them the search space;
  - `artifacts`: the small text artifacts, in practice `architecture.json` (the layer
    table with input and output shapes) and `design.py` (the student's design script).

Artifacts other than `.json`/`.txt`/`.md`/`.yaml`/`.py` files under 256 KB are never
submitted, so weights cannot end up in git. A `.py` artifact is read, never run: CI only
validates the JSON around it. A file is at most 2 MB, and one param or attr at most 20 KB.

A trial imported from the class repository is never submitted again: `submit_runs.py`
leaves out every trial that carries `submission_ref` or `source_run_id`, the attrs
`collect_runs.py` writes, with one line saying how many it left out. That keeps a
classmate's run out of a submission even when the student's store and the class store
are mixed up. A trial without a `run_id` was not recorded through `hpo.optunalog` and is left
out as well.

## Logging conventions

The class page reads what `hpo.optunalog` writes. A notebook that follows the protocol logs
every run worth discussing through it, for example a run that is given its params:

```python
from hpo.optunalog import log_run, use_study

use_study("lesson1-notebook-runs")
log_run("winner", "final", params=best, metrics=per_repeat_metrics,
        model=model, example=one_batch[:2])
```

where `per_repeat_metrics` holds one dict per repeat as `hpo.metrics.scores` gives it
(`hpo.exercise.train` returns it as `fit.metrics`).

and a search as a study of its own:

```python
from hpo.optunalog import Int, draw, record, search_study

space = {"hidden_dim": Int(32, 1024, log=True), ...}
study = search_study("lesson1-notebook", space)  # objective="accuracy" by default

def objective(trial):
    config = draw(trial, space)
    ...  # train; trial.report(metrics["accuracy"], epoch) after every epoch
    record(trial, "trial", dataset="dtd", model=model, example=one_batch[:2],
           curves=fit.curves, metrics=fit.metrics)
    return metrics["accuracy"]
```

- **Every metric**: a run logs every metric of `hpo.metrics`, whatever its study
  searches, so a question asked later is answered by runs already shared. The user attr
  `metrics` holds `accuracy`, `macro_recall`, and `recall`, `precision` and `f1` per
  class under the class names (`{"recall": {"N": 0.98, "F": 0.31, ...}}`): the value
  after the last epoch, or the mean over the repeats of a run that was repeated.
  `metrics_std` has the same shape and holds the spread over the repeats. The scalar
  metrics are also logged per epoch, as the lists `accuracy` and `macro_recall` next to
  `train_loss` and `test_loss`. Every metric is computed over the whole test set at once:
  macro recall, precision and F1 averaged over batches are other numbers. A class
  without rows in the test set has no recall and no F1, and is left out of the mean of
  `macro_recall` rather than counted as 0.
- **The objective**: a study names the one metric that is its trials' value and that
  its search maximises, `search_study(..., objective="macro_recall")` or `use_study(...,
  objective=...)`, written as the study attr `objective`; `accuracy` is the default. A
  study refuses trials on another objective, as it refuses another idea or another
  space. `<metric>_std` holds the spread of every scalar metric, next to `n_repeats`. The
  validator checks that `metrics` and `metrics_std` hold known metrics, numbers between 0
  and 1 (a spread at least 0) and class names of letters, digits and `_ - .`, that the
  objective is a scalar metric, and that a finished trial's value is its objective in
  `metrics`. A run without `metrics` logs its value only, and a study without
  `objective` is on accuracy: lesson 1's runs from before stay valid.

- **study names**: `lesson<N>-notebook` (the trials of the handed-out search) and
  `lesson<N>-notebook-runs` (the handed-out runs that are given their params: baseline,
  encoder comparison, winner rerun a few times, ablation) for the notebook as handed out;
  `lesson<N>-study-<name>` and `lesson<N>-study-<name>-runs` for the student's own study,
  `<name>` being the first name, a dash, and what it studies (`sam-depth-on-eurosat`). The
  lesson's other notebooks log to studies of their own, in lesson 1
  `lesson1-representation-quality` (notebook 01) and `lesson1-search` with
  `lesson1-search-runs` (notebook 04). The submit script takes the studies
  `lesson<N>-notebook*` and `lesson<N>-study*` by default; any other study only when named
  with `--experiment`.
- **origin**: whether a run is the notebook as handed out (`notebook`), the student's own
  study (`study`) or neither (`other`). It is derived from the name of the study, not
  logged: `lesson<N>-notebook` and `lesson<N>-notebook-runs` are `notebook`, the studies
  that start with `lesson<N>-study` are `study`.
- **role** attr: `baseline`, `trial` (one search trial), `final` (the result a student
  stands behind, rerun a few times), `ablation`, `hypothesis`.
- **dataset** attr, on every run: the dataset the run was trained and scored on. Scores of
  different datasets are never compared with each other.
- **a student's own study** carries, as study attrs, `origin`, `study` (its name),
  `space_is_default` (`true` when the space is the notebook's, unchanged) and `budget`
  (the minutes given to it, the seconds one trial costs on that machine and the study
  they were measured on, the trials that fit, epochs and repeats). A study run from
  `exercises/study.py` also carries `expects`: what the student expected and why, in one
  line, which the check never reads.
- **a search** (a study made by `search_study`) carries its space as study attrs:
  - `search_space`: the whole space on one line, for example
    `encoder=dinov2-small | hidden_dim ~ logint(32, 1024) | lr ~ logU(1e-4, 1e-2)`;
  - `searched_axes`: the names of the axes that are not fixed, sorted and
    comma-separated.

  Every trial of it draws its params from the space (`draw`), reports its accuracy after
  every epoch, and carries the whole config as the attr `config`, fixed values included.
- **params**: every choice that tells runs apart is a param, with one name and one kind of
  value across the lesson: `encoder`, `embed_dim`, `dataset`, `hidden_dim`,
  `n_hidden_layers`, `activation`, `norm`, `dropout`, `skip`, `gate`, `sigma`, `lr`. A
  switch that has no effect on a run is logged at its off value (`hidden_dim=0` on a linear
  head). A param is a number, a boolean, or a string from the list `CHOICES` holds for its
  name; any other string is written as an attr.
- **the spread between runs**: `log_run` takes `scores={"accuracy": [...]}`, one value per
  repeat. The mean is the trial's value, every repeat an intermediate value, and the
  attrs `accuracy_std` and `n_repeats` hold the spread between the repeats. Nothing is
  seeded: every repeat starts from other random weights, and the spread is what a
  difference has to clear before it means anything. `submit_runs.py --list` shows it as
  `accuracy=<mean> ± <std> (n=<repeats>)`.
- **curves**: the accuracy per epoch is the trial's intermediate values; the loss curves
  are the attrs `train_loss` and `test_loss`, one list per epoch.
- **model** + **example**: the attrs `n_params` and the one-line `architecture`
  (`GaussianNoise > Linear(384->128) > GELU > Linear(128->47)`), and the artifact
  `architecture.json`, traced by torchinfo through one example batch. The layer table is
  also the trial's note, which optuna-dashboard renders.
- **cost**: the attrs `train_seconds` (training only, the mean over repeats) and `n_params`.
- **session** attr: one value per execution of a notebook, so a rerun can be told apart
  from the first run. The class page keeps only a student's latest session of the
  handed-out runs.
- **run_id** attr: made once per trial; it is the name of the submission file.
- Trials made while notebooktester runs go to the separate store `artefacts/optuna/test.db`
  and are never submitted.
- **design**: `log_run(..., design="my_design.py")` (or `record(...)`) stores the text of
  the student's design script, the file that defines their architectures, as the
  artifact `design.py`.
- **builds_on** attr: the classmates' runs a run builds on, `log_run(...,
  builds_on="lesson1/bob/<run_id>")` (one or a list), written comma-separated. Each entry
  is the `submission_ref` of a run in the class store; `hpo.optunalog` and the validator both
  check that every entry has that form, and the validator that none names the run itself,
  not that the run exists (it may sit in a pull request not merged yet).
- **The idea of a study**: `idea("does depth still help on eurosat", builds_on=[...])`
  in `hpo.optunalog` gives the study attrs `idea` (the words), `idea_author` (the student's
  GitHub login, or `course` for a study a lesson hands out), `idea_parents` (the names of
  the ideas it builds on, comma-separated), `idea_id` and `idea_name`; `search_study` and
  `use_study` take it as `idea=`. The id is the sha256 of the author, the words and the
  parents' names, so an id holds its parents as a git commit's hash holds its parents';
  the name (`blissful-bee-18`) is the first three bytes of the id read as an adjective, an
  animal and a number. An idea with two parents is a merge: the class's ideas are a
  graph that branches and joins, and cannot hold a cycle. The design script, the search
  space and the scores are not in the id: other words are another idea, and a study
  holds one idea. The validator recomputes id and name from the file, checks that the
  author is the submitter (or `course`, for an experiment named `lesson<N>-notebook*`),
  that every parent is the name of an idea in the class repository or in the same pull
  request, and that a name stands for one id.
- **submission_ref** attr, written by `collect_runs.py` on every imported trial:
  `lesson<N>/<login>/<run_id>`, the path of its file without `.json`. It is what a
  student copies from optuna-dashboard to build on that run. The import also writes
  `student`, `lesson`, `source_run_id`, `source_run_name`, `submission_hash`,
  `submitted_at` and `code_*`.

### The class store

`collect_runs.py` builds `class.db` from the files on `main`, with every trial's attrs,
note and artifacts restored. The handed-out studies (`lesson<N>-notebook` and
`lesson<N>-notebook-runs`) are pooled: one study with every student's trials. Everyone ran
the same search space, so the class's trials together are one search of a few hundred
points. Every other study is kept per student as `<study>@<student>`, because each student
searched another space; a trial that does not fit the pooled study (a param of another
kind) goes there as well. Invalid files are reported and skipped.

### Reading a search in optuna-dashboard

`make optuna-dashboard` opens your own store at `http://127.0.0.1:8088`; `make dashboard
LESSON=<N>` opens the class store at `http://127.0.0.1:8089`. optuna-dashboard shows one
study at a time. Its pages are in the bar on the left:

- **History**: the accuracy of every trial in the order they ran, with the best so far;
  **Hyperparameter Importance**, which axis explains most of the differences in accuracy
  (with a few dozen trials a hint, to check against the spread between runs); and
  **Intermediate values**, every trial's accuracy per epoch, one line per trial. In a
  search the pruned trials stop early; in a `-runs` study the steps are the repeats of one
  run, so the height of a line's wiggle is its spread between runs.
- **Analytics**: accuracy against one param (Slice) and against two (Contour, Rank).
- **Smart Selection**: the parallel-coordinates plot. The params are the axes, and the
  numeric attrs `n_params` and `train_seconds` can be ticked as axes too, so the plot shows
  where more width or depth stops paying in accuracy and starts costing in time. Drag along
  an axis to keep the trials in that range; the history plot and the table below follow.
- **Trials (List)**: one page per trial, with its note (the layer table and the tensor
  shape at every layer), the accuracy per epoch, the params, the attrs (`role`, `dataset`,
  `n_params`, `architecture`, `train_seconds`, `run_id`, and in the class store
  `submission_ref` and `student`) and the artifacts `architecture.json` and `design.py`.
  The study's own page shows its attrs: its idea (`idea`, `idea_name`, `idea_author`,
  `idea_parents`, `idea_id`), `search_space`, `searched_axes`, and for a student's study
  `study`, `budget` and `space_is_default`.

Questions it answers at a glance: where more width or depth stops paying in accuracy and
starts costing in time; what a switch (`norm`, `skip`, `gate`) buys at equal width; how far
another encoder moves the result compared with everything in the head. Read a gap against
`<objective>_std` before calling it a result. optuna-dashboard shows the value, the
study's objective; the class page reads any logged metric (`make dashboard LESSON=<N>
METRIC=recall/F`), and names it on every figure.

`hpo` is the course's own package, installed by `uv sync`, so every lesson imports it the
same way. To bring a lesson into the protocol, log the exercise's runs through `log_run`
and `record` with every metric (`hpo.exercise.train` gives them), name the study's
objective, and end the exercise notebook with the submission section (lesson 1's
exercise notebook is the example). A lesson's exercise script is a `Harness` in its
`exercises/harness.py` and a call of `hpo.exercise.run_study` (lesson 1's
`exercises/` is the example).

### Ideas across lessons

An idea builds on any published idea, of its own lesson or an earlier one: the check
looks a parent up in the whole class repository. The class page of a lesson draws a
parent of another lesson as a dashed box that names its lesson.

## Setting up the runs repository (once per cohort)

1. Create a public repository per cohort, for example `raoulg/mads-ml-runs-2026-11`, with
   a `main` branch. Copy in `docs/runs-repo-template/` (README and
   `.github/workflows/validate.yml`). The logins allowed to write in any directory are
   set in one place: `TEACHERS` in the workflow's top-level `env`.
2. In the repository's settings:
   - **Actions → General → Workflow permissions**: leave the default, read-only. The
     workflow declares per job the permissions it needs, and only the merging job can
     write.
   - **General → Pull Requests**: keep *Allow squash merging* on; the workflow merges
     with a squash.
   - **Rules → Rulesets**: a ruleset on `main` that **blocks force pushes and
     deletions**, so history stays intact.
   - **Settings → Secrets and variables → Actions → Variables**: the variable
     `AUTO_MERGE`. Unset or anything but `false`, a pull request that passes the check is
     merged by the workflow; `false` leaves the passing ones for you to merge.
3. Add no collaborators: students submit from their own forks, and their GitHub login (the
   author of the pull request) is their identity in the protocol.
4. Put the repository URL in the course announcement. `docs/submit-runs.md` tells the
   student's agent to fork it and clone the fork to `../mads-runs`, next to the course
   repository.
5. Clone the class repository itself to `../mads-runs` (or set `MADS_RUNS_REPO`);
   `collect_runs.py` fetches its `origin/main` and reads the runs from there.

The workflow, `validate.yml`, runs on every pull request against `main` and on every push
to `main`:

- **A pull request** runs under `pull_request_target`: the workflow as it is on `main`,
  with a token that can write. Nothing from the pull request is checked out or run. The
  workflow fetches the pull request's commit into a checkout of `main` without
  credentials, lists the changed paths with `git diff` from the merge base, writes the
  submission files over that checkout with `git cat-file`, and runs the validator on
  them with `--actor <author of the pull request>`. Every changed path must be a valid
  `lesson<N>/<author>/<run_id>.json`, a regular file, with the author spelled exactly as
  their login; no run, by id or by content, may sit in another student's directory; and
  a pull request changes at most 200 files. A failure is posted as one comment on the
  pull request and fails the check; a pass is squash-merged, unless `AUTO_MERGE` is
  `false`.
- **A push to `main`** (your own) is validated as a set of changed files under
  `lesson*`, with `github.actor` as the actor.

The validator, `scripts/run_submission.py`, is taken from the course repository's
default branch, so that file must be on `master`. The workflow's actions are pinned to
commit SHAs, with the release tag in a comment.

## Opening a lesson

Set a deadline before the lesson. Then:

```bash
uv run scripts/collect_runs.py --lesson 1          # fetch main, rebuild class.db; prints runs per student
uv run scripts/class_analysis.py --lesson 1 --open # write the class page, artefacts/class-analysis/lesson1/index.html, and open it
make dashboard LESSON=1                            # both of the above, then optuna-dashboard on the class store
```

The class page is one self-contained HTML file, read from top to bottom in the lesson. A
row of tiles counts the students, the runs, the own studies, the trials of the handed-out
search and the runs with a design script. Scores of different datasets are never compared
with each other: every picture that compares scores does so per dataset. The pictures:

- **Who submitted what**: runs per student and role. A student with only baselines ran the
  handed-out part and no study of their own.
- **The handed-out baselines, side by side**: the same model on the same data, on every
  student's machine. Each dot is one student's baseline, the bar the spread between their
  own runs. How far the dots lie apart is how large a difference has to be to mean
  anything.
- **Baseline against final, per study**: each student's own study, the gain of the final
  over the baseline against the spread. Grey: a gain inside two sigma. Green: a gain that
  clears it.
- **The handed-out search, pooled over the class**: every trial of the handed-out search
  as one line in a parallel-coordinates plot, the class's trials together as one random
  search many times the size.
- **Axis by axis**: accuracy against each searched axis, every trial as a faint dot, the
  median per bin or value as a line, and the class's median baseline dotted.
- **Each student's own study**: the trials in the order they ran (crosses: stopped early by
  the pruner), the baseline's band of two standard errors in grey, the final, rerun a few
  times, as a star, and under it every study's search space on one line.
- **The ideas of the class**: one box per idea, an arrow from every idea to the one that
  builds on it.
- **Who built on whom**: per `builds_on` link, the student's run against the classmate's run
  it builds on, both scores and the difference; a link to a run not in the class store is
  listed as such.
- **What a larger head costs and buys**: every scored run of the most common dataset,
  accuracy against parameters.

`class_analysis.py` reads the files from `main` exactly as `collect_runs.py` does, and
`--demo` writes the page on made-up data of eight invented students. It shows what
optuna-dashboard cannot: the class pooled, and a student, a role or a dataset as the thing
to filter on. optuna-dashboard on the class store shows the single study: open
`lesson1-notebook` for the pooled handed-out search, and read it as above. In the class
store:

- A student's own study is the study `<study>@<login>`; its `-runs` study holds their
  baseline, final and arms.
- The trial page of any run has its `architecture.json` with the shape at each layer, and
  its `design.py` when the student logged one.
- The study attr `searched_axes` says which axes a search varied; `search_space` says over
  what.

The lesson discusses these runs live, with the students who made them: why did you do
this, what was the idea, what did you learn when it disappointed.

## What the check does not promise

- **A copy is caught by a heuristic.** A run relabelled as another student's is found by
  the hash of its content; a copy with one number changed is not. The check keeps honest
  mistakes out, and the lesson does the rest: a run is discussed by the student who made
  it.
- **Two pull requests opened in the same minute are checked apart.** Each is validated
  against `main` as it was when it was opened, so the same run in both merges twice.
  `collect_runs.py` imports both; nothing else breaks.
- **A file the check accepts can still be one Optuna refuses.** `collect_runs.py` skips
  such a run with one line and imports the rest.
- **There is no cap on what one student submits over time**, beyond 200 files and 2 MB
  per file in one pull request.

## Why not a tracking server

A hosted tracking server would make students write to one database live. It needs
hosting, credentials for every student, and a network during the exercise. The git
transport needs none of those, and it leaves a history of who submitted what and when.

The format here is what a trial holds, plus identity and validation: the student and the
lesson, a validator for CI, and a limit on what an artifact may be, so weights cannot end
up in git.
