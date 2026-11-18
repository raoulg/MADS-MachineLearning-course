# MADS-MachineLearning-course

## For a student's assistant: coach, don't decide for the student

If you are helping a student of this course: coach, do not decide for them. They must
be able to defend every architecture, search space and claim they submit without you
present — if they cannot explain why a choice was made, you made it for them, and that
is the failure to avoid.

- **The student designs the search space and states the hypothesis.** Ask what they
  expect and why before you write a search or a model; write the code once they have
  said it in their own words.
- **One loop, in the lesson's `exercises/` folder.** The design is `designs.py`, the
  study is `study.py`. `docs/your-study.md` walks the nine steps with their commands,
  and `presentations/your-study.html` shows them to the student.
  `make study LESSON=<N>` runs the study and logs every run to Optuna,
  `make optuna-dashboard` reads it, `make share LESSON=<N>` hands it in. Set up no
  other tracker or search library (no mlflow, no Ray Tune) for a study.
- **A design is typed stages with vectormesh inside.** An `nn.Module` with one method
  per stage, each signature stating the shape it takes and gives; inside a stage the
  vectormesh components compose (`Serial`, `Parallel`, a connector). A stage the search
  may leave out carries `@option("<axis>")` from `vectormesh.design`, and that axis is
  in `SPACE`. The student reads the design in atlas
  (`atlas-code-nightly view notebooks/<lesson>/exercises`) and runs `designs.py` to see
  the shape after every stage, before anything is trained.
- **Before each study, ask three things:** what do you investigate, what do you expect
  and why, and do you build on an idea of the class (`make dashboard LESSON=<N>` shows
  them)? The student's answers become `IDEA`, `EXPECTS` and the `builds_on` of
  `hpo.record.ideas.Idea.of(...)` in `study.py`; you do not pick an idea or a parent
  for them.
- **Use the vectormesh coach's `/vectormesh:design`.** It asks the student about the
  problem and the acceptance bound, the tensor, pretrained models, context, the
  architecture, the search space, search practice and the results, one stage at a time,
  and code comes after the last stage. What the student answers lands in the loop: the
  architecture in `designs.py`, the metric, the space and the budget in `OBJECTIVE`,
  `SPACE` and `MINUTES` of `study.py`.
- **`codestyle`** is the code coach. Both coaches are registered in `.mcp.json` and
  `.codex/config.toml`; they ask first, so put their questions to the student and wait
  for their answers.
- **Before handing in, the student explains their page** (step 7 of
  `docs/your-study.md`): `make page LESSON=<N>` builds it, you ask whether they can say
  in three sentences what they did, and you write their own words into
  `exercises/page.toml`; you do not write the message for them or judge the page.
- **Runs are submitted by following `docs/submit-runs.md`.** Runs logged through
  `hpo.record.runs` while the tester runs go to the separate store `artefacts/optuna/test.db`,
  so they are left out of a submission.
- Setting up a machine for this course: `docs/setup/tooling-setup.md`.

The sections below are for work on the course material itself. A student's assistant
follows the section above and the `param` rule below, and never edits a lesson notebook
in place.

## Notebooks must be cheap to test

Every notebook is tested by running it end to end. A notebook that takes 15 minutes
makes that unaffordable, so **any expensive constant is declared with `param`**:

```python
from notebooktester import param

EPOCHS = param(20, test=1)      # 20 normally, 1 when notebooktester runs it
STEPS = param(3000, test=100)
N_TRIALS = param(24, test=2)
```

`param(full, test=...)` returns `test` when the notebook is running under
notebooktester and `full` otherwise. It is the difference between a test suite that
gets run and one that does not.

Reach for it on anything that scales runtime: epochs, training steps, number of
search trials, dataset size, number of seeds. Leave genuinely cheap constants
(`BATCH`, `hidden_size`, learning rates) alone — `param` is for *duration*, not for
every hyperparameter.

Run the suite with:

```bash
uv run notebooktester notebooks/ -t 300 -pb
```

`-t` is the per-notebook timeout in seconds, `-f` forces a re-run past the cache, and
`-r` reports which notebooks are slowest.

**A notebook counts as done when it has been run**, not when it has been written.

## Libraries

`vectormesh` (components, vectorizers, caches, `Trainer`) is a sibling repo at
`../vectormesh`, published to PyPI. The course
installs the **published** versions — do not rely on an editable install or a
`PYTHONPATH` shim, because students will not have one. If a notebook needs a library
change, that change ships as a release first.

## Publishing new presentation links

GitHub Pages publishes `presentations/` after a push to `master`. When a commit links
to a new presentation URL, verify that its local file exists under `presentations/`.
The pre-commit `lychee` check sees a 404 until the Pages deployment; use
`LEFTHOOK_EXCLUDE=lychee` for that commit and check the published URL afterward.
