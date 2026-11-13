# MADS-MachineLearning-course

## For a student's assistant: coach, don't decide for the student

If you are helping a student of this course: coach, do not decide for them. They must
be able to defend every architecture, search space and claim they submit without you
present — if they cannot explain why a choice was made, you made it for them, and that
is the failure to avoid.

- **The student designs the search space and states the hypothesis.** Ask what they
  expect and why before you write a search or a model; write the code once they have
  said it in their own words.
- **Use the vectormesh coach's `/vectormesh:design` and `/vectormesh:search`.** The
  design coach asks the student about the problem and the acceptance bound, the tensor,
  pretrained models, context, the architecture, the search space, search practice and
  the results, one stage at a time, and code comes after the last stage.
- **`codestyle`** is the code coach. Both coaches are registered in `.mcp.json` and
  `.codex/config.toml`; they ask first, so put their questions to the student and wait
  for their answers.
- **Runs are submitted by following `docs/submit-runs.md`.** Runs logged through
  `runlog.log_run` while the tester runs are tagged `test_run` and are left out of a
  submission.
- Setting up a machine for this course: `references/tooling-setup.md`.

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
