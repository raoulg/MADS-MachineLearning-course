# MADS-MachineLearning-course

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

## planning/ is the source of truth

`planning/main_principles.md` holds the principles, the numbered decisions (D1, D2, …),
the measured evidence they rest on, and the `cuts` register; `planning/lesson<N>.yml`
holds the per-lesson detail. Read the relevant plan before changing a lesson.

Two standing rules from that directory:

- **Reality beats description; intent beats artifact.** When the plan *describes* the
  code and the cells disagree, the cells are right — fix the plan. When the plan
  *prescribes* a change the cells have not caught up with, the plan is right — change the
  cells. A refactor plan is meant to contradict the notebooks. Facts (shapes, row counts,
  accuracies) are settled by the code; decisions are settled by the plan, until the plan
  is explicitly revised.
- **Verify before asserting.** Several plan items were wrong until tested — check the
  dataset, the shape, the licence, rather than assuming.

## Libraries

`vectormesh` (components, vectorizers, caches) and `mltrainer` (Trainer, metrics) are
sibling repos at `../vectormesh` and `../mltrainer`, published to PyPI. The course
installs the **published** versions — do not rely on an editable install or a
`PYTHONPATH` shim, because students will not have one. If a notebook needs a library
change, that change ships as a release first.
