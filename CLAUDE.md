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

`planning/principles.md` holds the thesis and the standing rules, `planning/curriculum.md`
the lesson map and each lesson's target, `planning/evidence.md` every measured number the
decisions rest on, and `planning/cuts.md` what was removed and why. A `planning/lesson<N>.md`
exists only while that lesson is being rebuilt, and is deleted once it has been run — a
shipped lesson is described by its notebooks. Read the relevant file before changing a
lesson.

Two standing rules from that directory:

- **intent beats artifact.** When the plan *prescribes* a change the cells have not caught up with, 
the plan is right — change the cells. A refactor plan is meant to contradict the notebooks. 
Be aware of outdated plans that are lingering; at some point it is good to check if a plan is still
valid when a lot of commits have happened in between writing and execution.
Facts (shapes, row counts, accuracies) are settled by the code; 
decisions are settled by the plan, until the plan is explicitly revised.
- **Verify before asserting.** Several plan items were wrong until tested — check the
  dataset, the shape, the licence, rather than assuming.

## Libraries

`vectormesh` (components, vectorizers, caches) and `mltrainer` (Trainer, metrics) are
sibling repos at `../vectormesh` and `../mltrainer`, published to PyPI. The course
installs the **published** versions — do not rely on an editable install or a
`PYTHONPATH` shim, because students will not have one. If a notebook needs a library
change, that change ships as a release first.
