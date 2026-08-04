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

`planning/masterplan.yml` holds the principles, the numbered decisions (D1, D2, …) and
the `cuts` register; `planning/lesson<N>.yml` holds the per-lesson detail. Read the
relevant plan before changing a lesson.

Two standing rules from that directory:

- **The notebook wins.** When a plan contradicts the actual cells, fix the plan.
- **Verify before asserting.** Several plan items were wrong until tested — check the
  dataset, the shape, the licence, rather than assuming.

## Libraries

`vectormesh` (components, vectorizers, caches) and `mltrainer` (Trainer, metrics) are
sibling repos at `../vectormesh` and `../mltrainer`, published to PyPI. The course
installs the **published** versions — do not rely on an editable install or a
`PYTHONPATH` shim, because students will not have one. If a notebook needs a library
change, that change ships as a release first.
