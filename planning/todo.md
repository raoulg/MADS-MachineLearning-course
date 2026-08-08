# Open work

Living checklist. Delete an item when it is done, not when it is planned.

## Audit: unjustified epoch counts, course-wide

`principles.md` §3 now requires that any epoch count behind a *reported* number is
justified by a plateau or a convergence curve. That rule was written after the second time
an assumed count changed the sign of a conclusion (`evidence.md` §7 and §11). Both were
caught by the instructor asking, which means the process did not catch them, which means
**there are very likely more.**

Every notebook that trains a model and reports a number needs checking. For each: is the
epoch count justified anywhere, and does the metric actually plateau there?

| lesson | notebook | trains and reports? | checked |
|---|---|---|---|
| L1 | `02_nonlinearity` | yes | ☐ |
| L1 | `03_training` | yes — the baseline everything else is measured against | ☐ |
| L1 | `04_search` | search (equal budget, acceptable) + a final arm | ☐ |
| L1 | `05_exercises` | yes, incl. the data-size ablation | ☐ |
| L2 | `02_basemodels` | yes — the model comparison table | ☐ |
| L2 | `03_decomposition_and_nbeats` | yes — has an epoch sweep already (§3.4), so probably fine | ☐ |
| L3 | `05_gates_highways_moe` | yes — **has** a 5-seed × 3-epoch-count check, the model to copy | ✔ |
| L3 | `06_search` | search budget + a final comparison | ☐ |
| L4 | `03_long_documents` | yes, and the search | ☐ |
| L4 | `04_composition` | yes — the fusion ablation table | ☐ |
| L5 | `01_why_pretext_tasks_work` | yes — sweep done, EPOCHS=40 justified | ✔ |
| L5 | `02_selfsupervised` | yes — SSL, where a flat loss hides collapse | ☐ |
| L5 | `03_patch_tokens` | yes | ☐ |
| L5 | `04_ijepa` | yes | ☐ |
| L6 | `01_probabilistic` | yes — CRPS/calibration tables | ☐ |
| L7 | `01_foundation_models` | mostly zero-shot, but the trained baselines count | ☐ |

Priority order: L1 `03_training` first, because its baseline is the number the rest of the
course is compared against, and L4 `04_composition` second, because a fusion ablation is
exactly the shape of comparison an under-trained run distorts.

Where a notebook fails the check, the fix is the one L5's opener now models: run the sweep,
put the table in the notebook, keep the wrong reading next to the correction, and record it
in `evidence.md`.

**Consider making this cheap rather than manual.** If `vectormesh.training.Trainer` grew a
plateau-detecting early stop that reports *why* it stopped, and notebooks used it for final
models, the rule would be enforced by construction instead of by discipline. That is a
library change and ships as a release first.

## Dependencies to drop

- `mads-datasets` — last consumer is L3 `01_hypertuner`; leaves `pyproject.toml` when that
  is rewritten
- `bayesian-optimization==1.5.1` — pinned for one section of the same notebook
- `torch-geometric` — move to an optional group with `8_extras/graphs`

## Research to verify

`research/README.md` holds the rule: nothing moves to `evidence.md` without an independent
re-run. Done: the RankMe inversion (§10). Outstanding: everything in
`memory_beyond_rag.md` and `neurosymbolic_and_structure.md`, both of which L7 is built on.
