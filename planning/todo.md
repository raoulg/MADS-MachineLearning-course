# Open work

Living checklist. Delete an item when it is done, not when it is planned.

## Audit: unjustified epoch counts, course-wide

`principles.md` §3 now requires that any epoch count behind a *reported* number is
justified by a plateau or a convergence curve. That rule was written after the second time
an assumed count changed the sign of a conclusion (`evidence.md` §7 and §11). Both were
caught by the instructor asking, which means the process did not catch them, which means
**there are very likely more.**

**The question is whether a different budget would change a conclusion** — not whether the
run stopped exactly at the plateau. A baseline that is a quarter of a point below its
asymptote is not a finding; it changes no ranking and no verdict. What counts is a budget
that turns a real effect into "no effect", reverses which model comes first, or lets a
notebook claim something does not work when at a longer budget it does.

Check each notebook against *that*, and do not manufacture findings. Most of these are
probably fine.

| lesson | notebook | trains and reports? | checked |
|---|---|---|---|
| L1 | `02_nonlinearity` | yes | ☐ |
| L1 | `03_training` | yes — the baseline everything else is measured against | ✔ fine at 20; still climbing by ~0.3 pt to epoch 40, which changes nothing |
| L1 | `04_search` | search (equal budget, acceptable) + a final arm | ☐ |
| L1 | `05_exercises` | yes, incl. the data-size ablation | ☐ |
| L2 | `02_basemodels` | yes — the model comparison table | ☐ |
| L2 | `03_decomposition_and_nbeats` | yes — has an epoch sweep already (§3.4), so probably fine | ☐ |
| L3 | `01_gates_highways_moe` | yes — **has** a 5-seed × 3-epoch-count check, the model to copy | ✔ |
| L3 | `04_search` | search budget + a final comparison | ☐ |
| L3 | `02_scaling_curves` | yes — the epoch curve *is* the notebook's first cell | ✔ |
| L3 | `03_what_does_the_algorithm_know` | search at equal budget, winner retrained at a justified 20 | ✔ |
| L4 | `03_long_documents` | yes, and the search | ☐ |
| L4 | `04_composition` | yes — the fusion ablation table | ☐ |
| L5 | `01_why_pretext_tasks_work` | yes — sweep done, EPOCHS=40 justified | ✔ |
| L5 | `02_selfsupervised` | yes — SSL, where a flat loss hides collapse | ☐ |
| L5 | `03_patch_tokens` | yes | ☐ |
| L5 | `04_ijepa` | yes | ☐ |
| L6 | `01_probabilistic` | yes — CRPS/calibration tables | ☐ |
| L7 | `01_foundation_models` | mostly zero-shot, but the trained baselines count | ☐ |

Priority: L4 `04_composition`, because a fusion ablation is the shape of comparison an
under-trained run distorts most, then L5 `02_selfsupervised`, where a flat loss can hide
collapse and the budget genuinely decides whether the model learned anything.

L1 `03_training` is checked and fine: it is ~0.3 pt below its asymptote at 20 epochs, which
is real and irrelevant — no ranking moves. Left at 20.

Where a notebook genuinely fails — where the budget changed a conclusion — put the
convergence table in the notebook as the justification and record the number in
`evidence.md`. Where it passes, note it here and move on; do not add a paragraph to the
notebook about having checked.

**Consider making this cheap rather than manual.** If `vectormesh.training.Trainer` grew a
plateau-detecting early stop that reports *why* it stopped, and notebooks used it for final
models, the rule would be enforced by construction instead of by discipline. That is a
library change and ships as a release first.

## Dependencies to drop

- ~~`mads-datasets`~~ — **gone.** Its last two consumers were L3's `01_hypertuner`
  (deleted) and `hypertune.py` (rewritten against a published cache)
- ~~`bayesian-optimization`~~, ~~`configspace`~~ — **gone** with the algorithm-zoo notebook
- `torch-geometric` — still to move to an optional group with `8_extras/graphs`

## Research to verify

`research/README.md` holds the rule: nothing moves to `evidence.md` without an independent
re-run. Done: the RankMe inversion (§10). Outstanding: everything in
`memory_beyond_rag.md` and `neurosymbolic_and_structure.md`, both of which L7 is built on.
