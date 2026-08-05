# planning/

Source of truth for the course refactor. Read the relevant file **before** changing a
lesson.

| File | Holds |
|---|---|
| `main_principles.md` | The thesis, numbered decisions (D1…), the **evidence register** of measured results, blocking prerequisites, and the cuts register |
| `lesson<N>.yml` | Per-lesson detail: notebooks, sections, exercises, what moves to legacy |

## The two standing rules

- **Reality beats description; intent beats artifact.** A disagreement about a *fact*
  (shape, row count, accuracy, import) is settled by the code. A disagreement about a
  *decision* is settled by the plan, until the plan is explicitly revised here. A refactor
  plan is supposed to contradict the notebooks — that is its job.
- **Verify before asserting.** Everything in the evidence register is a number that was
  actually produced, not recalled. Several decisions in this round reversed once measured;
  see §3.1 and §3.4 for the two that mattered most.

## Round 2/3 in one paragraph

Move students up an abstraction level. `vectormesh` makes "encode once, train many cheap
heads" affordable, so lesson 1 now starts from a published `VectorCache` instead of
building a CNN on MNIST. Lesson 2's original "produce a cache and track experiments" plan
was cut (D9/D14 in `main_principles.md`) once ray and mlflow both landed inside lesson 1
itself — redundant before it was ever built. Lesson 2 is now time series, absorbing the
round-1 `3_recurrent_networks` rebuild (folder renamed `2_timeseries`). Lessons 2 and 6
stay un-vectorized on purpose and mark the boundary of the idea.

## Status

| Lesson | Status | Round |
|---|---|---|
| L1 | **done**, built and run | 2/3 |
| L2 | specified — redo of the time-series rebuild, see `lesson2.yml` | 1 (content) / 3 (redo) |
| L3 | needs rebuild — most stale lesson in the repo | 3 |
| L4 | partially done; notebook numbering collides | 1 & 3 |
| L5 | mostly done | 1 |
| L6 | needs review | 3 |

## Building the lesson-1 caches

```bash
make build-lesson1                      # eurosat + dtd x 3 encoders -> artefacts/
make publish-lesson1 ARGS=--dry-run     # inspect the cards
make publish-lesson1                    # upload to hf.co/pttrn-io
```
