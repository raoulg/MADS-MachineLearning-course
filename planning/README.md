# planning/ — the vectormesh refactor

Read these in order. `masterplan.yml` is the source of truth; the per-lesson plans
correct it when a notebook disagrees.

| file | what it is |
|---|---|
| [masterplan.yml](masterplan.yml) | principles, decisions D1-D11, 7 phases, cuts register, risks, backlog |
| [conventions.md](conventions.md) | cache layout, naming, dataloading, tracking, shape vocabulary (P0.1) |
| [../notebooks/1_pytorch_intro/tensor-ladder.md](../notebooks/1_pytorch_intro/tensor-ladder.md) | shape vs meaning, and the four rungs (P0.2) |
| [mads_datasets_killlist.md](mads_datasets_killlist.md) | the 14 import sites, their replacements, and whether each has been **run** |
| [lesson1.yml](lesson1.yml) | L1 refactor, cell-level |
| [lesson2.yml](lesson2.yml) | L2 refactor, cell-level; implements the L2/L4 re-split |
| [../docs/publishing-caches-to-huggingface.md](../docs/publishing-caches-to-huggingface.md) | how to publish and consume a VectorCache on the hub |

## Where things stand (2026-07-30, evening)

**Lesson 1 is implemented.** All 8 notebooks run top to bottom (2m17s for the lesson),
zero `mads_datasets` imports, kill-list rows green.

**Lesson 2 is implemented too** (built in parallel, see [lesson2.yml](lesson2.yml) for the
per-task status and corrections C1-C7). New arc: `01_losses` (unchanged) → `02_convolutions`
(dataset swapped) → **`03_pretrained`** (new, the frozen-encoder counterpart) → `04_mlflow`
(rewritten) → `05_exercises.md`. Two things were deliberately *not* done: `01_losses` did not
move to L1 (L2.Q3 is unanswered, and L1 was being edited concurrently), and nothing was removed
from L4 yet (the L2.5 handover).

Done:

- **P0.6** — `mltrainer` no longer rebuilds its dataloader iterator every step. Fixed in
  `../../mltrainer`, 4 regression tests, **released as 0.3.2 on PyPI**.
- **P0.1 / P0.2** — `conventions.md` and `tensor-ladder.md`. The ladder page moved into
  `notebooks/1_pytorch_intro/` and was rebuilt around the formal/semantic split.
- **P0.4 / P1.5** — `datasets` and `huggingface-hub` declared; README dependency story
  rewritten around `datasets` + `vectormesh` + `mltrainer`.
- **L1.1 - L1.7** — see [lesson1.yml](lesson1.yml) for per-notebook status and measured
  runtimes.

Renames: `05_image_classfication` → `05_classification`, `05_classfication` →
`06_vectors`, `06_exercises` → `08_exercises`.

### What blocks a real student run

**vectormesh 0.3.0 is not published**, and L2 needs it. The published release is 0.2.0, which has
no `ImageVectorizer`, `DatasetSchema` or `GaussianNoise` — those exist only in the
`../vectormesh` working tree. `03_pretrained.ipynb` was verified with
`PYTHONPATH=../vectormesh/src` (deliberately not an editable install, to avoid disturbing the L1
thread's venv). Publish 0.3.0 and raise the `pyproject.toml` floor. Exactly the shape of the
mltrainer 0.3.2 blocker, one lesson later.

Also worth knowing before building more caches: **`VectorCache.create` costs a flat ~21 seconds
per call** regardless of dataset size, because `datasets.map` pickle-hashes the whole encoder to
fingerprint the operation. Measured flat from n=250 to n=2000; bypassing it gives 940 img/s. See
lesson2.yml C2.


**P0.3 + Q8 — publish the IMDB granite cache.** Measured: the embedding pass is ~16 min
per split on MPS and **~125 min per split on CPU**.
So the cache download is mandatory, not optional, and `06_vectors.ipynb` currently falls back
to a local cache that only exists on Raoul's machine. The hub id is a single constant in that
notebook (`CACHE_REPO`), waiting on the user-vs-organisation decision (Q8).

## The one-paragraph version

The course moves from hand-rolling every architecture against `mads_datasets` to
**using pretrained models where you can and improving where you must** — combining and
creating with foundation models at the vector level, via `vectormesh`. Datasets come from
the HF hub; expensive encoding happens once and is published as a `VectorCache`; students
train small heads on frozen vectors on a laptop CPU. Hand-rolled implementations survive
only where they explain what is inside a frozen model, or where the student genuinely
builds something. Everything added must displace something — see the `cuts` register.

## Working agreements

- **The notebook wins.** When a plan contradicts the actual cells, fix the plan.
  `lesson1.yml` contributed corrections C1-C3, which became decisions D4/D8 and rewrote
  two masterplan tasks.
- **A migration is done when it has been run**, not when it has been written
  (masterplan D2). The kill list tracks that distinction explicitly.
- **Verify before asserting.** Several plan items were wrong until tested: Monash's
  sunspot series is daily not monthly, `datasets` 5.0 refuses script-based datasets, and
  folder-structured image datasets have effectively vanished from the hub. Each was a
  one-minute check that changed a decision.
