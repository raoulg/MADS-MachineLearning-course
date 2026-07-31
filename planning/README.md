# planning/ — the vectormesh refactor

Read these in order. `masterplan.yml` is the source of truth; the per-lesson plans
correct it when a notebook disagrees.

| file | what it is |
|---|---|
| [masterplan.yml](masterplan.yml) | principles, decisions D1-D11, 7 phases, cuts register, risks, backlog |
| [conventions.md](conventions.md) | cache layout, naming, dataloading, tracking, shape vocabulary (P0.1) |
| [../notebooks/1_pytorch_intro/tensor-ladder.md](../notebooks/1_pytorch_intro/tensor-ladder.md) | shape vs meaning, and the four rungs (P0.2) |
| [mads_datasets_killlist.md](mads_datasets_killlist.md) | the 17 import sites, their replacements, and whether each has been **run** |
| [lesson1.yml](lesson1.yml) | L1 refactor, cell-level |
| [lesson2.yml](lesson2.yml) | L2 refactor, cell-level; implements the L2/L4 re-split |
| [../docs/publishing-caches-to-huggingface.md](../docs/publishing-caches-to-huggingface.md) | how to publish and consume a VectorCache on the hub |

## Where things stand (2026-07-31)

**Lessons 1 and 2 are implemented and verified**, both against the *published*
`vectormesh 0.4.0` and `mltrainer 0.3.4` — no editable installs, no `PYTHONPATH` shims.

L2's arc (see [lesson2.yml](lesson2.yml) for per-task status and corrections C1-C10):
`01_convolutions` (dataset swapped) → **`02_pretrained`** (new, the frozen-encoder
counterpart) → `03_mlflow` (rewritten around the comparison) → `04_exercises.md`, with
`cnn.py` holding the CNN that 01 builds and 02/03 import.

**`01_losses` moved to lesson 1** as `05_losses`, ahead of `06_classification` (L2.Q3
answered). That is why both lessons renumbered — L1 gained a notebook in the middle, L2 lost
its first one. Goal 2.1 folded into les1.md goal 1.9, which had been pointing across the
lesson boundary at `2_convolutions/01_losses.ipynb` all along; fixing that inconsistency is
what motivated D9.

**The L2/L4 duplication is deliberate, and it stays for now.** Nothing has been removed from
lesson 4 — `03_minimal_transfer_learning` and `04_transfer_learning_with_resnet` are untouched
— so transfer learning currently exists in both places. Sequencing decided by the owner on
2026-07-31: **build L2 first, then decide what L4 drops, having seen the built L2 rather than a
plan.** So do not "tidy up" the overlap, and do not open lesson4.yml, until that call is made.

Done:

- **P0.6** — `mltrainer` no longer rebuilds its dataloader iterator every step. Fixed in
  `../../mltrainer`, 4 regression tests, **released as 0.3.2 on PyPI**.
- **P0.1 / P0.2** — `conventions.md` and `tensor-ladder.md`. The ladder page moved into
  `notebooks/1_pytorch_intro/` and was rebuilt around the formal/semantic split.
- **P0.4 / P1.5** — `datasets` and `huggingface-hub` declared; README dependency story
  rewritten around `datasets` + `vectormesh` + `mltrainer`.
- **L1.1 - L1.7** — see [lesson1.yml](lesson1.yml) for per-notebook status and measured
  runtimes.

Renames, L1: `05_image_classfication` → `06_classification`, `05_classfication` →
`07_vectors`, `06_exercises` → `09_exercises`, plus `07_extra_autograd` → `08_extra_autograd`
to make room for `05_losses`.

### What blocks a real student run

**vectormesh is no longer a blocker.** 0.4.0 is on PyPI with `ImageVectorizer`,
`DatasetSchema` and `GaussianNoise`; `pyproject.toml` pins `vectormesh>=0.4.0` and
`mltrainer[tune]>=0.3.4`. (If `uv sync` still resolves 0.2.0, its index cache is stale —
`uv sync --refresh-package vectormesh`.)

Still worth knowing before building more caches: **`VectorCache.create` costs a flat ~21
seconds per call** regardless of dataset size, because `datasets.map` pickle-hashes the whole
encoder to fingerprint the operation. Measured flat from n=250 to n=2000; bypassing it gives
940 img/s. Re-checked against 0.4.0 on 2026-07-31 — **still present**. `02_pretrained` builds
four caches, so a cold run pays ~88s of pure hashing, and the notebook carries a caveat
paragraph about it that should be deleted once this is fixed. See lesson2.yml C2.


**P0.3 + Q8 — publish the IMDB granite cache.** Measured: the embedding pass is ~16 min
per split on MPS and **~125 min per split on CPU**.
So the cache download is mandatory, not optional, and `07_vectors.ipynb` currently falls back
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
