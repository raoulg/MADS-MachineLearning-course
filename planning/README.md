# planning/ — the vectormesh refactor

Read these in order. `masterplan.yml` is the source of truth; the per-lesson plans
correct it when a notebook disagrees.

| file | what it is |
|---|---|
| [masterplan.yml](masterplan.yml) | principles, decisions D1-D11, 7 phases, cuts register, risks, backlog |
| [mads_datasets_killlist.md](mads_datasets_killlist.md) | the 14 import sites, their replacements, and whether each has been **run** |
| [lesson1.yml](lesson1.yml) | L1 refactor, cell-level |
| [lesson2.yml](lesson2.yml) | L2 refactor, cell-level; implements the L2/L4 re-split |
| [../docs/publishing-caches-to-huggingface.md](../docs/publishing-caches-to-huggingface.md) | how to publish and consume a VectorCache on the hub |

## Where things stand (2026-07-30)

**Planning is paused. Execution starts on L1 + L2.** Plans for lessons 3, 4 and 5 are
deliberately deferred: the remaining "what do we cut" decisions depend on how much
material L1 and L2 actually hold once built, and guessing at that would be planning
against numbers we don't have.

**Start with P0.6** — the mltrainer iterator fix. It lives in another repo, it blocks the
`num_workers` default and every timing claim in both lessons, and until it lands, any
notebook using a `DataLoader` reports validation metrics computed on one repeated batch.

Already done:

- `notebooks/1_pytorch_intro/05_classfication.ipynb` — dead `mads_datasets` import and
  unused `TOMLSerializer` removed; stale "we will use fashion MNIST" header fixed

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
