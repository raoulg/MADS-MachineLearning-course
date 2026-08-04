# planning/ — the vectormesh refactor

Read these in order. `masterplan.yml` is the source of truth; the per-lesson plans
correct it when a notebook disagrees.

| file | what it is |
|---|---|
| [masterplan.yml](masterplan.yml) | principles, decisions D1-D23, 7 phases, cuts register, risks, backlog |
| [conventions.md](conventions.md) | cache layout, naming, dataloading, tracking, shape vocabulary (P0.1) |
| [../notebooks/1_pytorch_intro/tensor-ladder.md](../notebooks/1_pytorch_intro/tensor-ladder.md) | shape vs meaning, and the four rungs (P0.2) |
| [mads_datasets_killlist.md](mads_datasets_killlist.md) | the 17 import sites, their replacements, and whether each has been **run** |
| [lesson1.yml](lesson1.yml) | L1 refactor, cell-level |
| [lesson2.yml](lesson2.yml) | L2 refactor, cell-level; implements the L2/L4 re-split |
| [lesson3.yml](lesson3.yml) | L3 refactor, cell-level; the dataset-gap lesson (sunspots + gestures) |
| [lesson4.yml](lesson4.yml) | L4 refactor, cell-level; the cost of an experiment, and the L4 handover D9 left open |
| [lesson5.yml](lesson5.yml) | L5 refactor, cell-level; attention, and what to do when a document does not fit |
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
lesson 4 — `03_minimal_transfer_learning` and `04_transfer_learning_with_resnet` are untouched,
and **this is now the settled state, not a pending call**. The owner decided on 2026-08-01
(**D19**) not to collapse them: "leave them, add new notebooks besides those." So transfer
learning stays duplicated between L2 and L4 indefinitely, on purpose — see
[lesson4.yml](lesson4.yml) D19 and masterplan `cuts.phase_2.l2_accounting.still_owed`. The owner
separately floated moving transfer learning to L2 entirely, which would resolve this a different
way, but explicitly deferred it (lesson4.yml L4.Q8) — **do not act on either without being asked.**

**Lesson 4's direction: go deeper into ray, not sideways into cost-accounting.** Two things are
now decided and built:

- **D21 — `tune.run` removed, `tune.Tuner` in.** All five searches in `01_hypertuner.ipynb`
  (random, grid, bayes, hyperband, hyperopt) migrated from `tune.run` to
  `tune.Tuner(...).fit()` + `ResultGrid`. `visualize.py`'s `parallel_plot` now takes a dataframe
  directly instead of an `analysis` object. **Verified: full notebook re-executed via
  notebooktester after the migration, passes in 66s.**
- **D19 — a new notebook, not a collapsed one.** `01_hypertuner` keeps searching a head on the
  flowers cache (unaffected by the direction change — it already matched what the owner wants).
  New: **`05_architecture_search.ipynb`** (planned, not yet built) — ray searches *which
  vectormesh building blocks* compose the head (plain MLP vs `MoE` vs `Highway`-wrapped),
  alongside hidden_size/depth, on the same flowers cache. Component signatures verified against
  vectormesh 0.4.1 source. Content plan in lesson4.yml L4.7.

`03_finetune_vs_frozen.ipynb` and the optional `04_schedulers.ipynb` are **cancelled**, along
with the make_room payment they would have made — L4 now adds a notebook with no offsetting cut,
an explicit owner-sanctioned exception (lesson4.yml `make_room.status`). **D20 (hymenoptera) is
deferred**, not decided: the notebook that uses it is untouched anyway.

Still true: **lesson 4 did not run at all before today** — `ray.train.report` raised inside a
Tune function on ray 2.55, so every trial in `01_hypertuner` errored, and `hypertune.py` failed
the same way through mltrainer's `ReportTypes.RAY`. That is masterplan **P0.7**. The mltrainer
half shipped in **two parts**: `RayReporter` and one mkdir race are fixed and published as
**mltrainer 0.3.5**; a *second* mkdir race (`Trainer.dir_add_timestamp`) is fixed in source and
version-bumped to **0.3.6** but **not yet built or published**.
**Known issue right now: `pyproject.toml` was bumped to `mltrainer[tune]>=0.3.6` ahead of that
publish, so a fresh `uv sync`/`uv run` fails to resolve** (0.3.6 does not exist on PyPI yet;
`uv.lock` is still on 0.3.5). `uv run --frozen` works around it using the existing lock. This
needs either the 0.3.6 publish to land, or the pin to drop back to `>=0.3.5` until it does.

The lesson's own thesis is measured: a 5-epoch trial costs **1.2 s** on the flowers cache,
**159 s** on pixels, and fine-tuning resnet18 costs **135 s per epoch** — so hypertuning becomes
teachable exactly where D9 said it would. The surprise: **ray's per-trial process overhead
(~2-3 s) is larger than the trial it schedules**, which turns "what is ray for?" into a
measurement with a boundary.

One knock-on worth knowing: **L4 has left the gestures blocked group**. Its search runs on an
image cache, so it has no GESTURES site at all — **L3.Q1 decided for L3 and L5 only**, and both
are now resolved to `monster-monash/WISDM` (**D22**).

**Lesson 3 is planned, not built** ([lesson3.yml](lesson3.yml), written 2026-07-31 ahead of
lesson4 at the owner's instruction, revised 2026-08-01 after measuring five candidate series
and a foundation model). It carries **D13** (Monash over SILSO, and the parquet-branch loader
that only works with explicit `data_files`), **D14** (the taught series is
`pedestrian_counts`, not sunspot), **D15** and **D16** — see below.

**The gestures replacement is decided: `monster-monash/WISDM`** (**D22**, 2026-08-01) —
17,166 × 3 × 100, 6 classes, 36 subjects, 20.6 MB, two `hf_hub_download` calls. It measures
0.902 on a random split and **0.761 on the dataset's own subject fold**, and that 14-point gap
is the leakage artefact goal 3.3 has never had. 3 channels means `input_size=3` is unchanged,
so the edit is confined to the loading cells in L3/04 and L5/03, done in one pass. **This was
the last open dataset gap in the plan.**

### Three measured results that change what L3 claims

Full protocol and caveats in [lesson3.yml](lesson3.yml) C7–C11; summarised as masterplan
D15/D16.

1. **A linear layer beats the GRU**, on all four candidate series, at ~100× less compute.
   That is the LTSF-Linear result reproduced, and today's `03_basemodels` arc implies the
   opposite. Teach it — it is the north star pointed at architectures.
2. **TimesFM 2.5 zero-shot beats everything trained** — 0.140 against the best trained
   model's 0.216, no training, and *no new dependency*: `TimesFm2_5Model` is in the
   installed transformers 5.12.1.
3. **The freeze-and-cache pattern loses here** (0.499 vs 0.216 for a linear layer on raw
   values). Two hypotheses were tested and both falsified. The rule of thumb that survives:
   *if the encoder expands the data it is re-encoding, not extracting* — images compress
   ~100×, a 512-step window expands **40×**. The course gets to show its own central
   technique failing, which is worth more than another success.

And one habit worth carrying into every lesson that touches a foundation model: **check the
pretraining corpus before quoting a zero-shot number.** `pedestrian_counts` is in LOTSA and
in `chronos_datasets`; TimesFM's card documents no corpus at all. `Salesforce/GiftEval`
(apache-2.0, parquet-native, 28 configs) is where an honest number comes from.

Sunspot is not discarded — it becomes exercise E4, carrying the two things pedestrian cannot:
3.240 NaNs all sitting before 1849, and a dominant cycle ~32× longer than any window the
lesson uses.

**One L3 notebook is built and executed**, ahead of the rest of the lesson refactor:
[`07_foundation_models.ipynb`](../notebooks/3_recurrent_networks/07_foundation_models.ipynb),
at the owner's direct request. Standalone (does not depend on the not-yet-built 01-04), 17
cells, **passes notebooktester in 39s**. Walks through the TimesFM zero-shot win, the
frozen-encoder loss and its diagnosis, the storage-expansion rule of thumb, and the
pretraining-corpus contamination check — see lesson3.yml **L3.Q6**.

**Two general principles settled 2026-08-01, not just for L3** (lesson3.yml **L3.Q3/L3.Q4**,
now in [conventions.md](conventions.md) §1a/§5a): a `VectorCache` is used *if it helps* —
reduces compute, eases training, improves performance — never because it's mandated; and
`vectormesh.components.metrics` is preferred over `mltrainer.metrics` wherever the two overlap,
as an interim step toward `Trainer` itself eventually folding into vectormesh.

**Lesson 5 is planned, and its two new notebooks are built and executed**
([lesson5.yml](lesson5.yml), written 2026-08-01; notebooks the same day). It carries **D17**,
**D18** and **D23**.

- **IMDB leaves lesson 5** (**D17**, dataset `scotus` confirmed by the owner). It cannot teach
  chunking, because it does not chunk: **86% of reviews are a single 512-token chunk**. The
  replacement — `coastalcph/lex_glue` config `scotus`, CC-BY-4.0, parquet-native, 13
  imbalanced classes — has **median 6.496 tokens and 18.3 chunks per document**.
- **P2.6 was already built — in lesson 1.** So L1 keeps chunking-as-mechanics and L5 takes
  chunking-as-a-decision (**D18**): *the context window is a budget, measure what truncation
  costs you.* The phase_2 aggregator-ablation exit criterion is ticked at L1.
- **[`05_long_documents.ipynb`](../notebooks/5_attention/05_long_documents.ipynb)** — built,
  runs green, three seeds per configuration *inside the notebook*: truncation costs ~4
  accuracy points, the curve flattens by C=4, and the learned aggregators (attention,
  a 1M-parameter TransformerBlock) do not beat a plain mean.
- **[`06_composition.ipynb`](../notebooks/5_attention/06_composition.ipynb)** (**D23**, at the
  owner's request) — built, runs green. The composition half of vectormesh the course never
  taught (`Parallel`, `Concatenate2D/3D`, `CollateParallel`, `Gate → Highway → MoE`), on two
  streams: the granite embedding and a **designed US-legal regex stream**
  ([`legal_patterns.py`](../notebooks/5_attention/legal_patterns.py), new — vectormesh only
  ships the Dutch one; 58 chunk-aligned features, 0 alignment mismatches over 900 docs). Its
  centrepiece is a **measured negative result**: chunk-level fusion appears to win (0.609 vs
  0.584) until the one-variable ablation shows the win came from a Projection bottleneck, not
  the regex or the MoE (same shape, no regex: **0.624**). Accuracy and macro-F1 rank the wide
  head and the bottleneck in opposite orders, which hands the metrics notebook its opening
  problem. It also ships the best deliberate bug in the course so far: a regex cache built
  with the wrong `context_size` **concatenates cleanly and silently means nothing** — shape
  contracts catch rank errors, not semantic misalignment.

With D23, L5 **breaks even** under make_room instead of paying a surplus (8 notebooks in,
8 out; roughly −64 cells, +59). The shrink lever, if ever needed, is L5.Q6 (fold
`03_transformer` into `05_long_documents`). Three of its five kill-list rows still close by
*deletion* — the old `05_imdb_preprocessing`/`06_imdb_attention` are deleted in the L5.2/L5.9
pass, so the directory temporarily holds both numbering generations.

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


**P0.3 + Q8 — DONE 2026-07-31.** Four caches are published under
[pttrn-io](https://huggingface.co/pttrn-io) and round-tripped from a clean `local_dir`
(~2s per image split, ~4s per IMDB split), so `07_vectors.ipynb` no longer falls back to
building one. That matters because the embedding pass is ~16 min per split on MPS and
**~125 min per split on CPU** — the download is mandatory, not a convenience. Publishing is
`make publish-all` via `scripts/publish_cache.py`.

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
