# Main principles — round 2 refactor

The organising idea of this round: **students should work one abstraction level up.**
Not "write a training loop for a CNN on MNIST", but "pick a representation, compose a
head, and let a search tell you which composition was worth it".

`vectormesh` is what makes that affordable. This document holds the principles, the
numbered decisions (D1, D2, …), the measured evidence those decisions rest on, and the
cuts register. Per-lesson detail lives in `lesson<N>.yml`.

---

## 0. Standing rules

- **Reality beats description; intent beats artifact.** These are two different cases and
  collapsing them causes damage:
  - When this plan **describes** what the code does and the cells disagree, the cells are
    right — fix the plan. A plan that misdescribes the repo is worse than no plan, because
    it gets trusted.
  - When this plan **prescribes** a change and the cells have not caught up, the plan is
    right — change the cells. A refactor plan is *supposed* to contradict the notebooks;
    that is its entire job. "The notebook wins" would mean MNIST survives every round
    simply by already existing.

  Rule of thumb: if the disagreement is about a fact (a shape, a row count, an accuracy,
  which import is used), the code wins. If it is about a decision, the plan wins until the
  decision is explicitly revised here.
- **Verify before asserting.** Several items in this file were wrong until measured.
  Check the dataset, the shape, the accuracy — do not assume. Everything in §3 is a
  number that was actually produced on this machine, not a recollection.
- **A notebook counts as done when it has been run**, not when it has been written.
- **`param(full, test=...)` on anything that scales runtime** (epochs, steps, trials,
  dataset size, seeds). See `CLAUDE.md`.
- **Published libraries only.** `vectormesh` and `mltrainer` ship a release *before* a
  notebook depends on the change. No editable installs, no `PYTHONPATH` shims —
  students will not have one.

---

## 1. The thesis

### 1.1 Embed once, train many heads

Running a large pretrained encoder over your data every epoch, for every experiment, is
pure waste — you recompute identical vectors to answer questions that have nothing to do
with the encoder. `vectormesh` splits it:

1. **Encode** — frozen pretrained model, raw data → vectors. Expensive, needs hardware,
   done **once**, written to disk as a `VectorCache` with `metadata.json`.
2. **Head** — a small trainable network on top of those vectors. Cheap, done hundreds of
   times, runs on a laptop CPU in seconds.

The consequence for this course is concrete: **the instructor runs step 1 and publishes
the cache; students live entirely in step 2.** MNIST was chosen years ago because it ran
on weak hardware. A cached encoder does that better, on a real dataset.

### 1.2 This is not universal

Vectorization only pays where a good frozen encoder exists. It does **not** apply to:

- **Time series** (L3) — no general-purpose vectorizer worth freezing; windowing and
  training end-to-end remains correct. Confirmed by measurement: a linear model beat
  N-BEATS on both pedestrian_counts and ETTh1.
- **Graphs** (L7) — structure is the signal; there is no per-node encoder to cache.
- **Small domain-specific image sets far from web images** — worth *showing* the failure,
  which is what the DTD/EuroSAT accuracy gap gives us (§3.2).

Teaching when the abstraction does **not** apply is as important as teaching when it does.
A student who vectorizes a time series because "that's what we did in lesson 1" has
learned the wrong thing.

### 1.3 Architecture as composition, searched

Once vectors are on disk, the design space between the vectors and the loss is reachable
in seconds: aggregate, fuse, gate, project, route. `Serial`/`Parallel` make those
recombinations cheap, so architecture becomes something you **search**, not something you
guess. That is why ray.tune moves to lesson 1 — it is the natural tool the moment a
training run costs two seconds.

---

### 1.4 No free lunch — the actual pedagogical goal

The measurements in §3 are emphatic: fusion loses, depth loses, the baseline wins. It
would be a mistake to teach *those verdicts*. They are contingent — different data, a
different encoder pair, a genuinely complementary second stream, and the sign flips.
Sometimes fusion wins. That is what no-free-lunch means, and it is why a fixed answer is
the wrong thing to hand a student.

**What we are actually teaching is the loop:**

> have an idea → state it as something that could fail → build it cheaply → measure it
> against a baseline you established *first* → report what happened, including "nice idea,
> simpler won"

The cache is what makes that loop affordable — a hypothesis costs two seconds — and that
is the real reason lesson 1 starts there.

Consequences for how notebooks are written:

- **Never present an architecture as correct because the idea is elegant.** Gating,
  fusion, mixtures of experts are appealing *ideas*; whether they earn their parameters on
  your data is an empirical question with a cheap answer.
- **Pose fusion as a hypothesis, not a demonstration.** Have students predict the sign
  before running it. A wrong prediction they made themselves teaches more than a correct
  one they were handed.
- **Report negative results as results.** "We tried it, the baseline won, here is the
  number" is a complete and respectable outcome. A course that only ever shows things
  working teaches students to hide the runs that did not.
- **Do not over-generalise our own findings either.** §3 measured *these* encoder pairs on
  *these* datasets. Written as "fusion never helps" it becomes exactly the kind of
  received wisdom this section exists to prevent.

The habit is transferable and the verdict is not. Grade the habit.

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | L1 is rebuilt cache-first. Students download a published cache before they write any model code. | The economics only land if they *feel* the 2-second training run. |
| **D2** | MNIST / Fashion-MNIST is cut as the spine of L1. | Chosen for weak hardware; the cache solves that better on a real dataset. See `cuts`. |
| **D3** | **EuroSAT** is the demo dataset; **DTD** is the exercise dataset. | Measured: EuroSAT 94.5% (impressive, not saturated), DTD 78.5% (real headroom). §3.2 |
| **D4** | Flowers is **not** the L1 spine — it is a single demo cell. | Measured saturated: 99.3% trained, **99.1% with 1-NN and no training at all**. §3.1 |
| **D5** | ray.tune from L1, TPE (`HyperOptSearch`) only. Algorithm zoo stays in L4. | One tool, used properly, beats five tools named. |
| **D6** | The L1 search space includes **encoder choice** alongside width/depth/noise/lr. | Measured: encoder moves accuracy 10–40 pts; head architecture moves it <1 pt and usually downward. §3.3 |
| **D7** | Nonlinearity is taught once, unifying dense and conv: `σ(Wx + b)` and `σ(W * x)`. | Convolution is a linear operator with weight sharing; the nonlinearity is the shared essential part. Lets L1 absorb the old L2 conv material. |
| **D8** | Old L1 + L2 notebooks move to `notebooks/legacy/`, not deleted. | Reference during transition; deleted once the replacements have been run. |
| **D9** | L2 becomes "produce a cache + track experiments" (was: convolutions/pretrained/mlflow). | Clean split: L1 consumes a cache, L2 produces one. |
| **D10** | Two-encoder fusion is taught as a **hypothesis the student tests**, and the notebook is honest that simpler won. | No free lunch. The deliverable is the research habit, not the verdict. See §1.4. |
| **D11** | `VectorCache.from_hub()` and `get_or_create_cache` move **into vectormesh**. | Blocking prerequisite — see §4. |
| **D12** | Caches are joined on `source_idx`, with an explicit alignment assertion. | Row order happens to match today; relying on that silently is how you get a model trained on mismatched pairs. |
| **D13** | **No notebook builds a full cache.** Every notebook downloads a published one. Building is taught on a ~200-image subsample that finishes on any hardware, and the full build lives in `make build-lesson1`. | Student hardware ranges from new MacBooks to CPU-only VMs. See §3.7. |

---

## 3. Evidence register

Measured on this machine, 2026-08-04, vectormesh 0.4.4. Reproduce with the scripts noted.
**These numbers are the reason several of the decisions above are not what we first
intended.** Anything added here must be a number someone actually ran.

### 3.1 Flowers is saturated — the representation *is* the model

Oxford Flowers-102, 4000 train / 1000 test, published caches, `NeuralNet(dim, 102)` head:

| encoder | dim | accuracy |
|---|---|---|
| dinov2-small | 384 | **0.9932** |
| mobilenet_v2 | 1280 | 0.9082 |
| resnet-18 | 512 | 0.9025 |

Total runtime for all three, CPU: **~9 seconds.**

Checked for leakage, because 99.3% on 102 classes deserves suspicion:

- train and test come from **different source splits** (`source_idx` 1–7168 vs 0–1019).
- test is balanced, 8–11 images/class; chance = 1.0%.
- near-duplicates: 4/1000 above cosine 0.99, 45/1000 above 0.95 — cannot carry the number.
- **resnet-18 gets 90% through the identical rows and head.** A leak would lift every
  encoder, not one.

The decisive statistic:

> **1-nearest-neighbour on raw DINOv2 vectors, zero training: 0.9910.**

The head contributes ~0.2 points. This is the single best demo in the course for "the
representation is the model" — and the reason flowers cannot be a tuning target.

### 3.2 Dataset ceilings (dinov2-small, 2000 train, 2-line head)

| dataset | classes | accuracy | verdict |
|---|---|---|---|
| oxford-flowers | 102 | 0.9932 | saturated — demo cell only |
| **eurosat** | 10 | **0.9447** | **D3 demo** — works impressively, not a lookup |
| resisc45 | 45 | 0.8209 | alternative |
| **dtd** (textures) | 47 | **0.7849** | **D3 exercises** — textures are out-of-domain for an object-centric encoder |
| stanford_cars | 196 | 0.6034 | hard; 2000 train over 196 classes is near few-shot |

### 3.3 Head architecture does not help — and it is not a learning-rate artifact

Oxford Flowers, resnet-18 cache, sweeping the head:

| head | 5/class | full 4000 |
|---|---|---|
| **baseline `NeuralNet(512, 102)`** | **0.7502** | **0.9025** |
| + GaussianNoise σ=0.05 | 0.7602 | 0.9006 |
| proj→64→head | 0.5467 | 0.8197 |
| proj→256→head | 0.6250 | 0.8412 |
| proj→1024→head | 0.6432 | 0.8314 |
| Skip block | 0.7063 | 0.8887 |

Every added layer **loses**, by up to 8 points. Ruled out the obvious confound — jointly
sweeping lr, full dataset:

| head | lr=1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| baseline | 0.8635 | 0.8938 | **0.9025** | 0.8971 |
| proj→256→head | 0.7939 | 0.8420 | 0.8412 | 0.7992 |

The baseline wins at **every** learning rate. Noise is a wash (±0.5 pt).

Repeated on **DTD** — a genuinely unsaturated task (0.78, not 0.99), to rule out "this is
only true because the task is easy". 3000 train / 1880 test, 47 classes:

| head | accuracy |
|---|---|
| baseline `NeuralNet(384, 47)` | 0.8033 |
| + GaussianNoise σ=0.3 | **0.8069** |
| proj→128→head | 0.7880 |
| proj→512→head | 0.7915 |
| Skip block | 0.7953 |
| resnet-18 baseline (weak encoder) | 0.6589 |
| fusion dinov2+resnet, width 128 | 0.7943 |
| fusion dinov2+resnet, width 384 | 0.7788 |

Same verdict on hard data: the two-line baseline plus a touch of noise wins, depth loses,
fusion loses. The one thing that moved the metric 14 points was **which encoder**.

### 3.4 Two-encoder fusion lost 5 out of 5

`Parallel([NeuralNet(d1,128), NeuralNet(d2,128)]) → Concatenate2D → head`:

| setup | best single | fusion | Δ |
|---|---|---|---|
| flowers, dinov2+resnet | 0.9932 | 0.9902 | −0.30 |
| fashion-mnist, dinov2+mobilenet | 0.9113 | 0.9055 | −0.58 |
| flowers, resnet+mobilenet | 0.9082 | 0.8945 | −1.37 |
| flowers 5/class | 0.9912 | 0.9750 | −1.62 |
| flowers 2/class | 0.9604 | 0.9266 | −3.38 |

The weak stream drags the strong one down, and **the penalty grows as data shrinks**.
This is the teachable result (D10): a second stream must add *complementary* information,
not just parameters. Do not sell fusion as an upgrade — make students measure it.

### 3.4b The encoder axis is not equally sharp on every dataset

Measured on the caches built by `make build-lesson1`, 2-line head:

| encoder | eurosat (6000) | dtd (3760) | flowers (4000) |
|---|---|---|---|
| dinov2-small | **0.9502** | **0.8033** | **0.9932** |
| resnet-18 | 0.9380 | 0.6589 | 0.9025 |
| mobilenet_v2 | 0.9258 | — | 0.9082 |
| **spread** | **2.4 pt** | **14.4 pt** | **9.1 pt** |

Consequence for D3/D6: **EuroSAT is a good demo but a poor search target.** Its encoder
axis is nearly flat, so a search there would show *nothing* moving — not the intended
lesson ("the encoder dominates"), just an uninformative flat result. The 14-point spread
on DTD is where that lesson is legible. `lesson1.yml` notebook 04 therefore runs on DTD,
not on the demo dataset. EuroSAT keeps notebooks 01–03, where the point is that the thing
works at all.

### 3.5 `Concatenate2D` cannot concatenate different widths — a vectormesh bug

Found while running §3.4. The docstring promises `ndim = dim1 + dim2 + ...`, but the
annotation forbids it:

```python
# connectors.py — Concatenate2D
tensors: tuple[Float[Tensor, "batch dim"], ...]     # `dim` BINDS across the tuple
# Concatenate3D, on the same file, gets it right:
tensors: tuple[Float[Tensor, "batch chunks _"], ...]  # `_` does not bind
```

jaxtyping binds a named axis once across a variadic tuple, so feeding `(B, 384)` and
`(B, 512)` raises `BeartypeCallHintParamViolation: ... dimension dim is 512 which does not
equal the existing value of 384`. Concatenating raw encoder widths — the most natural
fusion a student will try — is impossible today. Every fusion measurement in §3.4 had to
project both branches to a common width first.

`Concatenate3D` proves the intended fix: `"batch dim"` → `"batch _"`. One word.
`Stack2D`'s binding `dim1` is *correct*, since stacking genuinely requires equal widths.

### 3.6 Implication for the L1 tuning notebook

A search over head width/depth/fusion will **rediscover the two-line baseline**. That is
a fine lesson, but only if designed for: put encoder choice in the search space (where the
10–40 pt differences live), and let the architecture axes demonstrate that capacity added
after a frozen encoder is nearly always wasted. See `lesson1.yml`, notebook 04.

---

### 3.7 Encoding throughput, and why students must not do it

Measured, 200 DTD images, this machine (Apple silicon):

| encoder | CPU | MPS | speedup |
|---|---|---|---|
| dinov2-small | 81 img/s | 245 img/s | 3.0× |
| resnet-18 | 95 img/s | 481 img/s | 5.1× |

This is a *fast* CPU. An older laptop or a basic cloud VM lands nearer 10–30 img/s. The
whole lesson-1 build (eurosat 8000 + dtd 5640 rows, × 3 encoders) took **~90 s here** and
would take **most of an hour** on a weak CPU-only VM — repeated by every student, to
reproduce a byte-identical artefact that already exists.

Hence D13, and the two paths stated to students explicitly:

- **Default, everyone:** `VectorCache.from_hub(...)`. No encoder runs. Works identically
  on a 2015 laptop and an M4.
- **Opt-in, if you have the hardware:** build a cache for something nobody has published,
  and publish it. At 200 images the demo costs ~2.5 s on this CPU and maybe 20 s on a
  weak VM, so *everyone* can run the teaching version; only the full build needs hardware.

Framing matters: "encoding needs hardware you may not have" is a fact about the split, not
an apology. It is the same asymmetry that exists in industry, and it is why the cache is a
deliverable. Students without a GPU are not doing a reduced version of the course — they
are doing the part where the thinking is.

## 4. Blocking prerequisites

Ordered. Nothing in L1 can be finished before these land.

| # | Item | Where | Status |
|---|---|---|---|
| **P1** | `VectorCache.from_hub(repo, split=...)` — `VectorCache.load()` is local-disk only; every notebook currently open-codes `snapshot_download` + `load`. | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P2** | Promote `notebooks/5_attention/cache_utils.py` (`get_or_create_cache`, encoder-in-the-tag naming, metadata-mismatch raise, `datasets.disable_caching()`) into vectormesh proper. A course notebook folder is the wrong home for the shared helper. | vectormesh | todo |
| **P3** | `VectorCache.join(other, on="source_idx")` — fusion needs two caches as one dataset with two columns; today that is a hand-rolled `add_column` plus a `with_format(None)` dance that silently fails when the format is torch. | vectormesh | **coded**, branch `feat/from-hub-join` (D12) |
| **P4** | Build + publish **EuroSAT** and **DTD** caches, 3 encoders each, with `source_idx`. | `make build-lesson1` / `make publish-lesson1` | **done** 2026-08-04 |
| **P6** | Fix `Concatenate2D`'s binding axis (`"batch dim"` → `"batch _"`) so branches of different widths concatenate, as its docstring already promises. §3.5 | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P7** | sdist shipped 50 MB in 0.4.4, almost all of it `references/` — four copyrighted papers republished on PyPI. Restricted to source/tests/docs: 50 MB → 48 KB. | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P5** | Release vectormesh 0.5.0 (P1, P3, P6, P7) and bump the floor to `vectormesh>=0.5.0` in the course `pyproject.toml`. | both | **awaiting publish** |

Known API limit, not blocking: `CollateParallel` takes exactly **two** columns. Three-way
fusion needs a library change; L1 stays at two.

### Published caches (hf.co/pttrn-io)

Built and published 2026-08-04, `make build-lesson1 && make publish-lesson1`. Every one
carries `label`, `source_idx`, `embed` and a dataset card pinning the source-dataset and
encoder revisions.

| dataset | rows | encoders | measured 2-line-head accuracy |
|---|---|---|---|
| `eurosat-*` | 6000 / 2000 | dinov2-small, resnet-18, mobilenet_v2 | 0.9502 / 0.9380 / 0.9258 |
| `dtd-*` | 3760 / 1880 | dinov2-small, resnet-18, mobilenet_v2 | 0.8033 / 0.6589 / — |
| `oxford-flowers-*` | 4000 / 1000 | dinov2-small, resnet-18, mobilenet_v2 | 0.9932 / 0.9025 / 0.9082 |
| `fashion-mnist-*` | 4000 / 1000 | dinov2-small, mobilenet_v2 | 0.9113 / 0.8678 |
| `imdb-granite-…` | text, chunked | granite-embedding-small | L5 |

---

## 5. Cuts register

| Cut | Was | Why |
|---|---|---|
| MNIST / Fashion-MNIST as the L1 spine | `1_pytorch_intro/05_classification` | D2 — solved a hardware problem the cache solves better |
| Hand-rolled data generators | `1_pytorch_intro/02_datagenerators` | Superseded by `Collate` + `DataLoader`; the streaming case is a legacy note |
| Hand-rolled dataloader (full notebook) | `1_pytorch_intro/03_dataloader` | Compressed to one section of the L1 training notebook |
| Double for-loop hyperparameter search | L1/L2 exercises | D5 — replaced by ray.tune from the start |
| `mads_datasets` | L4 notebooks, `pyproject.toml` | Everything loads from the HF hub now; the dependency is dead weight |
| `tune.run` legacy API | `4_tuning_networks/hypertune.py` | ray 2.55 ships `tune.Tuner` + `tune.TuneConfig` + `tune.RunConfig`; no `ray.train` import needed |
| Separate "transfer learning" notebooks | `4_tuning_networks/03`, `04` | Transfer learning *is* the L1 premise now; a whole lesson restating it is redundant |

**Cut, do not stage.** If material no longer reflects practice, remove it — do not invent
a scenario that keeps old teaching code alive.

---

## 6. Lesson map after the refactor

| Lesson | Spine | Vectorized? |
|---|---|---|
| L1 | Cache → nonlinearity → training → ray search | yes (EuroSAT / DTD) |
| L2 | Build your own cache; encoder choice; track it (mlflow) | yes (produces caches) |
| L3 | Time series — windowing, base models, N-BEATS, foundation models | **no** (D-rationale §1.2) |
| L4 | The search algorithms themselves; the full ray zoo | yes |
| L5 | Attention, long documents, chunked text | yes (IMDB / legal) |
| L6 | Self-supervised — where representations come from | yes |
| L7 | Graphs | **no** |

Per-lesson detail in `lesson<N>.yml`. L1 is fully specified; L2 is specified to the level
needed to know what L1 hands over; L3–L7 record only what this round changes, since L3,
L5 and L6 were rebuilt in round 1 and L4/L7 are round-3 work.
