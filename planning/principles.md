# Principles

What this course teaches, why, and the rules that keep it honest. The lesson-by-lesson
map is `curriculum.md`; the numbers behind every claim here are in `evidence.md`; things
deliberately removed are in `cuts.md`.

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

The consequence is concrete: **the instructor runs step 1 and publishes the cache;
students live entirely in step 2.** A cached encoder solves the weak-hardware problem
better than a toy dataset does, on real data.

### 1.2 The cache pattern generalises past embeddings

Anything expensive, deterministic, and reusable is a cache: image and text embeddings,
patch tokens, and — in L7 — the outputs of a generative model. Precompute once, publish,
and let a whole class do the thinking on a CPU. When a lesson needs a model nobody can
run, the answer is a published cache with an optional live path, not a hardware floor.

### 1.3 This is not universal

Vectorization only pays where a good frozen encoder exists. It does **not** apply to:

- **Time series** (L2) — no general-purpose vectorizer worth freezing; windowing and
  training end-to-end remains correct. Measured: a linear model beats N-BEATS on most
  series, and a time-series foundation model's frozen embeddings score *worse than the
  raw window* (`evidence.md` §5).
- **Graphs as structure** — node embeddings are learned jointly with the task and depend
  on the graph, so they cannot be frozen and shipped the way an image encoder can.
- **Small domain-specific image sets far from web images** — worth *showing* the failure,
  which is what the DTD/EuroSAT accuracy gap gives us (`evidence.md` §2).

Teaching when the abstraction does **not** apply is as important as teaching when it does.
A student who vectorizes a time series because "that's what we did in lesson 1" has
learned the wrong thing.

### 1.4 Architecture as composition, searched

Once vectors are on disk, the design space between the vectors and the loss is reachable
in seconds: aggregate, fuse, gate, project, route. `Serial`/`Parallel` make those
recombinations cheap, so architecture becomes something you **search**, not something you
guess. That is why ray.tune belongs in lesson 1 — it is the natural tool the moment a
training run costs two seconds.

---

## 2. No free lunch — the actual pedagogical goal

The measurements in `evidence.md` are emphatic: depth loses, fusion is a wash, gating
ties its capacity-matched control, sparse routing buys nothing here. It would be a
mistake to teach *those verdicts*. They are contingent — different data, a different
encoder pair, a genuinely complementary second stream, and the sign flips. Sometimes
fusion wins. That is what no-free-lunch means, and it is why a fixed answer is the wrong
thing to hand a student.

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
- **Every capacity claim needs a capacity-matched control.** "The gate helped" means
  nothing against a model with fewer parameters. This is the single most repeated finding
  in `evidence.md` and it is the organising question of L3.
- **Pose a design as a hypothesis, not a demonstration.** Have students predict the sign
  before running it. A wrong prediction they made themselves teaches more than a correct
  one they were handed.
- **Report negative results as results.** "We tried it, the baseline won, here is the
  number" is a complete and respectable outcome. A course that only ever shows things
  working teaches students to hide the runs that did not.
- **Do not over-generalise our own findings either.** `evidence.md` measured *these*
  encoder pairs on *these* datasets. Written as "fusion never helps" it becomes exactly
  the kind of received wisdom this section exists to prevent.
- **Enough seeds to have the right to a conclusion.** Three seeds have produced a false
  finding twice in this repo (§1, §7). Both the wrong number and the correction stay in
  the notebook.

The habit is transferable and the verdict is not. Grade the habit.

---

## 3. Standing rules

- **Intent beats artifact.** When a plan **prescribes** a change and the cells have not
  caught up, the plan is right — change the cells. A refactor plan is *supposed* to
  contradict the notebooks; that is its job. When a plan merely **describes** what the
  code does and the cells disagree, the cells are right — fix the plan, because a plan
  that misdescribes the repo is worse than no plan once it gets trusted.

  Rule of thumb: a disagreement about a fact (a shape, a row count, an accuracy, which
  import is used) is settled by the code. A disagreement about a decision is settled by
  the plan, until the decision is revised here.

  **Watch for plans that have gone stale.** A prescription written many commits ago may
  no longer be the right call — the repo it assumed has moved. Before executing an old
  plan item, check that it still makes sense; the authority of a plan is not permanent,
  and re-deciding is cheaper than un-building.
- **Verify before asserting.** Everything in `evidence.md` is a number that was actually
  produced on this machine, not a recollection. Several decisions reversed once measured.
- **A notebook counts as done when it has been run**, not when it has been written.
- **Never assume an epoch count for a number you report.** This is basic training hygiene
  and it belongs in every trainer, students' included. The two cases are different:
  - **While searching**, under-training is fine *provided every config gets the same
    budget*. You are ranking configs, not measuring one — 10 epochs each when 50 would be
    optimal is a legitimate, stated economy.
  - **For any number that goes in a table** — a final architecture, a baseline, an
    ablation arm — the budget must be justified, not chosen. Train to a plateau
    (early stopping) or show the convergence curve, and say which. A single (epochs,
    seeds) configuration is not a result.

  This is not pedantry. An unjustified epoch count has changed the **sign** of a
  conclusion twice in this repo: `evidence.md` §7 (a 3-seed check found a gate beating its
  control; 5 seeds × 3 epoch counts erased it) and §11 (12 epochs said supervised
  representations transfer for free; 60 epochs said the opposite at 11 sigma). Both times
  the error was caught by someone asking, not by the process.
- **`param(full, test=...)` on anything that scales runtime** (epochs, steps, trials,
  dataset size, seeds). See `CLAUDE.md`.
- **Published libraries only.** `vectormesh` and `mltrainer` ship a release *before* a
  notebook depends on the change. No editable installs, no `PYTHONPATH` shims — students
  will not have one.
- **No notebook builds a full cache.** Every notebook downloads a published one. Building
  is taught on a ~200-item subsample that finishes on any hardware; the full build lives
  in `make build-lesson1`. See `evidence.md` §4 for the throughput numbers behind this.
- **Any notebook that subsamples a cache shuffles first.** The DTD caches are grouped by
  label; `select(range(n))` returns a class-truncated subset that reads like a learning
  curve and is not one. `evidence.md` §3.
- **Cut, do not stage.** If material no longer reflects practice, remove it — do not
  invent a scenario that keeps old teaching code alive. See `cuts.md`.

---

## 4. Hardware floor

Students range from new MacBooks to CPU-only VMs. Every graded notebook runs on the
weakest of those. Where a lesson genuinely wants a model that does not fit — a time-series
foundation model, an LLM — the pattern is:

- **Default, everyone:** a published cache of that model's outputs. Deterministic, free,
  CPU-only, identical results for every student.
- **Opt-in, if you have the hardware:** regenerate the cache yourself, clearly marked
  optional. For LLM work this means a local ~3B-class model via ollama; students without
  the RAM use the cache and lose nothing gradable.

Framing matters: "encoding needs hardware you may not have" is a fact about the split,
not an apology. It is the same asymmetry that exists in industry, and it is why the cache
is a deliverable. Students without a GPU are not doing a reduced version of the course —
they are doing the part where the thinking is.

---

## 5. This folder

| File | Holds | Lifetime |
|---|---|---|
| `principles.md` | this file — thesis, stance, standing rules | durable |
| `curriculum.md` | the lesson map and each lesson's target | durable |
| `evidence.md` | every measured number the decisions rest on | durable, append-only |
| `cuts.md` | what was removed and why, so it is not re-added | durable |
| `lesson<N>.md` | build plan for a lesson currently being rebuilt | **deleted once the lesson has been run** |

A shipped lesson is described by its notebooks, not by a plan file. Keeping a 500-line
spec next to a notebook that already runs guarantees the two drift and the spec gets
believed. Durable knowledge (what we measured, what we decided, what we cut) is kept;
build scaffolding is thrown away.
