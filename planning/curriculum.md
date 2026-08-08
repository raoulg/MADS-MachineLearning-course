# Curriculum

Seven lessons. Each one states what it is for, what it must contain, and what it
deliberately excludes. Principles in `principles.md`, numbers in `evidence.md`, removals
in `cuts.md`.

The course is one argument, not seven topics:

> A frozen representation plus a cheap head beats almost anything you can train yourself
> (L1) — except where no such representation exists (L2), and the capacity you add on top
> mostly does not pay (L3), and long inputs make the representation a design problem (L4),
> and here is where representations come from in the first place (L5), and none of it is
> worth anything unless you can say how sure you are (L6), and the same logic applies to
> models you cannot train at all, where all the leverage is in what you feed them (L7).

| # | Folder | Spine | Vectorized? |
|---|---|---|---|
| L1 | `1_pytorch_intro` | Cache → tensors → nonlinearity → training → ray search | yes (EuroSAT / DTD) |
| L2 | `2_timeseries` | Windowing, baselines, decomposition + N-BEATS, search | **no** (§1.3) |
| L3 | `3_scaling_and_search` | What did that capacity buy you? Scaling curves, gating/MoE, search as instrument | yes |
| L4 | `4_language` | Attention, tokens as budget, long documents, fusion | yes (IMDB / SCOTUS) |
| L5 | `5_unsupervised` | Self-supervised — where representations come from | yes |
| L6 | `6_uncertainty` | Modelling, scoring, and deciding with uncertainty | both |
| L7 | `7_generalists_and_memory` | Models you did not train; structuring what you feed them | yes |
| — | `8_extras` | GNNs, swarms/cellular automata, deployment, julia | optional |

---

## L1 — Representation, composition, search

**Status: built and run. No changes this round.**

`01_the_cache`, `01b_tensors`, `01c_build_your_own_cache` (optional), `02_nonlinearity`,
`03_training`, `04_search`, `05_exercises`.

The lesson establishes everything downstream depends on: a published cache, reasoning
over tensor shapes, `σ(Wx+b)` and `σ(W*x)` as one idea (which is how dense and
convolutional layers become a single lesson instead of two), the training loop at the
`vectormesh.training.Trainer` level, and ray.tune with a single algorithm used properly.

Excludes on purpose: the search-algorithm zoo (L3), the 2D/3D rungs of the tensor ladder
(L4 — images give one vector per item, so the short ladder is the honest one here).

---

## L2 — Time series

**Status: slimmed from nine notebooks to four, done and run.**

`01_windowing`, `02_basemodels`, `03_decomposition_and_nbeats`, `04_exercise`.

Moved out:

- `05_gates_highways_moe` **and** `06_search` → **L3**. Gating is general-purpose
  composition, not a time-series trick, and L3 needs an architecture family to search
  over. The two move together because `06_search` has eight references to `05` and its
  entire premise is "the same tool on the family notebook 05 measured" — splitting them
  would force L2 to re-introduce gating and duplicate L3.
- `04_probabilistic` → **L6**. Quantile heads, CRPS and calibration are a competence in
  their own right, not a time-series footnote.
- `08_foundation_models` → **L7**. It is the time-series instance of "a generalist you did
  not train", and it keeps L7 from being LLM-only. Its §8.5 rule — *does the encoder
  compress or expand?* — is the sharpest test of L1's thesis in the course and must
  survive the move intact.

Cut: `09_composition_extra` — it exists to keep ETTh1 alive after ETTh1 was dropped from
the graded sequence. That is staging, not teaching (`cuts.md`). `ts_features.py` went
with it; `ett.py` became `windowing.py`, holding the two generic helpers every windowing
notebook imports.

Datasets: `sunspot` and `solar_10_minutes`. `australian_electricity_demand` follows the
gating and probabilistic notebooks into L3 and L6.

L2 no longer runs a ray search of its own. That is deliberate rather than a loss: L1
searches images, L3 searches this lesson's own composition family on `solar_10_minutes`,
so students still exercise the habit twice in two modalities — which is the whole reason
the second search exists.

---

## L3 — What did that capacity buy you?

**Status: rebuild. Most stale folder in the repo.**

The old lesson was "here is ray". Ray now lands in L1 §04 and L2's search notebook, so
that lesson has no subject. The subject it *should* have is the question every measurement
in `evidence.md` keeps answering the same way: **you added parameters — show that you got
something for them.**

That is scaling laws, made empirical. The course can do what almost none can: a training
run costs seconds, so students *draw* the capacity/return curve on their own data instead
of reading about it.

Target content:

1. **The control question.** Any architecture claim measured against a capacity-matched
   control. Gate vs `deeper` at equal parameters (`evidence.md` §7) is the worked example,
   including the three-seed result that was wrong and the five-seed check that caught it.
2. **The three axes, drawn.** Accuracy vs parameters, vs training rows, vs compute per
   trial. Ingredients already measured: the DTD data-size curve (§3), the sparse/dense
   scaling table (§8), *params trained vs params used* (§9).
3. **Search as the instrument, not the topic.** What does an algorithm know that random
   does not — grid → random → TPE (brief, they have used it twice) → **population methods
   (PSO/PBT, the family that keeps a population, which TPE does not)** → multi-fidelity
   schedulers. Each motivated by a failure of the previous.
4. **The budget question.** Fixed compute: 50 trials × 10 epochs or 10 × 50? Ask it before
   showing ASHA as the answer.
5. **Multi-objective.** Accuracy is not the only axis. A Pareto front over accuracy /
   parameters / latency, which sparse-vs-dense MoE supplies directly.
6. **Reading a flat result as information.**

The search space is a **vectormesh** composition (mechanism, hidden, n_experts,
expert_hidden, lr), not a hand-rolled GRU. `05_gates_highways_moe` and `06_search`
arrive from L2 carrying that space already built and measured; they are the starting
point, to be renumbered into the rebuilt sequence rather than rewritten from scratch.

Cut: `03_minimal_transfer_learning`, `04_transfer_learning_with_resnet` — transfer
learning *is* L1's premise. Drop `mads_datasets` and the legacy `tune.run` API; L3 is
their last consumer, which clears `pyproject.toml`.

---

## L4 — Language

**Status: stripped from ten notebooks to four, done and run** (4 passed, 37 s).

L4 is where the full tensor ladder earns its keep. L1 teaches the short image ladder —
`(D) → (B, D) → (B, K)` — because images give one vector per item. Text does not: a long
document overflows a 512-token window and becomes a `(chunks, dim)` matrix, which is where
padding, masking and aggregation become decisions rather than vocabulary.

`01_attention` (`01_reweighing` + `02_transformer` merged — attention derived from
proximity, then positional encoding as the deliberately separate ingredient that puts
distance back), `02_tokenization` (BPE, reframed as *tokens are your budget*),
`03_long_documents` (the spine — padding, masking, aggregation, a real search),
`04_composition` (two representations fused, with the ablation that keeps you honest).

The merge also removed a broken `<img src=transformers.pbm>` whose file was already
missing from the repo; the encoder block is described in prose instead.

Cut: `03_Attention_gestures`, `05_imdb_preprocessing`, `06_imdb_attention` (all
`mads_datasets` + `mltrainer.rnn_models`, all the old altitude), `08_excercise`, and
`07_metric` — L1 `03_training` already separates loss from metric; the one non-redundant
idea in it, thresholding under cost asymmetry, moves to L6.

Excludes on purpose: generative models. L4 is about *representation* of language. The
generalist alternative is L7.

---

## L5 — Self-supervised

**Status: `src/` deleted; conceptual opener still to build.**

L1 hands students a representation and shows it dominates every other choice. The obvious
next question — where did it come from, and could I make one for a domain that has none? —
is this lesson.

Add `01_why_pretext_tasks_work` ahead of the existing three. The gap today is that the
lesson starts at contrastive SSL with no answer to *why* a pretext task yields a useful
space. Build three spaces on one cache and probe all three linearly:

| model | built from | organises the space by |
|---|---|---|
| autoencoder | `Serial(NeuralNet([d, 64, 32]), NeuralNet([32, 64, d]))` | input similarity |
| contrastive | `Siamese` | invariance to the augmentation you chose |
| supervised | `NeuralNet(d, k)` | the labels, which you do not have |

The linear probe is the instrument, and the punchline is that the *choice of pretext task
is the choice of what the space forgets*. This costs seconds and removes the last hand-
rolled module directory in the graded course.

`5_unsupervised/src/` is gone (`autoencoder.py`, `vae_embeddings.py`, `show_vae.py`,
`query_embeddings.py`, `settings.py`) — verified unused by any notebook first — and the
`Makefile` with it, since every target ran one of those scripts. `02_patch_tokens` and
`03_ijepa` stand as they are, to be renumbered when the opener lands.

**Candidate closing notebook: "what you cannot learn from the shape of an embedding
cloud."** If the probe is the instrument this lesson leans on, the obvious student
question is whether you can skip it and read quality off the geometry instead. Measured
on the published caches, you cannot, and the failure is crisp enough to teach: RankMe
ranks the three encoders *backwards* on all three datasets, pure Gaussian noise scores
higher than every real encoder, and whitening drives RankMe, stable rank and mean pairwise
cosine all to their theoretical optima while the probe does not move to four decimals.
The invariance argument is the point — a linear probe is invariant to invertible linear
maps and these metrics are not, so they cannot be measuring what the probe measures.
Full numbers and an 8-cell proposal in `research/embedding_space_measurement.md`.
**Re-run before promoting any of it to `evidence.md`.**

---

## L6 — Uncertainty: modelling it, scoring it, deciding with it

**Status: new lesson, assembled mostly from material that was mis-filed elsewhere.**

A point prediction is an answer to a question nobody asked. This lesson is the competence
of saying how sure you are, scoring that claim, and then acting on it — and it spans both
halves of the course, because the same instruments work on a forecast distribution and on
a stochastic generator.

1. **Forecast uncertainty** — L2's `04_probabilistic`, intact: CRPS, Gaussian head with
   NLL, quantile head, calibration, Autoformer/PatchTST.
2. **Distribution-free intervals** — conformal prediction. Coverage guarantees without a
   distributional assumption, which is what makes it usable.
3. **Sampling as a distribution** — the bridge to L7. Sampling *k* completions from a
   generator is a Monte Carlo estimate, so self-consistency is an ensemble, and "is this
   model's confidence calibrated?" is the same question as §4.5's Gaussian head, scored
   with the same tools. Runs off a published generation cache (*k* samples precomputed
   once), so it costs a student nothing.
   **State the distinction rather than blurring it:** sampling temperature is randomness
   by construction; forecast spread is uncertainty about the world. Same instruments,
   different objects.
4. **Deciding** — cost-asymmetric decisions, thresholding under class imbalance (from
   L4's `07_metric`), the newsvendor problem. What you *do* with an interval, and
   abstention as a legitimate output.

---

## L7 — Generalists you did not train, and the memory you give them

**Status: new lesson. Build last — see the risk note.**

The arc: you cannot train it, so every bit of leverage is in what you feed it.

1. **The generalist, measured.** Zero-shot versus your own trained head, on cost as well
   as accuracy, in two modalities: TimesFM on time series (L2's `08_foundation_models`,
   including its contamination check and the *compress or expand* rule) and an LLM on the
   L4 classification task. The honest finding L2 already produced — the foundation model's
   frozen embeddings score worse than the raw window — is the anchor.
2. **Retrieval over your own cache.** A `VectorCache` *is* a retrieval index; RAG is L1
   infrastructure re-used, not new machinery. Then why flat RAG fails: chunk soup, no
   relations, no time, no provenance.
3. **Structure as the fix.** Entities and relations extracted into a graph, community
   detection over it to find topics (`6_graphs/02_clusters_in_graphs` earns its keep
   here), hybrid retrieval and multi-hop traversal. Graphs return as **infrastructure with
   a job**, which is a better argument for them than GNN theory on Cora was.

   `research/neurosymbolic_and_structure.md` supplies the two measurements this section
   needs, and they pair into one honest arc rather than a sales pitch. **The ceiling:** on
   a two-hop corpus where the answer document shares no content word with the query,
   cosine recall@20 is 0.110 at 133 documents and **0.000 at 720** — it gets *worse* with
   corpus size, because this is an information-theoretic limit rather than a quality gap,
   so no encoder can fix it. Traversal reads two documents and gets 100%. **The tax:**
   traversal accuracy under per-edge extraction error is `(1-p)^h`, giving an explicit
   break-even of p ≈ 0.45 at two hops and 0.26 at four — while published zero-shot
   relation-extraction F1 sits at 0.15–0.42, i.e. p ≈ 0.6–0.85. A three-hop query over an
   LLM-built graph therefore answers roughly 6% of the time. Both halves are student-
   measurable in seconds, and the second is the number nobody publishes inside a GraphRAG
   demo.

   Two more results worth a cell each: retrieval **inverts** under negation — on "A but
   not B" queries, precision@10 on the ¬B half is 0.000 against a 0.626 base rate, so
   writing "not B" returns B every time when ignoring the query would have been right 63%
   of the time (externally corroborated by NevIR, EACL 2024) — and hierarchy direction is
   recoverable from frozen vectors by a head, but that head does not transfer across
   domains (0.88 within mammals, 0.517 — chance — on musical instruments), which
   reproduces Levy et al.'s lexical-memorisation critique in the course's own idiom.
4. **Memory over time.** What to keep, what to summarise, what to forget — measured
   against flat RAG on a task you can actually score.
5. **Distillation** — the generalist's outputs as labels, a cheap head trained on them,
   and the size of the gap you got for nothing. This closes the course's argument.

**Model access:** a published generation cache is the spine, so the whole lesson runs on
CPU with deterministic results. An optional, clearly marked path regenerates it with a
local ~3B-class model via ollama for students with the RAM.

**Risk, stated up front:** this is the lesson with the most new material and the least
measured evidence, and every other lesson in the course rests on numbers. Build it last,
and measure its claims — a lesson that asserts structured memory beats flat RAG without a
number contradicts `principles.md` §2 in the course's own final lesson.

**That risk is now partly retired.** `research/memory_beyond_rag.md` reports four
synthetic retrieval experiments on generated corpora that run in ~1.5 s of CPU and
produce the effects directly, with scripts in `research/memory_experiments/`. Two are
strong enough to anchor notebooks: a staleness task where flat cosine recall@1 goes
0.117 → 0.983 under a recency prior *while recall@5 stays at 0.933* (the standard metric
says the flat system is fine and it is not), and a multi-hop task where flat retrieval
finds the bridge document 100% of the time and the answer document 0%, rising to 0.93
with one-hop entity expansion. Forgetting gets a Pareto table rather than a slogan:
soft forgetting trades "work now" against "work in 2019", while hard deletion puts the
gold document outside the index for 100% of historical questions. Retrieval metrics need
no LLM at student time, so this half of the lesson costs nothing to run.

The same report supplies the lesson's honesty: **HippoRAG's headline does not survive
inspection** — its "up to 20% higher recall" is a max over three datasets (+20.9 on
2Wiki, +2.7 on MuSiQue, −1.6 on HotpotQA), and 2Wiki is template-generated from Wikidata
triples, so an entity-graph retriever is being scored on a corpus whose data-generating
process *is* an entity graph. That is a leakage story students can reproduce with a
dial. GraphRAG's headline is LLM-as-judge win rates on "comprehensiveness" with no ground
truth, and is not falsifiable by a student; it stays prose.

**Re-run everything before promoting any of it to `evidence.md`.**

---

## 8_extras

Not graded, kept because it is good material without a lesson-sized job.

- `graphs/` — moved from `6_graphs`, intact for now. GNN theory on Cora; `torch-geometric`
  still needs moving to an optional dependency group. The *structural* graph material
  students need lives in L7.
- `swarms/` — cellular automata and El Farol as emergence; the population **optimisers**
  move into L3, where they are a search algorithm that knows something TPE does not.
- `deployment/`, `julia_dl/` — unchanged.

---

## Renames

| From | To |
|---|---|
| `3_tuning_networks` | `3_scaling_and_search` |
| `4_attention` | `4_language` |
| `6_graphs` | `8_extras/graphs` |
| — | `6_uncertainty` (new) |
| — | `7_generalists_and_memory` (new) |

`pyproject.toml`'s `[tool.ty.environment] extra-paths` tracks these.
