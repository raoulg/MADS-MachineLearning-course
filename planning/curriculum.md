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

**Status: built and run. This round removes four notebooks.**

Keep: `01_windowing`, `02_basemodels`, `03_decomposition_and_nbeats`, `06_search`,
`07_exercise` (renumber to 01–05).

Move out:

- `05_gates_highways_moe` → **L3**. Gating is general-purpose composition, not a
  time-series trick, and L3 needs an architecture family to search over.
- `04_probabilistic` → **L6**. Quantile heads, CRPS and calibration are a competence in
  their own right, not a time-series footnote.
- `08_foundation_models` → **L7**. It is the time-series instance of "a generalist you did
  not train", and it keeps L7 from being LLM-only. Its §8.5 rule — *does the encoder
  compress or expand?* — is the sharpest test of L1's thesis in the course and must
  survive the move intact.

Cut: `09_composition_extra` — it exists to keep ETTh1 alive after ETTh1 was dropped from
the graded sequence. That is staging, not teaching (`cuts.md`).

Datasets stay at three: `sunspot`, `solar_10_minutes`, `australian_electricity_demand`.

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
expert_hidden, lr), not a hand-rolled GRU — L2's `06_search` space is written to be
reusable as-is and is the starting point.

Cut: `03_minimal_transfer_learning`, `04_transfer_learning_with_resnet` — transfer
learning *is* L1's premise. Drop `mads_datasets` and the legacy `tune.run` API; L3 is
their last consumer, which clears `pyproject.toml`.

---

## L4 — Language

**Status: strip from ten notebooks to four.**

L4 is where the full tensor ladder earns its keep. L1 teaches the short image ladder —
`(D) → (B, D) → (B, K)` — because images give one vector per item. Text does not: a long
document overflows a 512-token window and becomes a `(chunks, dim)` matrix, which is where
padding, masking and aggregation become decisions rather than vocabulary.

Keep, renumbered 01–04:

- `01_reweighing` + `02_transformer` **merged** — attention derived from proximity, then
  positional encoding. The one mechanism worth building by hand before importing it.
- `04_tokenization` — BPE, reframed as *tokens are your budget*. More relevant now than
  when it was written, not less.
- `05_long_documents` — the spine. Padding, masking, aggregation, and a real search.
- `06_composition` — two representations, fused, with the ablation that keeps you honest.

Cut: `03_Attention_gestures`, `05_imdb_preprocessing`, `06_imdb_attention` (all
`mads_datasets` + `mltrainer.rnn_models`, all the old altitude), `08_excercise`, and
`07_metric` — L1 `03_training` already separates loss from metric; the one non-redundant
idea in it, thresholding under cost asymmetry, moves to L6.

Excludes on purpose: generative models. L4 is about *representation* of language. The
generalist alternative is L7.

---

## L5 — Self-supervised

**Status: three good notebooks, plus one conceptual opener, minus `src/`.**

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

Delete `5_unsupervised/src/` (`autoencoder.py`, `vae_embeddings.py`, `show_vae.py`,
`query_embeddings.py`, `settings.py`). `02_patch_tokens` and `03_ijepa` stand as they are.

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

---

## 8_extras

Not graded, kept because it is good material without a lesson-sized job.

- `graphs/` — `01_intro` and `03_gnn`. GNN theory on Cora, with `torch-geometric` moved to
  an optional dependency group. The *structural* graph material students need lives in L7.
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
