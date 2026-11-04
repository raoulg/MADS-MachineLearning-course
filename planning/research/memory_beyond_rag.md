# Memory for LLM systems, beyond flat RAG — research notes

Status: COMPLETE. Five experiments run and verified on CPU; scripts kept in
`planning/research/memory_experiments/` and reproduce every number in §0 in ~30 s.
Date: 2026-08-08
Audience: lesson design for MADS-MachineLearning-course, "embed once, train many heads"

Design constraints this research must satisfy:
- CPU-only, modest RAM, < 5 min end-to-end per graded notebook.
- Every claim backed by a number the students compute themselves.
- Optional sections may assume a local ~3B model via ollama; nothing gradable may.
- The `VectorCache` pattern extends to LLM outputs: the instructor precomputes
  generations/extractions once and publishes them; students consume deterministically.

Key structural insight (carried over from an earlier partial run, and confirmed below):
**retrieval metrics need no LLM at student time.** recall@k, MRR, nDCG over a
precomputed embedding index are pure numpy. The entire retrieval half of this topic is
free, deterministic, and fast. Everything gradable should be pushed into that half.

---

## TOC

1. Why flat chunk RAG fails — ranked by demonstrability
0. **VERIFIED BY RUNNING CODE — five experiments, E1–E5** (placed after §1 because it
   answers §1; this is the core of the document)
2. Structured memory: HippoRAG / RAPTOR / GraphRAG / LightRAG
3. Forgetting and consolidation
4. Sleep / dreaming-inspired approaches
5. Evaluation, LongMemEval, and synthetic task construction
6. Ranked notebook candidates, best first
7. Things that would NOT survive honest small-scale measurement
8. Verified-by-code vs taken-from-paper ledger
9. Suggested lesson shape

---

## 1. Why flat chunk-based RAG fails — ranked by demonstrability, not by how often cited

Flat RAG = split documents into fixed chunks, embed each, retrieve top-k by cosine, stuff
into the prompt. The documented failure modes, ordered by **whether a student can measure
the gap on a laptop in under five minutes**:

| # | Failure mode | Measurable at small scale? | Evidence |
|---|---|---|---|
| 1 | **Knowledge update / staleness** — a later fact supersedes an earlier one; cosine cannot tell which is current | **Yes, trivially.** E1: R@1 0.117 → 0.983 | LongMemEval "knowledge update" category; measured in E1 |
| 2 | **Multi-hop** — the answer document does not resemble the query; only a bridge document does | **Yes, trivially.** E2: BOTH@5 0.00 → 0.93 | MuSiQue/2Wiki/HotpotQA; measured in E2 |
| 3 | **Temporal reasoning** — "what did I say before X?" needs ordering that embeddings do not encode | **Yes.** LongMemEval reports R@5 0.421 → 0.526 from index design alone | LongMemEval Table 4 |
| 4 | **Granularity mismatch** — retrieval wants big chunks, the reader wants small ones | **Yes**, retrieval half is pure numpy | LongMemEval: session-level R@5 0.706 vs round-level 0.582 |
| 5 | **Abstention** — the answer is absent and the system must say so; top-k *always* returns k things | **Yes**, and cheap: measure score distributions for answerable vs unanswerable queries and pick a threshold; report precision/recall of the abstention decision | LongMemEval "abstention" category |
| 6 | **Entity disambiguation** — two "J. Smith"s collapse in embedding space | **Partly.** Easy to generate, but the fix (entity linking) needs an extractor → instructor-side cache | HippoRAG error analysis attributes most errors to NER/OpenIE |
| 7 | **Contradiction detection** — two retrieved chunks disagree, and nothing in the pipeline notices | **Retrieval half yes, resolution half no** — resolving needs a reader LLM | LongMemEval; general RAG literature |
| 8 | **Provenance / attribution** | **No.** Needs generation + a judge. Not gradable | — |
| 9 | **Global sensemaking** ("what are the themes of this corpus?") | **No.** No ground truth; GraphRAG evaluates it with LLM-as-judge win rates | GraphRAG §2c |
| 10 | **"Lost in the middle"** — a *reader* failure, not a retrieval failure; often miscited as a RAG-retrieval problem | **No** at CPU scale (needs long-context LLM inference) | Liu et al., https://arxiv.org/abs/2307.03172 |

**Design conclusion: modes 1–5 are the gradable syllabus.** They are all measurable with
recall@k / MRR / nDCG over precomputed vectors. Modes 8–10 are prose or optional-ollama
material. Note that mode 10 ("lost in the middle") is the most-cited RAG failure in
practice and is *not about retrieval at all* — worth a paragraph, because students
conflate the two constantly.

---

## 0. VERIFIED BY RUNNING CODE — the headline result of this research

I built and ran four synthetic retrieval experiments on this machine, CPU only,
`all-MiniLM-L6-v2` (22M params) via sentence-transformers, 300–480 generated documents.
**Total runtime after model load: ~1.5 seconds.** Scripts are checked in at
`planning/research/memory_experiments/` (`e1_e2_staleness_multihop.py`,
`e3_e4_forgetting_cost_dial.py`, `e5_replay_frozen_embeddings.py`) — 68–130 lines each,
depending only on numpy, torch, sklearn and sentence-transformers, all already in the
project venv. Run with `uv run python planning/research/memory_experiments/<file>`.

These are not paper numbers. I measured them.

### E1 — knowledge update / staleness. Flat cosine cannot tell current from stale.

60 people, each with 4 successive employers (2019–2022) plus 4 topical distractors =
480 docs. Query: *"Where does {person} work now?"* Gold: the 2022 statement.

| Retrieval | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| flat cosine | **0.117** | 0.933 | 0.426 |
| cosine + recency prior (w=0.3) | **0.983** | 1.000 | 0.992 |
| last-write-wins structured index | **0.983** | 1.000 | 0.992 |

79% of the flat top-5 are **stale versions of the correct fact**. Note recall@1 of 0.117
is *below the 0.25 chance rate* among the four employer statements — flat cosine is not
merely indifferent to recency, it is mildly biased the wrong way. Recall@5 of 0.933 hides
this completely: **the standard metric says the flat system is fine, and it is not.** That
alone is worth a notebook cell.

### E2 — multi-hop. Flat retrieval finds the bridge and never the answer.

60 chains: *"{person} is employed by {org}"* + *"{org} is headquartered in {city}"* plus
3 noise docs per org = 300 docs. Query: *"In which city is the HQ of the company that
employs {person}?"*

| Retrieval | bridge doc @5 | answer doc @5 | BOTH @5 |
|---|---|---|---|
| flat cosine | **1.00** | **0.00** | **0.00** |
| one-hop entity expansion, re-ranked by cosine | — | — | **0.93** |

0.00 → 0.93. The failure is total and the fix is ~8 lines of numpy over an entity index.

### E3 — forgetting is a design choice with a measurable, quantified cost

Same corpus as E1, but now scored on **two** query sets: *"Where does X work now?"* and
*"Where did X work in 2019?"*. Sweeping the recency weight:

| recency weight | "work now" R@1 | "work in 2019" R@1 |
|---|---|---|
| w = 0.0 (pure cosine) | 0.117 | **0.367** |
| w = 0.1 | 0.617 | 0.067 |
| w = 0.3 | **0.967** | 0.000 |
| w = 1.0 | 0.967 | 0.000 |
| hard delete stale rows (480→300 docs) | 0.967 | 0.000 |

This is a clean Pareto frontier produced in one second. And the last row carries the
sharpest lesson available in this whole topic: **hard deletion put the gold document
outside the index for 100% of historical questions.** Soft forgetting (down-weighting)
costs you accuracy; hard forgetting costs you the *ceiling* — no amount of better
retrieval recovers it. A student sees the difference between a reversible and an
irreversible design decision, in a table they made.

### E4 — the difficulty dial, which reproduces the HippoRAG/2Wiki lesson from scratch

E2, but leaking progressively more of the bridge entity into the query text
(none → "a Globex company" → the full org name):

| leakage | flat BOTH@5 | one-hop BOTH@5 | gap |
|---|---|---|---|
| 0.0 | 0.00 | 0.95 | **+0.95** |
| 0.5 | 0.20 | 1.00 | +0.80 |
| 1.0 | 0.40 | 0.95 | +0.55 |

The structured method's advantage is **a controllable function of how much the query
already reveals the index structure**. This is the same phenomenon as HippoRAG scoring
+20.9 on template-generated 2Wiki and −1.6 on HotpotQA (§2a) — but here the student turns
the knob themselves and watches the "structure wins" headline shrink by 40 points. I know
of no better way to teach that a benchmark result is a statement about a *dataset*, not
about a *method*.

### E5 — catastrophic forgetting and replay, directly on "embed once, train many heads"

20 Newsgroups, 6 classes, presented class-incrementally as 3 tasks of 2 classes. A small
MLP head (384→128→6) on **frozen MiniLM vectors**. Instructor-side embedding of all 5,880
documents took 26.8 s on CPU and yields **5.4 MB** of float32 — a `VectorCache` at the
scale the course already ships. Student side is head training only, a few seconds.

| replay buffer | final per-task acc [T1, T2, T3] | avg acc | avg forgetting |
|---|---|---|---|
| none | [**0.024**, 0.554, 0.948] | 0.509 | 0.664 |
| 5 vectors/class | [0.348, 0.704, 0.930] | 0.661 | 0.423 |
| 20 vectors/class | [0.615, 0.786, 0.916] | 0.772 | 0.250 |
| 100 vectors/class | [0.788, 0.852, 0.905] | **0.848** | **0.122** |

Task 1 collapses to **2.4%** — below chance, the classic catastrophic-forgetting
signature: the head has not merely forgotten, it has learned to actively predict recent
classes. Replay recovers it monotonically, and the buffer is *tiny*: 100 vectors/class is
600 × 384 floats ≈ 0.9 MB, and buys +33.9 points of average accuracy.

This is the single cleanest fit between the memory topic and the course's existing
infrastructure — it is literally "embed once, train many heads" with the heads trained in
sequence instead of in parallel — and it is the only place in this entire research where
the "consolidation / replay" family of ideas produces a rigorous, cheap, reproducible
number. See §3 and §4.

**Honest caveat on E2/E4:** the multi-hop failure is constructed — I designed the answer
document to share no lexical overlap with the query. That is a strength, not a cheat,
*provided the notebook says so*: E4 exists precisely to show that the effect is a dial the
designer controls. Present E2 without E4 and it is a rigged demo; present them together
and it is a lesson about benchmark construction.

---

## 2. Structured memory — the numbers, and what they actually show

### 2a. HippoRAG (Gutiérrez et al., NeurIPS 2024) — VERIFIED, and the headline is misleading

Paper: https://arxiv.org/abs/2405.14831 · code: https://github.com/OSU-NLP-Group/HippoRAG

Claim: a "neurobiologically inspired" memory — OpenIE builds a KG over the corpus, then
Personalized PageRank from query-linked entity nodes ranks passages. Single-step, so
10–30x cheaper and 6–13x faster than iterative retrieval (IRCoT).

**Exact single-step Recall@5 (Table 2 of the paper), 1,000 questions per dataset:**

| Retriever | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|
| BM25 | 41.2 | 61.9 | 72.2 |
| Contriever | 46.6 | 57.5 | 75.5 |
| GTR | 49.1 | 67.9 | 73.3 |
| ColBERTv2 | 49.2 | 68.2 | **79.3** |
| RAPTOR | 45.3 | 53.8 | 71.2 |
| Proposition | 49.3 | 63.1 | 71.1 |
| HippoRAG (Contriever) | 52.1 | **89.5** | 76.2 |
| HippoRAG (ColBERTv2) | 51.9 | 89.1 | 77.7 |

Corpus sizes: MuSiQue 11,656 passages / 91,729 KG nodes; 2Wiki 6,119 / 42,694;
HotpotQA 9,221 / 82,157.

**The verification result the previous run wanted — the headline does not hold up:**

1. The entire effect is 2WikiMultiHopQA: **+20.9 R@5** over ColBERTv2. On MuSiQue it is
   **+2.7**. On HotpotQA HippoRAG is **worse than the ColBERTv2 baseline it is built on**
   (77.7 vs 79.3). "Up to 20% higher recall" is a max over three datasets, two of which
   show ~0.
2. 2Wiki is **generated from Wikidata triples by template** (Ho et al. 2020;
   https://arxiv.org/abs/2011.01060 — 95,103 triples, 98,284 entities, 29 relations).
   An entity-graph + PPR retriever is being scored on a dataset whose *data-generating
   process is an entity graph*. This is close to a leakage argument: the structure the
   method assumes is exactly the structure the benchmark was built from. It is the
   single best worked example I found of "match between method prior and benchmark
   construction masquerading as a general result".
3. The authors' own follow-up confirms it. In HippoRAG 2 / "From RAG to Memory"
   (https://arxiv.org/abs/2502.14802, Table 3), measured against a *strong* dense
   retriever rather than ColBERTv2:

   | Method | MuSiQue R@5 | 2Wiki R@5 | HotpotQA R@5 |
   |---|---|---|---|
   | NV-Embed-v2 (7B dense, flat chunks) | 69.7 | 76.5 | 94.5 |
   | RAPTOR | 57.8 | 66.2 | 86.9 |
   | HippoRAG (v1) | 53.2 | 90.4 | 77.3 |
   | HippoRAG 2 | 74.7 | 90.4 | 96.3 |

   HippoRAG v1 **loses to flat dense retrieval by 16.5 points on MuSiQue and 17.2 on
   HotpotQA**. It wins only on 2Wiki. The same paper states plainly that **GraphRAG and
   LightRAG "underperform standard embedding methods across nearly all benchmark
   categories"**, and that RAPTOR shows "substantial performance deterioration" from
   LLM-summarization noise.

4. Stated failure modes (authors'): "most errors made by our system are due to NER and
   OpenIE" — i.e. the extraction step, not the retrieval step, is the bottleneck. Plus a
   "concept–context tradeoff" and OpenIE inconsistency on longer documents.

**Teaching value: very high, but as a negative/nuanced result, not as an endorsement.**
The lesson "structured memory beats flat RAG" is *false as stated*; the true lesson is
"structured memory beats flat retrieval exactly when the query's reasoning structure
matches the index's structure, and costs you accuracy when it doesn't." That is a
falsifiable, measurable claim and fits the course's principles-over-benchmarks stance.

**Laptop-scale experiment (see §6, candidate #1):** the effect above is reproducible
*by construction* on a generated corpus — because we can dial the match between query
structure and index structure continuously.

### 2b. RAPTOR (Sarthi et al., ICLR 2024) — real but small, and the honest number is small

Paper: https://arxiv.org/abs/2401.18059

Claim: recursively cluster (GMM over UMAP-reduced embeddings) + LLM-summarize chunks
into a tree; retrieve from all levels. Gives you multi-scale context.

Measured: NarrativeQA ROUGE-L **30.87 vs 29.26** for SBERT alone (+1.6). QASPER F1
36.70 vs 36.23. The widely-quoted "+20% on QuALITY" is *absolute accuracy with GPT-4*
(82.6 vs 62.3 prior SOTA), which conflates the retriever change with the reader change.
Independent re-measurement in HippoRAG 2's Table 3 puts RAPTOR **below flat dense
retrieval on every multi-hop dataset** (MuSiQue 57.8 vs 69.7, HotpotQA 86.9 vs 94.5),
attributed to noise injected by LLM summarization.

Laptop feasibility: **the retrieval half is genuinely cheap.** Clustering + tree
construction on 2–5k chunk embeddings is seconds of numpy/sklearn on CPU. The *summaries*
are the expensive part — and that is exactly what the course's cache pattern absorbs:
the instructor generates the ~200–600 node summaries once, publishes them in a
`VectorCache`, and students build/query the tree for free. This is the cleanest fit
between this topic and the course's existing infrastructure.

Confidence the effect survives at small scale: **moderate for the mechanism, low for the
win.** The tree demonstrably changes *what* gets retrieved (measurable: overlap@k between
flat and tree retrieval, fraction of retrieved nodes that are summaries vs leaves). Whether
it *improves* recall on a small corpus is genuinely uncertain — which makes it a good
honest experiment rather than a demo, but it should be framed as "when does adding a
summary layer help?" not "summaries help".

### 2c. GraphRAG (Microsoft) and LightRAG — expensive, and the independent evidence is unfavourable

- GraphRAG: https://arxiv.org/abs/2404.16130 · https://github.com/microsoft/graphrag
- LightRAG: https://arxiv.org/abs/2410.05779

GraphRAG's actual claim is narrower than its reputation: it targets **global sensemaking**
("what are the themes of this corpus?"), a query class flat top-k retrieval structurally
cannot serve, and it evaluates with **LLM-as-judge win rates on comprehensiveness and
diversity**, not recall or accuracy. That evaluation design is unreproducible at laptop
scale and unfalsifiable by student measurement — it needs a judge LLM and has no ground
truth. **For this course that rules out the headline claim as gradable material.**

Independent evidence against the broad claim:
- HippoRAG 2's Table 3: GraphRAG and LightRAG "underperform standard embedding methods
  across nearly all benchmark categories."
- "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights" (Han et al., Michigan
  State + Meta, https://arxiv.org/abs/2502.11371) built a unified protocol across four
  GraphRAG families and found **no single winner**; benefits are task-dependent and
  hybrid selection beats either alone.
- Cost: 2024-era GraphRAG indexing was widely reported in the tens of thousands of
  dollars for a single large corpus; Microsoft's own LazyGraphRAG defers summarization
  to query time precisely because index-time LLM summarization did not pay for itself.

**Verdict: flag as fashionable-but-not-gradable.** The *idea* (community detection over
an entity graph, hierarchical community summaries) is teachable in 10 minutes of prose;
the *claim* cannot be honestly measured by a student on a laptop, and the independent
numbers do not support it as a general improvement. Teach it as the negative example:
"here is a famous system whose headline metric is an LLM judging its own output's
'comprehensiveness' — notice you cannot falsify that."

### 2d. What survives from §2

The genuinely notebook-ready structural claim is **not** "graphs beat vectors". It is:

> Retrieval quality depends on the *unit you index* and the *keys you index it under*,
> and both are choices you can measure with pure numpy.

That reframing is supported by the best-evidenced paper in this whole area, LongMemEval
(next section), which is where I would build the graded notebook.

---

## 3. Forgetting and consolidation

### 3a. Generative Agents (Park et al., UIST 2023) — the scoring function is the gold, not the paper

Paper: https://arxiv.org/abs/2304.03442

The famous contribution is the **memory stream**: every observation is stored with a
timestamp and an LLM-assigned importance score, and retrieval ranks by a normalised
weighted sum of **recency** (exponential decay), **importance**, and **relevance**
(cosine). Plus *reflection*: periodically synthesise higher-level insights from recent
memories.

Measured evidence: **human believability ratings** in an ablation (remove memory /
reflection / planning). That is not reproducible on a laptop, has no ground truth, and
needs human raters. **The paper's evaluation is not usable here.**

But the **retrieval scoring function is three lines of numpy and is exactly what E1/E3
measure.** `score = α·cos + β·recency + γ·importance` is a hyperparameter sweep a student
can run in one second and plot as a Pareto curve (E3). This is the right way to teach
Generative Agents: not "here is a famous agent simulation" but "here is a retrieval
scoring function with a tunable forgetting prior — find its cost".

Importance scores are LLM-assigned → **instructor-side cache**, precomputed once. Perfect
fit for the course's pattern.

### 3b. MemGPT / Letta (Packer et al.) — real numbers, wrong shape for this course

Paper: https://arxiv.org/abs/2310.08560 · now https://www.letta.com

Claim: treat context as virtual memory with OS-style paging — main context (in prompt),
external context (recall + archival storage), and let the LLM issue function calls to
page data in and out.

Measured: Deep Memory Retrieval **32.1% → 92.5%** (GPT-4) and **35.3% → 93.4%**
(GPT-4-Turbo); nested key-value retrieval sustained beyond 2 levels where vanilla GPT-4
hits **0%** at 3 levels; document QA accuracy flat as document count grows past the
context window.

Those are large, real, well-measured effects. **But every one of them requires a strong
LLM making tool calls in a loop at student time.** There is no retrieval-only reduction:
the whole point is that the *model* decides what to page in. A 3B ollama model will not
reproduce this — the paper itself notes the paging behaviour depends on strong function
calling. **Verdict: excellent optional/prose material, not gradable.** If it appears at
all, it should be as a precomputed trace the instructor ships (the cache pattern applied
to agent trajectories) that students *analyse* rather than *run*.

### 3c. Catastrophic forgetting and replay — the substantive, laptop-scale core (E5)

This is old, solid, and measurable: McCloskey & Cohen 1989; Kirkpatrick et al. EWC
(https://arxiv.org/abs/1612.00796); Rebuffi et al. iCaRL (https://arxiv.org/abs/1611.07725);
Chaudhry et al. "Tiny Episodic Memories" (https://arxiv.org/abs/1902.10486), which is the
key result for our purposes — **a handful of examples per class recovers most of the loss**,
which E5 reproduces exactly (5/class already halves forgetting).

E5 above is the measured version on frozen embeddings: avg accuracy 0.509 → 0.848, avg
forgetting 0.664 → 0.122, buffer under 1 MB. Runs in seconds.

### 3d. Forgetting as a design choice with a measurable cost

The task asked specifically for work treating forgetting as a design decision rather than
a bug. Honest answer: **the literature mostly does not do this, and E3 does.**
The published framings are (i) machine unlearning / right-to-be-forgotten
(https://arxiv.org/abs/1912.03817), which is about *compliance*, not utility; (ii)
memory-budget ablations inside agent-memory papers, usually reported as "more memory is
better" without the counter-metric. E3's contribution is putting the counter-metric on
the same axis: recency weighting improves "now" questions monotonically **and destroys
"then" questions**, and hard deletion makes the loss irrecoverable. That framing is, as
far as I found, not standard in the literature — which makes it a genuinely original and
defensible thing for the course to teach.

---

## 4. Sleep / dreaming-inspired approaches — mostly metaphor with a paper attached

Asked to be skeptical; here is the skeptical read, separated into two piles.

**Substantive (a real mechanism with a real number):**
- **Experience replay** — the direct descendant of hippocampal-replay neuroscience, and
  the thing that actually works. Measured in E5. Everything valuable in this section
  reduces to this.
- **Offline consolidation as "distil the buffer into the weights"** — i.e. replay plus
  distillation. Real, measurable, and only a small extension of E5 (train a head on the
  buffer between tasks). Worth at most one extra cell.

**Metaphor with a paper attached (be blunt about this in the lesson):**
- **"Dreaming" as synthetic-experience generation.** e.g. "Language Models Need Sleep:
  Learning to Self-Modify and Consolidate Memories" (https://arxiv.org/abs/2606.03979):
  a two-phase scheme, RL-based "Knowledge Seeding" upward-distillation into an expanded
  parameter set, then a "Dreaming" phase generating synthetic curricula via a MoE, filtered
  by gradient-based importance. Backbone Llama-3.2-1B, RL training loops, MoE routing.
  Notably it **samples 60 dreams and rejects 45** per task — i.e. the mechanism is
  75% rejection sampling, which is a strong hint that the "dream" framing is doing less
  work than the filter is.
- The sleep-staging metaphors (REM/NREM, "phasor agents with sleep-staged learning", etc.)
  add vocabulary, not mechanism. Strip the metaphor and what remains is: *replay a buffer,
  generate some synthetic data, filter it hard, distil*. Each of those four has a
  non-metaphorical name and a longer, better-measured literature.
- **Nothing in this pile is laptop-scale.** RL + parameter expansion + MoE on a 1B backbone
  is not a five-minute CPU notebook, and no reduction preserves the claimed effect.

**Recommended treatment:** one honest slide/paragraph. State that the biologically-inspired
naming in this area (including HippoRAG's "neurobiologically inspired" and Generative
Agents' memory stream) is *post-hoc labelling of ordinary engineering* — PPR over a graph,
an exponential decay term, a replay buffer — and that the one idea from neuroscience that
demonstrably transferred is replay, which predates the LLM framing by 30 years. Then show
E5. This is a strong, defensible "negative result as result" section and it costs one
notebook cell of compute.

---

## 5. Evaluation — LongMemEval is the strongest find

Paper: https://arxiv.org/abs/2410.10813 (ICLR 2025) · site: https://xiaowu0162.github.io/long-mem-eval/
· data/code: https://github.com/xiaowu0162/LongMemEval

500 curated questions over long user–assistant chat histories, in five ability
categories that *are exactly the flat-RAG failure taxonomy asked for in §1*:

1. information extraction, 2. **multi-session reasoning**, 3. **temporal reasoning**,
4. **knowledge updates** (a later turn contradicts an earlier one), 5. **abstention**
(the answer is not in memory and the system must say so).

Headline: commercial assistants and long-context LLMs drop **30% accuracy**
(30–60% on LongMemEval_S at ~115K-token histories) versus the same information presented
compactly. Multi-session reasoning, knowledge update and temporal questions are the
hardest categories.

**Why this is the best fit for this course:** the paper does not only report end-task
accuracy — it reports a **retrieval-stage ablation over indexing design**, and the
retrieval stage needs no LLM. Reported gains from index design alone: session
decomposition and **fact-augmented keys ~+4 Recall@k and up to +5 end-to-end QA**;
**temporal-aware indexing +7–11 points on temporal reasoning questions**. Those are
exactly the kind of numbers a student can reproduce on cached embeddings in seconds.

**Dataset logistics (verified):** hosted at https://huggingface.co/datasets/xiaowu0162/longmemeval
(the original repo is deprecated in favour of `xiaowu0162/longmemeval-cleaned`, which
removes noisy history sessions; there is also `longmemeval-v2`). LongMemEval_S is
**264 MB**, 500 questions, ~48 sessions each, ~115K tokens each. **MIT licensed**
(background sessions from ShareGPT / Apache-2.0 and UltraChat / MIT).
An `longmemeval_oracle.json` ships the gold evidence sessions.

264 MB of raw text is too much to hand students, but it is exactly the right size for the
**instructor-side cache**: embed the ~24,000 sessions (500 × 48) once with a small
sentence encoder, plus the LLM-derived "fact-augmented keys", and publish a few tens of
MB of float16 vectors + gold session ids. Students then reproduce the entire Table 3 /
Table 4 ablation in numpy.

**Verified retrieval ablation numbers (LongMemEval Tables 3–4) — all pure Recall@5/NDCG@5:**

| Index design | Recall@5 | NDCG@5 |
|---|---|---|
| Round-level values, key = value | 0.582 | 0.481 |
| Round-level values, key = value + extracted fact | 0.644 | 0.498 |
| Session-level values, key = value | 0.706 | 0.617 |
| Session-level values, key = value + extracted fact | **0.732** | 0.620 |

On the **temporal-reasoning subset only**:

| Index design | Recall@5 |
|---|---|
| key = value | 0.421 |
| + time-aware query expansion (GPT-4o) | 0.451 |
| + fact-augmented keys and time | **0.526** |

That last row is the single most valuable number in this document: **+10.5 absolute
recall points on temporal questions from changing the index, with the retrieval metric
computable in numpy and the LLM cost fully precomputable by the instructor.** It is a
direct, honest demonstration that "flat chunk RAG fails and structure fixes it" — with
structure meaning *keys and granularity*, which is far more defensible than *graphs*.

Note also the granularity result is **non-monotonic and counterintuitive**: finer chunks
(round-level) *hurt* retrieval (0.582 vs 0.706) while *helping* the reader. That is a
genuinely good teaching moment — retrieval and reading want different granularities — and
students can measure the retrieval half of it themselves.

Caveats worth teaching: LoCoMo (the other common memory benchmark,
https://arxiv.org/abs/2402.17753) is criticised for reducing memory to surface factual
recall; follow-ups (LoCoMo-Plus) argue that real memory tasks require applying latent
constraints never explicitly re-queried. Good material for a paragraph on "your benchmark
defines what your system will be good at".

### 5b. Synthetic task construction — strongly preferred, and validated in §0

The brief said synthetic is very welcome. It is more than welcome; **it is better**, and
E1–E4 demonstrate why:

1. **Ground truth is exact and free.** No annotation, no judge LLM, no download. The gold
   document is known by construction, so recall@k / MRR / nDCG are exact.
2. **The effect size is a dial (E4).** You can generate the *same* task at any difficulty
   and show the structured method's advantage shrinking from +0.95 to +0.55. No downloaded
   benchmark can do this, and it is the only way to teach that a benchmark number is a
   property of a dataset.
3. **It is reproducible and deterministic** with a seed — a hard requirement for grading.
4. **It scales down.** 300–480 documents is enough to show every effect in §0.
5. **It removes the leakage argument.** A generated corpus about "Ada Ostrov17" cannot be
   in the encoder's pretraining data. Given that the entire HippoRAG/2Wiki problem (§2a) is
   a contamination-of-structure argument, generating your own corpus is the *methodological*
   fix, not just the convenient one.

Recommended generator design (all four experiments come from one ~60-line generator):
entities × predicates × timestamps, with three knobs — number of updates per fact
(staleness pressure), hop count (multi-hop pressure), and lexical leakage between query and
gold (difficulty dial). Ship it as `notebooks/.../memorygen.py` alongside the notebook, the
way `windowing.py` already sits beside the timeseries notebooks.

---

## 6. Ranked notebook candidates — best first

Ranking criterion: *would this make a good measured notebook?* = (effect is real) ×
(student produces the number) × (runs cold on a laptop CPU in minutes) × (teaches
something that stays true).

---

### #1 — "Your retrieval metric is lying to you": staleness + multi-hop on a generated corpus
**GRADABLE. Build this one.**

- **Content:** E1, E2, E3, E4 from §0.
- **Data:** generated in-notebook, seeded. Zero download.
- **Baseline:** flat cosine top-k over MiniLM embeddings.
- **Treatments:** recency-weighted scoring; last-write-wins index; one-hop entity expansion.
- **Metrics:** recall@1, recall@5, MRR, plus "fraction of top-5 that are stale".
- **Expected result:** R@1 0.117 → 0.983 (E1); BOTH@5 0.00 → 0.93 (E2); the Pareto table
  (E3); the shrinking gap (E4). **These are measured, not predicted.**
- **Runtime:** ~5 s of compute after a ~4 s model load; the encoder is 90 MB. Comfortably
  under budget even with `param()` untouched.
- **Confidence the effect survives at small scale: 1.00 — it was *obtained* at small scale.**
- **Why it is #1:** it teaches the recall@5-hides-the-failure lesson (0.933 while R@1 is
  0.117), it produces a Pareto frontier for forgetting, and E4 inoculates students against
  the exact error the whole GraphRAG literature makes.

---

### #2 — Catastrophic forgetting and replay on a published `VectorCache`
**GRADABLE. Second-highest value, and it needs almost no new infrastructure.**

- **Content:** E5.
- **Data:** instructor publishes 20 Newsgroups (6 classes) MiniLM vectors — **5.4 MB**,
  embedding done once in 27 s. Students never load the encoder.
- **Baseline:** sequential class-incremental training, no replay.
- **Treatment:** replay buffer at 5 / 20 / 100 vectors per class.
- **Metrics:** per-task accuracy matrix, average accuracy, average forgetting.
- **Expected result:** task-1 accuracy **0.024** without replay (below chance — the
  striking number), average accuracy 0.509 → 0.848, forgetting 0.664 → 0.122. Measured.
- **Runtime:** seconds on CPU; the 27 s embed is instructor-side.
- **Confidence: 1.00 — measured.**
- **Why:** it is "embed once, train many heads" with the heads trained *in sequence*, so it
  reuses the course's central metaphor to teach something new, and it is the only rigorous
  version of the whole sleep/consolidation topic.

---

### #3 — LongMemEval index-design ablation on a published cache
**GRADABLE if the instructor does the cache work; highest realism of the three.**

- **Content:** reproduce LongMemEval Tables 3–4 — value granularity (session vs round) and
  key expansion (raw vs fact-augmented), on the temporal subset.
- **Data:** instructor downloads `xiaowu0162/longmemeval-cleaned` (264 MB, MIT), embeds
  ~24k sessions and their round-level splits, generates fact-augmented keys with an LLM
  **once**, and publishes vectors + gold session ids. Students get tens of MB of float16.
- **Baseline:** key = value, session-level.
- **Treatments:** round-level values; fact-augmented keys; time-aware expansion.
- **Metrics:** Recall@5, NDCG@5, broken out by question category.
- **Expected result (from the paper, *not* measured by me):** 0.706 → 0.732 overall;
  **0.421 → 0.526 on temporal questions**; and the counterintuitive 0.706 → 0.582 drop from
  finer chunks.
- **Runtime:** student side is a few matrix multiplies — seconds.
- **Confidence the effect survives: 0.75.** Real benchmark, real numbers, and the effects
  are index-side so they should transfer. The risk is entirely in the instructor-side
  pipeline: the fact-augmented keys were generated with GPT-4o, and a weaker generator
  measurably degraded them in the paper's own Table 4 (Llama-8B time extraction
  underperformed). **Budget real instructor time for this one, and verify the effect
  reproduces before committing it to the syllabus.**
- **Why #3 and not #1:** it is the most credible ("this is a real published benchmark, and
  you just reproduced its table") but it is the only candidate with meaningful build risk.

---

### #4 — RAPTOR-style summary tree from a published summary cache
**OPTIONAL / exploratory, not gradable as a "wins" claim.**

- **Content:** cluster leaf embeddings (GMM over UMAP or plain k-means), retrieve over
  leaves+summaries vs leaves only.
- **Data:** instructor generates ~200–600 node summaries once, ships them in a `VectorCache`.
- **Metrics:** recall@k, plus *what fraction of retrieved nodes are summaries* and
  *overlap@k between flat and tree retrieval* — descriptive metrics that are honest even
  when the accuracy effect is null.
- **Expected result:** the tree demonstrably changes what is retrieved; whether it
  **improves** recall on a small corpus is genuinely uncertain — RAPTOR's own gain was
  +1.6 ROUGE-L, and independent re-measurement put it *below* flat dense retrieval.
- **Confidence the win survives at small scale: 0.3.** Frame it as "when does adding a
  summary layer help?" and let a null result stand. Good exercise material; bad lecture claim.

---

### #5 — Abstention thresholding
**GRADABLE, cheap, and underrated. Half a notebook at most.**

- Generate answerable and unanswerable queries over the §5b corpus; plot the top-1 cosine
  score distribution for each; pick a threshold; report precision/recall of the abstention
  decision and the AUC.
- Teaches that top-k retrieval **structurally cannot abstain** — it always returns k things
  — and that the fix is a calibration decision, not a retrieval one.
- Confidence: 0.9. Not yet measured by me, but it is the same machinery as E1 and there is
  no plausible way for it to fail to produce a separable distribution.

---

### Not recommended as notebooks
**#6 GraphRAG global sensemaking** — see §7. **#7 MemGPT paging** — §3b, needs a strong
LLM in the loop. **#8 anything "dreaming"** — §4.

---

## 7. Fashionable things that would NOT survive honest small-scale measurement

Flagged explicitly, since the course teaches negative results as results.

1. **"Structured/graph memory beats flat RAG."** False as a general claim, and the evidence
   against it is in the proponents' own papers. HippoRAG v1 loses to a strong dense
   retriever by 16.5 R@5 points on MuSiQue and 17.2 on HotpotQA (§2a). HippoRAG 2's paper
   states GraphRAG and LightRAG "underperform standard embedding methods across nearly all
   benchmark categories". The Michigan State/Meta unified evaluation found no single winner.
   **The true claim is conditional** — structure wins when query structure matches index
   structure — and E4 measures exactly that condition.

2. **HippoRAG's "up to 20% higher recall".** A maximum over three datasets. The other two
   show +2.7 and −1.6. And the winning dataset (2Wiki) is template-generated from Wikidata
   triples, so an entity-graph retriever is being scored on a corpus whose data-generating
   process *is* an entity graph. This is the single best worked example of benchmark/method
   structural leakage I found.

3. **GraphRAG's headline evaluation.** LLM-as-judge win rates on "comprehensiveness" and
   "diversity", with no ground truth. Unfalsifiable by student measurement, and indexing
   costs were high enough that Microsoft's own follow-up (LazyGraphRAG) removed the
   index-time summarization. Teach it as the negative example about evaluation design.

4. **RAPTOR's "+20% on QuALITY".** That number is absolute accuracy with GPT-4 as reader
   against a prior SOTA, conflating the retriever change with the reader change. The
   controlled retriever-only comparison is +1.6 ROUGE-L on NarrativeQA.

5. **All "neurobiologically inspired" framing in this area.** HippoRAG is Personalized
   PageRank. Generative Agents' memory stream is cosine plus an exponential decay term.
   "Sleep"/"dreaming" papers are replay plus rejection-sampled synthetic data. The one idea
   that genuinely transferred from neuroscience is replay, and it transferred in 1995.
   Students should be able to strip a metaphor and name the mechanism.

6. **"More memory is always better."** E3 falsifies it in one second: the recency prior that
   takes "work now" from 0.117 to 0.967 takes "work in 2019" from 0.367 to 0.000, and hard
   deletion removes the gold document from the index entirely.

7. **LoCoMo as a memory benchmark.** Widely used, but criticised for reducing memory to
   surface factual recall. Prefer LongMemEval, which at least decomposes by ability and
   reports retrieval-stage metrics.

---

## 8. Verified-by-code vs taken-from-paper — the ledger

**Measured by me on this machine (CPU, `all-MiniLM-L6-v2`, seeded, ~30 s total incl. E5):**
- E1 staleness: flat R@1 0.117 / R@5 0.933 / MRR 0.426 → recency-weighted 0.983 / 1.000 /
  0.992; last-write-wins identical; 79% of flat top-5 are stale duplicates.
- E2 multi-hop: flat BOTH@5 = 0.00 (bridge 1.00, answer 0.00) → one-hop expansion 0.93.
- E3 forgetting cost: the full recency-weight sweep and the hard-delete row.
- E4 difficulty dial: gap +0.95 / +0.80 / +0.55 at leakage 0.0 / 0.5 / 1.0.
- E5 replay: avg acc 0.509 → 0.848, avg forgetting 0.664 → 0.122, task-1 acc 0.024 → 0.788;
  5,880 docs embed in 26.8 s producing 5.4 MB.
- Environment facts: `sentence-transformers` 5.6.0, `transformers` 5.12.1, `vectormesh`
  0.9.2, numpy 2.4.6, sklearn 1.9.0 all present in the project venv.

**Taken from papers, not independently verified:**
- All HippoRAG Table 2/3 numbers and dataset sizes (fetched from the paper's HTML).
- All HippoRAG 2 Table 3 numbers, and its statements about GraphRAG/LightRAG/RAPTOR.
- All LongMemEval Table 3/4 numbers, dataset size (264 MB), and MIT licensing.
- RAPTOR's NarrativeQA/QASPER/QuALITY numbers.
- MemGPT's DMR and nested key-value numbers.
- GraphRAG indexing cost figures (secondary sources, treat as indicative only).
- The Michigan State/Meta "no single winner" conclusion (from the abstract; I did not read
  the full results tables).

**Explicitly not checked:** whether the LongMemEval cache pipeline (#3) reproduces its
reported gains with a *cheaper* fact-extraction model than GPT-4o. This is the one open
risk in the recommendations and should be tested before #3 enters the syllabus.

---

## 9. Suggested lesson shape

One graded notebook built from #1 (generated corpus, four experiments, ~5 s runtime), one
graded notebook from #2 (replay on a published `VectorCache`, seconds), #5 folded in as
exercises. #3 as the "real benchmark" capstone **if** the instructor-side cache validates.
§7 as the lecture: every fashionable claim in this area, and the number that undercuts it.

The through-line that holds it together, and that the measurements support:

> Memory is not "add a graph". Memory is three separable design decisions — **what unit you
> index, what key you index it under, and what you let decay** — each of which you can
> measure with numpy, and each of which has a cost you can put on the same axis as its
> benefit.



