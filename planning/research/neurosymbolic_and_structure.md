# Neurosymbolic AI and structure vs. vector space — research for the final lesson

Status: IN PROGRESS (written incrementally). Each item is tagged:

- **MEASURED HERE** — I ran code on this machine and the number below is the output.
- **FROM PAPER** — taken from a cited source, not reproduced here.
- **ASSERTED** — claim in circulation with no measurement I could find. Treat as suspect.

Constraint filter used throughout: *a claim only counts if a student can produce the
number themselves, CPU-only, < 5 min.* Anything failing that filter is marked
NOT GRADABLE and may only appear as an optional/ollama section or as narrative.

## Log

### F1. Analogy arithmetic is mostly a nearest-neighbour artefact (item 5) — FROM PAPER, reproducible here

Claim: the famous `king - man + woman = queen` offset result is largely produced by the
standard evaluation *excluding* the three input words a, b, c from the candidate set.
Source: Tal Linzen, *Issues in evaluating semantic spaces using word analogies*, RepEval
2016, https://aclanthology.org/W16-2503/ ; corroborated by Rogers et al., *The (too many)
problems of analogical reasoning with word vectors*, *SEM 2017,
https://aclanthology.org/S17-1017.pdf and Nissim et al., *Fair is better than sensational*,
CL 2020, https://direct.mit.edu/coli/article/46/2/487/93368/.

Measured (Linzen 2016, FROM PAPER):
- With the standard exclusion of a/b/c, 3CosAdd scores ~0.71 overall on the Google set.
- Drop the exclusion and accuracy falls to ~0.21 — the model mostly returns **b** itself.
- Accuracy is **0** in 9 of the 15 Google categories once the exclusion is removed.
- A `nearest neighbour of b` baseline (ignore the analogy entirely) gets **70%** on the
  plural-noun category; 3CosAdd adds only ~10 points on top of it.

Why this is a great final-lesson notebook: the entire measurement is
`argmax cosine` over a cached embedding matrix. No LLM, no training, no GPU.
It teaches the exact lesson the course wants — *the vector space does far less structural
work than the demo implies, and the harness was doing the rest.*

Confidence the effect survives at small scale: **very high**. It is not a subtle
statistical effect; it is a property of where the answer vector lands.

### F2. GraphRAG does not uniformly beat vector RAG (item 4) — FROM PAPER

- *When to use Graphs in RAG: A Comprehensive Analysis for Graph RAG*, GraphRAG-Bench,
  arXiv 2506.05690, https://arxiv.org/abs/2506.05690 — abstract states plainly that
  "recent studies report that GraphRAG frequently underperforms vanilla RAG on many
  real-world tasks". Benchmark + code at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- A unified-protocol comparison (identical chunking, embeddings, generation) reports
  vanilla RAG **ahead** on single-hop factual lookup (Natural Questions, F1 64.8 vs 63.0
  for the best graph method) and **behind** on multi-hop (MultiHop-RAG, 70.3 vs 67.0).
  This is the cleanest one-line summary of the whole topic: **graphs buy multi-hop, and
  cost you single-hop.**

Teaching value: this is the honest framing for the lesson — the gain is not "graphs are
better", it is "graphs are better *at the compositional part*, and the compositional part
is a minority of queries in most corpora."

### F3. Negation: retrievers score at or below random (item 2) — FROM PAPER, partly reproducible

- Weller et al., *NevIR: Negation in Neural Information Retrieval*, EACL 2024,
  https://aclanthology.org/2024.eacl-long.139.pdf — 2,556 contrastive document pairs;
  metric is **pairwise accuracy** (must rank both queries correctly, so random = 25%).
  Finding: most IR models perform **at or below random**.
- Reproduction: *Reproducing NevIR: Negation in Neural Information Retrieval*, SIGIR 2025,
  https://dl.acm.org/doi/10.1145/3726302.3730294 — result holds; listwise LLM rerankers
  are the only family that improves, and still below human.
- Taxonomy follow-up: arXiv 2507.22337, https://arxiv.org/pdf/2507.22337

This is the strongest *published* "cosine provably cannot" result, and the NevIR data is
small and public — a student can score a cached bi-encoder on it in seconds.

### F4. Two-hop retrieval: cosine recall collapses to literally zero — **MEASURED HERE**

Script: `scratchpad/multihop.py` (reproduced below in outline). Encoder
`sentence-transformers/all-MiniLM-L6-v2` on CPU. Corpus of atomic facts:

- hop-1 doc: `"<person> is employed by <company>."`
- hop-2 doc: `"<company> has its headquarters in <city>."`
- query: `"In which city does <person> work?"`

The hop-2 document shares **no content word** with the query — the bridge entity
(the company) only appears in the hop-1 document. Answering requires both.

MEASURED (single run, seed 0, whole sweep = **10 seconds wall clock including model load**):

| n_docs | k | recall hop-1 doc | recall hop-2 doc | both retrieved |
|---|---|---|---|---|
| 133 | 1 | 1.000 | 0.000 | 0.000 |
| 133 | 20 | 1.000 | 0.110 | 0.110 |
| 400 | 1 | 0.987 | 0.000 | 0.000 |
| 400 | 20 | 1.000 | 0.007 | 0.007 |
| 720 | 1 | 0.972 | 0.000 | 0.000 |
| 720 | 20 | 1.000 | **0.000** | **0.000** |

Graph traversal on the same corpus (hop-1 doc → parse company → adjacency lookup)
retrieves both documents for **100%** of queries while reading exactly 2 documents.

This is the sharpest artefact in the whole report. It is not "cosine is worse"; it is
**cosine at k=20 out of 720 documents finds the needed document 0% of the time, and the
failure gets *worse* with corpus size** — recall@20 goes 0.110 → 0.007 → 0.000 as the
corpus grows, because the hop-2 document is being retrieved at chance. A student can dial
`n_people` and watch the curve go to zero. The scaling knob is what makes it a lesson
rather than an anecdote.

Caveat to teach honestly: the reason is not that the encoder is bad, it is that the
*information needed to score the hop-2 document is not in the query*. No encoder, however
good, can fix this — it is an information-theoretic ceiling, not a quality gap. That is
the strongest possible version of "structure beats vectors", and it is fully constructive.

Confidence: **certain** — it is measured, deterministic, and the mechanism is provable.

### F5. Is-a direction: cosine is at chance, an *untrained* geometric read-out is at 0.73–0.83, a trained head at 0.88 — **MEASURED HERE**

Script: `scratchpad/hypernym.py`. WordNet subtrees via nltk; phrases embedded with
all-MiniLM-L6-v2; direction task = given a hyponym/hypernym pair in random order, say
which is the parent. Chance = 0.500. Whole experiment: **6.2 seconds on CPU.**

Subtree sizes (direct hyponym edges): mammal 1170, plant 4493, vehicle 538,
musical_instrument 174.

| method | mammal | plant | vehicle | instrument |
|---|---|---|---|---|
| best threshold on **cosine** (symmetric) | 0.503 | 0.505 | 0.532 | 0.552 |
| **untrained**: nearer the taxonomy centroid ⇒ parent | 0.729 | 0.763 | 0.829 | 0.793 |
| logistic head on `[u-v, u*v]`, in-domain random split | 0.880 | 0.903 | 0.903 | 0.862 |
| head trained on **mammals**, tested cross-taxonomy | — | 0.846 | 0.725 | **0.517** |
| head, **node-disjoint** split inside mammals (n_tr=299, n_te=303) | 0.825 | — | — | — |

Four separate teachable results out of one 6-second script:

1. **Cosine is at chance and always will be.** Not an empirical weakness — a symmetric
   function assigns the same score to (x,y) and (y,x), so it cannot express an ordered
   relation. This is the "vector space is bad at asymmetry" claim, proved rather than
   asserted.
2. **The hierarchy is nevertheless in the vectors.** A single dot product against the
   taxonomy centroid — zero training, one line — recovers direction at 0.73–0.83.
   Generality is (partly) a *radial* coordinate.
3. **A tiny head does better still (0.88–0.90), and it is not pure lexical memorisation**
   — the node-disjoint split (no surface word shared between train and test) still gets
   0.825. This is the course's own "train a head on cached vectors" idiom applied to a
   relation rather than a class.
4. **But it does not generalise across domains.** The mammal-trained head is at
   **0.517 — chance — on musical instruments.** That is the Levy et al. 2015
   lexical-memorisation critique (*Do supervised distributional methods really learn
   lexical inference relations?*, NAACL 2015, https://aclanthology.org/N15-1098/)
   reproduced in a graduate notebook in six seconds. The head learned "what mammal
   hierarchies look like", not "hypernymy".

Result 4 is the one that earns the lesson its place: the honest conclusion is *the vector
space encodes some hierarchy, a head can read part of it, and the reading does not
transfer.* Confidence: **high** for 1 (provable) and 3–4 (measured, large n);
**medium** for 2 pending the confound check in F6.

### F6a. The centroid heuristic is not a string-length artefact — **MEASURED HERE**

Obvious objection to F5 result 2: WordNet leaf lemmas are long multi-word phrases
("Tibetan mastiff") and roots are short ("dog"), so "closer to the centroid" might just be
"shorter string". Checked directly (`scratchpad/negation.py`, part a):

| taxonomy | centroid rule | `shorter string` rule | `fewer tokens` rule | centroid, **restricted to token-count-tied pairs** |
|---|---|---|---|---|
| mammal | 0.729 | 0.626 | 0.603 | 0.694 (n=536) |
| plant | 0.763 | 0.829 | 0.749 | 0.788 (n=1690) |
| vehicle | 0.829 | 0.597 | 0.524 | 0.881 (n=270) |
| musical_instrument | 0.793 | **0.414** | **0.368** | 0.792 (n=77) |

The surface baseline is *below chance* on instruments while the centroid rule holds at
0.79, and on every taxonomy the centroid rule survives restriction to pairs with equal
token count. So the radial-generality signal is real, not a spelling artefact. Note plant
is the one case where the length baseline is competitive — worth showing students that the
confound check sometimes fires.

### F6b. Negation: cosine precision is **exactly 0.000** against a base rate of 0.255 — **MEASURED HERE**

Script `scratchpad/negation.py`, part b. 800 synthetic product docs, each carrying 2–4
attributes from a vocabulary of 8, e.g. `"The backpack is waterproof, foldable."`
56 queries of the form `"a product that is A but not B"`. Structured attribute metadata is
kept alongside, so the symbolic filter is exact.

MEASURED, k=10, all-MiniLM-L6-v2:

| measurement | value | base rate |
|---|---|---|
| precision@10, full constraint (A ∧ ¬B) | **0.000** | 0.255 |
| precision@10, positive half (A) | 0.998 | — |
| precision@10, negative half (¬B) | **0.000** | 0.626 |
| symbolic filter over the same metadata | 1.000 | — |

Read the third row carefully: writing "not B" in the query makes the retriever return a
document containing B **100% of the time**, where ignoring the query entirely would have
got it right 62.6% of the time. **Negation does not merely fail to help — it inverts.**
The retriever handles the positive half essentially perfectly (0.998), so this is a
targeted failure of one operator, not general incompetence.

Runtime for parts (a)+(b) together: **7.1 seconds** on CPU.

Honest caveat to state in the notebook: this is a constructed worst case, and the
mechanism is partly lexical (the token B is in both query and document). The naturalistic
confirmation is NevIR (F3), which finds the same conclusion on human-written contrastive
pairs. Pair them: the synthetic gives the clean zero and the scaling knob, NevIR gives
external validity. Confidence: **certain** at small scale; **high** that the direction
(not the exact 0.000) holds on real corpora, on NevIR's evidence.

### F7. Extraction error compounds as `(1-p)^h` — the graph's advantage has an explicit break-even — **MEASURED HERE**

Script `scratchpad/noisy_graph.py`, pure numpy, **0.03 seconds**. Gold chains of `h` edges;
the extractor gets each edge wrong with probability `p` and, when wrong, points confidently
at the wrong tail entity (the realistic LLM failure mode, not a missing edge). Traversal
answers `entity_h` given `entity_0`.

MEASURED answer accuracy (400 trials/cell; matches closed form `(1-p)^h` to sampling noise):

| hops | p=0.00 | p=0.05 | p=0.10 | p=0.20 | p=0.30 | p=0.50 |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.953 | 0.915 | 0.805 | 0.728 | 0.542 |
| 2 | 1.000 | 0.930 | 0.843 | 0.630 | 0.480 | 0.250 |
| 3 | 1.000 | 0.838 | 0.698 | 0.507 | 0.273 | 0.113 |
| 4 | 1.000 | 0.812 | 0.618 | 0.375 | 0.217 | 0.050 |

Break-even: for the graph to beat a vector baseline of 0.30, the per-edge error rate must
stay under **0.452 at h=2, 0.331 at h=3, 0.260 at h=4**.

Now put F7 next to the *published* extraction error rates in F8: zero-shot LLM relation
extraction lands around **F1 0.15–0.42** on standard benchmarks, i.e. `p` between 0.58 and
0.85. Substitute: a 3-hop query over an LLM-extracted graph with p=0.6 answers correctly
**6% of the time**. The graph's structural advantage is real (F4: it beats 0.000) but it is
being spent on extraction noise long before it reaches the user.

This is the single most useful *composite* result in the report, and both halves are
measurable by students: F4 gives the ceiling, F7 gives the tax, F8 gives the tax rate.

### F8. LLM knowledge-graph extraction accuracy (item 3) — FROM PAPER

The honest picture, and it is much worse than the GraphRAG marketing implies:

- Zero-shot relation extraction with frontier models: GPT-4 scores **31.03** on DuIE2.0
  (ChatGPT 10.3); a 2026 benchmark reports GPT-5-mini at **42.4% F1 on CoNLL04, 18.6% on
  DocRED, 15.0% on FewRel**. Survey: *LLMs for Knowledge Graph Construction and Reasoning*,
  arXiv 2305.13168, https://arxiv.org/pdf/2305.13168 ; *Aligning Instruction Tasks Unlocks
  LLMs as Zero-Shot Relation Extractors*, https://arxiv.org/pdf/2305.11159
- LLMs still do **not** beat fully supervised small models on relation extraction, and the
  gap *widens with document complexity*: *LLMs Underperform Graph-Based Parsers on
  Supervised Relation Extraction for Complex Graphs*, arXiv 2604.08752,
  https://arxiv.org/abs/2604.08752 — "the graph-based parser increasingly outperforms the
  LLMs as the number of relations in the input document increases". (I could not retrieve
  the per-dataset F1 table from the abstract page; treat the direction as established and
  the magnitudes as unverified.)
- Domain pipeline evaluations report triple-extraction F1 around **0.73** when the ontology
  is fixed and the domain is narrow (*Performance evaluation of LLMs for automated knowledge
  graph generation*, https://www.sciencedirect.com/science/article/pii/S2666827026000885).
  So `p` ranges roughly 0.27 (narrow, ontology-constrained) to 0.85 (open, multi-relation).

**Nobody publishes precision for the graph inside a GraphRAG demo.** That absence is itself
the teaching point: the pipeline is evaluated end-to-end on answer quality, so a bad graph
is never separately visible. A notebook that extracts a graph *and then scores the graph*
would be an unusually honest artefact.

### F9. Neurosymbolic AI, blunt per-item verdict (item 1) — FROM PAPER

**(a) LLM writes a program / query, a real engine executes it — REAL, and the strongest
item in the whole neurosymbolic family.**
- Program-of-Thought (Chen et al., TMLR 2023, https://arxiv.org/abs/2211.12588): ~12 point
  average gain over chain-of-thought on math word problems, because the arithmetic is
  delegated to Python. Reproduced widely; this is the mechanism behind every "code
  interpreter" feature shipped since.
- Text-to-SQL: Spider 1.0 execution accuracy ~**86–91%** (DIN-SQL / DAIL-SQL / GPT-4 era).
  BIRD (real databases, dirty schemas): top systems ~**75.6%** (XiYan-SQL), best reported
  single model ~**80.0%**, against a **92.96%** human baseline — https://bird-bench.github.io/.
  Spider 2.0 (enterprise workflows, ICLR 2025, https://spider2-sql.github.io/): **~6%**.
- Verdict: **the 86% → 6% collapse from Spider 1.0 to Spider 2.0 is the number to teach.**
  Program generation is a genuine capability on toy schemas and largely aspirational on
  real ones. Same story for text-to-Cypher/SPARQL, worse, with less data.

**(b) Constrained / grammar-guided decoding — REAL as a *validity* guarantee, CONTESTED as a
*quality* improvement.** Grammar-constrained decoding makes malformed output impossible by
construction — that part is a theorem, not a result. But *Let Me Speak Freely?* (Tam et al.,
EMNLP 2024 Industry, https://aclanthology.org/2024.emnlp-industry.91/, arXiv 2408.02442)
measures a **significant decline in reasoning accuracy under format restriction, worsening
with stricter constraints**. The `.txt`/Outlines camp disputes the methodology (prompt
quality confound). Verdict: **use it to guarantee parseability, do not claim it makes the
model smarter.** Excellent "fashionable claim that does not survive" material.

**(c) Logical / symbolic verification of LLM output — REAL and cheap.** Not a research
direction: parse the generated SQL, type-check it against the schema, execute it, check the
JSON against the schema, run the extracted triples against ontology constraints. Every one
of those is a deterministic checker producing a number. This is the part of neurosymbolic AI
that is *boring and works*, and it is exactly what fits the course's cached-generation model.

**(d) Differentiable logic (DeepProbLog, LTN, Scallop, neural theorem provers) —
RESEARCH DIRECTION WITH A DEMO.** Fifteen years of MNIST-addition-scale results.
Verdict: **do not build a graded notebook on it.** Mention in one paragraph as the
"beautiful idea that has not scaled", and move on. Anything claiming otherwise at
laptop scale is a toy dressed as a result.

### F10. Hybrid retrieval, measured (item 4) — FROM PAPER

- **BM25 beats dense out-of-domain.** BEIR (Thakur et al., NeurIPS 2021 D&B,
  https://arxiv.org/abs/2104.08663): dense retrievers trained on MS MARCO frequently fail
  to beat BM25 zero-shot across the 18 datasets. This is *still* the most under-taught
  result in applied retrieval and it is fully reproducible at laptop scale on one or two
  small BEIR sets.
- **RRF fusion of BM25 + dense is a real but modest gain**: +0.033 mean nDCG@10 over
  BM25-only, best case FiQA +0.079 (Elastic's reproduction,
  https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid).
  Reproducible reference numbers: *Resources for Brewing BEIR*, arXiv 2306.07471.
  Read the size honestly: **+0.03 nDCG is a real effect and a small one**; a student who
  measures it with a bootstrap CI will find it is not always separable from noise on a
  single small dataset. That is a valuable thing to discover.
- **Vector + graph loses on single-hop** (F2): 63.0 vs 64.8 F1 on Natural Questions.
- **Vector + symbolic filter is not a "hybrid" at all** — it is a correctness fix (F6b:
  0.000 → 1.000). Different category, and by far the largest effect size in this report.
  Teach the distinction: *fusion* buys percentage points, *filtering* buys correctness.

### F11. Hyperbolic beats Euclidean for a hierarchy — reproduced qualitatively, **MEASURED HERE**, with an honest reproduction gap

Claim: Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*,
NeurIPS 2017, https://arxiv.org/abs/1705.08039 — a tree needs far fewer hyperbolic than
Euclidean dimensions, because hyperbolic volume grows exponentially with radius.

Script `scratchpad/poincare.py`: ~60 lines of torch, Riemannian SGD on the Poincaré ball
(Euclidean gradient scaled by `(1-‖x‖²)²/4`, projected back inside the ball), softmax loss
over sampled negatives — the paper's own objective. Data: WordNet `mammal.n.01` transitive
closure built from nltk, **1170 nodes / 6448 edges** (the paper uses 1,180 / 6,541 — the
small difference is WordNet version).

MEASURED, 1500 epochs, 50 negatives, total **126 s** CPU for the whole 8-cell sweep:

| dim | geometry | mean rank | MAP |
|---|---|---|---|
| 2 | hyperbolic | **195.8** | **0.0856** |
| 2 | euclidean | 302.9 | 0.0493 |
| 5 | hyperbolic | **117.4** | **0.1184** |
| 5 | euclidean | 192.8 | 0.0538 |
| 10 | hyperbolic | **86.2** | **0.1439** |
| 10 | euclidean | 113.4 | 0.0730 |
| 20 | hyperbolic | 78.1 | **0.1817** |
| 20 | euclidean | **62.1** | 0.1287 |

A cheaper 200-epoch / 20-negative run finishes the same sweep in **8.3 s** and shows the
same ordering more starkly (hyperbolic dim 5 mean rank 128.5 already beats euclidean
dim 20's 140.7 — a 4× dimension saving).

What is reproduced: **hyperbolic wins at every low dimension, by ~2× on MAP, and the
advantage shrinks as dimension grows** — at dim 20 Euclidean actually overtakes on mean
rank while still losing on MAP. That crossover is the honest version of the story and is
itself worth teaching: the geometry argument is a *low-dimensional* argument.

What is NOT reproduced: absolute quality. The paper reports MAP in the 0.8s at dim 5; my
minimal optimiser gets 0.12. My implementation is undertrained (full-batch SGD, no
proper burn-in schedule, no Adam-on-manifold). **Do not present my absolute numbers as a
replication of theirs.** For a notebook, either (a) budget ~2 min and teach the *ratio*
between geometries at matched compute, which is what actually matters, or (b) use
`gensim.models.poincare` and get closer to the published absolutes.

Confidence the effect survives at small scale: **high** for the ordering at dim ≤ 10;
**low** for any specific MAP value from a hand-rolled optimiser.

### F12. Linear representation hypothesis: relations are linearly decodable — for **48%** of relations — FROM PAPER

Hernandez et al., *Linearity of Relation Decoding in Transformer Language Models*,
ICLR 2024, https://arxiv.org/abs/2308.09124. A relation's subject→object mapping is
approximated by a single affine map (LRE) estimated from one prompt. Measured: robust,
faithful LREs exist for **48% of the relations tested**; for the rest the model clearly
knows the relation but does not encode it linearly.

Teaching value: this is the correct, quantified statement of the linear representation
hypothesis, and the number to quote is **48%, not "yes"**. It is the antidote to the
`king - man + woman` folklore in F1. Not directly reproducible under the course's
constraints (needs hidden states from a real LM), but it is the right *frame* for the
notebook that IS reproducible: F5's centroid read-out is exactly an LRE-flavoured
result — a relation partially recoverable by a linear operation on frozen vectors, that
does not transfer across domains.

---

## Ranking: which of these makes the best measured notebook

Ordered by (a) does a student produce the number, (b) is the number surprising,
(c) does it fit CPU / 5 min.

**1. "Cosine cannot reach hop 2" + "the graph's advantage has a break-even" (F4 + F7).**
The flagship. 10 s + 0.03 s. Produces an exact 0.000 against a 100% graph traversal, a
scaling knob (`n_docs`) that drives recall to zero, and then immediately undercuts the
graph with the `(1-p)^h` tax and a stated break-even (p_max = 0.45 at h=2, 0.26 at h=4).
Argues *both* sides with numbers a student generated. Nothing else here comes close.

**2. Negation inverts retrieval (F6b), backed by NevIR (F3).**
7 s. `precision@10 = 0.000` against a base rate of 0.255, and the ¬B half at 0.000 vs a
0.626 base rate — saying "not B" makes the retriever return B *every single time*. The
symbolic filter is 1.000. Largest effect size in the report and the clearest argument for
"put the constraint in a WHERE clause, not in the query string". NevIR gives it external
validity so it does not read as a rigged synthetic.

**3. Is-a direction: chance, then 0.79 untrained, then 0.88 trained, then chance again
across domains (F5 + F6a).**
6 s, four results, one script, no LLM. Hits the course's "embed once, train many heads"
idiom directly, includes a built-in confound check that partially fires (plant), and ends
on a reproduction of a 2015 critique that most practitioners have never heard of.

**4. Analogy arithmetic is a harness artefact (F1).**
Needs a static word-vector matrix (GloVe-6B-50d, ~65 MB, or a static-embedding
sentence-transformer). Pure `argmax cosine`, seconds. Delivers 0.71 → 0.21 and
"nearest-neighbour-of-b alone scores 70%". Perfect final-lesson demolition of a famous
demo. Only reason it is not higher: it needs one downloadable artefact the course does not
already cache, and it is about word vectors rather than the sentence/image encoders the
course uses.

**5. BM25 vs dense out-of-domain, and the honest size of RRF fusion (F10).**
Reproducible on one small BEIR set with `rank_bm25` (not currently installed). The
teachable moment is the **+0.033 nDCG@10** — a real effect that a bootstrap CI on a single
small dataset may not separate from noise. Exactly the course's "measure the effect size,
not the direction" lesson. Slightly lower because it needs a dataset download and careful
CI work to be honest.

**6. Symbolic verification of cached LLM generations (F9c).**
Publish a `VectorCache`-style artefact of generations, then have students run
deterministic checkers over them: JSON-schema validity, SQL parse + schema type-check
(`sqlglot`), triple-vs-ontology consistency. Produces a real error rate on real model
output with zero inference. Fully gradable. Ranked here only because it needs the
instructor to build and publish the cache first.

**7. Hyperbolic vs Euclidean hierarchy embedding (F11).**
8 s (cheap config) to 126 s (better config), reproduces the ordering, does not reproduce
the absolutes. Beautiful and self-contained, but the honest framing ("we reproduce the
direction, not the magnitude") makes it a weaker graded artefact than 1–6. Good optional
section.

## Flagged: fashionable, would NOT survive honest small-scale measurement

- **Differentiable logic / neuro-symbolic program induction (DeepProbLog, LTN, Scallop).**
  Every demo is MNIST-addition-scale. There is no laptop-scale measurement that shows a
  win over just training a network. Do not build a graded notebook on it. One paragraph,
  cited, moved past.
- **"GraphRAG beats RAG."** F2 and F10 say it loses on single-hop (63.0 vs 64.8 F1) and
  wins on multi-hop (70.3 vs 67.0). A notebook that reproduced "GraphRAG is better" would
  be reproducing a marketing claim. Teach the split, not the headline.
- **"Constrained decoding improves output quality."** It guarantees *validity* (a theorem)
  and measurably *hurts reasoning* (F9b). The conflation is everywhere. Any notebook
  claiming a quality gain from grammar constraints is measuring prompt quality.
- **"LLMs can build your knowledge graph."** F8: zero-shot relation-extraction F1 of
  0.15–0.42 on standard benchmarks. Combined with F7's `(1-p)^h`, an LLM-built graph is
  below the break-even for 3-hop queries. Building a graph in a notebook is easy;
  building one that *helps* is not, and the notebook should measure that, not assume it.
- **Hyperbolic embeddings as a drop-in upgrade.** F11's own dim-20 row shows Euclidean
  overtaking on mean rank. The claim is about *low-dimensional* efficiency, and it is
  routinely restated without that qualifier.

## What I ran vs. what I took from a paper

**Ran code on this machine (MEASURED HERE):** F4 (two-hop retrieval collapse), F5 (is-a
direction, four conditions), F6a (length confound), F6b (negation inverts), F7 (noisy-graph
`(1-p)^h` degradation and break-even), F11 (hyperbolic vs Euclidean, hand-rolled
Riemannian SGD — ordering reproduced, absolutes not).

**Took from papers (FROM PAPER), not reproduced here:** F1 (Linzen analogy numbers),
F2 (GraphRAG-Bench), F3 (NevIR), F8 (extraction F1 ranges), F9 (PoT, Spider/BIRD/Spider 2.0,
Let Me Speak Freely), F10 (BEIR, RRF gains), F12 (LRE 48%).

Scripts are checked in next to this file, in `planning/research/neurosymbolic_scripts/`:
`multihop.py` (F4), `hypernym.py` (F5), `negation.py` (F6a+F6b), `noisy_graph.py` (F7),
`poincare.py` (F11). They are short and self-contained; port them into the lesson notebook
rather than re-deriving. All except `noisy_graph.py` need
`sentence-transformers/all-MiniLM-L6-v2` (already in the local HF cache); `hypernym.py`,
`negation.py` and `poincare.py` additionally need `nltk` + the `wordnet` corpus
(`uv run --with nltk`, then `nltk.download("wordnet")` — ~10 MB, one time).
