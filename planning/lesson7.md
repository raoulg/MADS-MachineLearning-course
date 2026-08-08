# Lesson 7 — Systems built on a model you did not train

> Build plan. Delete this file once the lesson has been run; the notebooks describe it
> after that.

## The problem this lesson has to solve

L7 was going to be "generalists and memory", and the research came back mostly negative:
GraphRAG's headline is unfalsifiable, HippoRAG's is a max over three datasets, LLM
relation extraction is too noisy to traverse. A lesson made of demolitions is not a
lesson. Students who leave knowing only "be careful with the hype" have learned caution,
not a skill.

But the research is not actually negative about *structure*. It is negative about
**inferring** structure. On the two-hop task, graph traversal scored 100% reading two
documents while cosine scored 0.000 — structure won outright. What failed was asking an
LLM to hallucinate the edges first. That is a much more useful finding, and it is the
lesson's hinge.

## The spine

The course has argued since L1 that **the representation is the model**: encoder choice
moves accuracy 10-40 points, head architecture under one. L7 is the same sentence one
level up.

> When you cannot train the model, all your leverage is in **what you feed it and what
> you fall back on**. The index and the cheap path are the design surface. The prompt is
> not.

Every notebook is a measured intervention on that surface. Three of them win.

## Notebooks

### 01 — The generalist, measured

Exists already as `01_foundation_models.ipynb`, moved from L2. Keep intact: the
zero-shot TimesFM comparison, the contamination check (verified against
`Salesforce/lotsa_data`'s file listing, not assumed), and the rule it produces — *does
this encoder compress or expand?* — with the measured collapse table (`evidence.md` §6:
raw window 0.796, embedding+raw 0.925, mean-pooled embedding 1.855, all vs naive).

**To add:** the language half. An LLM zero-shot on the L4 SCOTUS/IMDB task versus the
head students trained in L4, on accuracy *and* cost. Runs off a published generation
cache. This is the notebook that makes the lesson two-modality rather than LLM-only.

### 02 — The index is the model

The heart of the lesson. Four interventions on one retrieval index, on generated corpora
that run in seconds. Three work; the fourth is the honest tax.

| intervention | measured | status |
|---|---|---|
| add a **time axis** (recency prior) | recall@1 0.117 → 0.983 | works |
| add a **symbolic filter** (negation) | precision@10 0.000 → 1.000 | works |
| traverse **real edges** (two-hop) | 0.000 → 1.00 | works |
| **extract** those edges with an LLM | ~6% at three hops | the tax |

Three things to build the teaching around:

- **The metric trap comes first.** In the staleness case recall@**5** is 0.933 while
  recall@**1** is 0.117 — the standard metric says the flat system is fine and it is not.
  Students should meet that before any fix, so the fixes answer a question they already
  believe in.
- **Negation inverts, it does not degrade.** On "A but not B", precision on the ¬B half
  is 0.000 against a 0.626 base rate: writing "not B" returns B every time, when ignoring
  the query would have been right 63% of the time. Corroborated externally by NevIR
  (EACL 2024). A four-line filter fixes it completely.
- **Ground-truth edges versus extracted edges is one dial, not two notebooks.** Real
  structure is everywhere — foreign keys, org charts, git history, citation graphs,
  import graphs, ticket links. There `p ≈ 0` and `(1-p)^h ≈ 1`. Ask an LLM to infer edges
  from prose and published zero-shot relation-extraction F1 is 0.15-0.42, so `p ≈
  0.6-0.85` and three hops answers ~6% of the time. The break-even is explicit:
  `p_max = 0.452` at two hops, `0.260` at four.

The closing line: **don't extract a graph, use the one you already have.** Graphs are
demoted from *topic* to *one of four interventions*, which is the honest weight.

### 03 — The cheap path: routing and distillation

The constructive centre, and the answer to "what actually helps".

**Routing.** Not every query deserves the expensive model. Send the easy ones to a cheap
head on cached vectors and escalate only the hard ones — and this is *literally L3's MoE
router* one level up, which is worth saying out loud. Measure the accuracy/cost Pareto:
at what escalation rate do you keep 95% of the generalist's accuracy? The honest control
is the flat baselines at both ends.

**Distillation.** Since the question came up, concretely: the instructor runs the big
model **once** over the training set and publishes its outputs — including soft
probabilities, not just hard labels — as a cache. Students then train a small head on
frozen embeddings against *those* targets instead of against ground truth. Three numbers
and a cost column:

| model | what it costs | what it scores |
|---|---|---|
| generalist, zero-shot | seconds per item, API or GPU | ? |
| head distilled from its outputs | milliseconds, CPU | ? |
| head trained on real labels | milliseconds, CPU | ? |

The interesting outcomes are all teachable. The distilled head usually lands close to the
generalist at a tiny fraction of the cost — the course's own cache argument, applied to
competence rather than to embeddings. Sometimes it *beats* the teacher, because the
teacher's errors are noisy and the student averages over them. And the third row is the
control that keeps it honest: **when you have real labels, distillation is often worse
than just using them**, so distillation is what you do when labels are expensive and the
generalist is cheap-ish — not a free upgrade.

### 04 — Memory over time

What happens when next month's data arrives. Measured already: class-incremental heads on
frozen vectors collapse to **0.024** on task 1 (below chance) with no replay, and a
**sub-1MB** buffer of 100 vectors per class restores average accuracy 0.509 → 0.848 while
cutting forgetting 0.664 → 0.122. Instructor ships ~5.4 MB; students train in seconds.

This is "embed once, train many heads" applied to *time*, and it is the most practical
thing in the lesson.

Forgetting then gets a Pareto table rather than a slogan: soft forgetting (a recency
weight) trades "work now" against "work in 2019" — w=0 gives 0.117 / 0.367, w=0.3 gives
0.967 / 0.000 — while hard deletion puts the gold document outside the index for 100% of
historical questions. **Deciding what to forget is a design choice with a measurable
cost.**

The sleep/dreaming literature gets one honest paragraph: replay under a different name is
the substantive part, and the rest is largely metaphor with a paper attached. One of the
"dreaming" papers samples 60 dreams and rejects 45, which tells you the filter is doing
the work.

### 05 — Exercise

Students bring their own corpus (or extend the L4 one) and must produce: one retrieval
intervention that measurably helped, one honest tax or negative result, and a cost column
on every row. Same deliverable contract as the rest of the course.

## Model access

Published caches are the spine — generations, extractions and soft labels precomputed by
the instructor, so every graded cell is CPU-only, deterministic and free. An optional,
clearly marked path regenerates them with a local ~3B model via ollama for students with
the RAM. Retrieval metrics (recall@k, MRR, nDCG) need no model at all at student time,
which is why notebook 02 costs nothing to run.

## Candidates considered

Kept because they measure well and are constructive:

- **Routing and distillation** — 03 above. Closes the course's argument.
- **Replay / continual learning** — 04 above. Already measured, tiny, practical.
- **Index interventions** — 02 above.

Considered and not taken this round:

- **Evaluating generative systems** (LLM-as-judge, its failure modes, agreement with
  human labels, building a cheap classifier-based eval instead). Genuinely strong and
  practical, and the best candidate if the lesson needs a sixth notebook or if 03 splits.
  Left out only for size.
- **Active learning / which 100 examples to label** — cheap on caches, ties to the data
  axis of L3's scaling curves. A good exercise, probably not a notebook.
- **Quantization and inference efficiency** — real, but hardware-dependent results are
  exactly what the cache pattern exists to avoid.
- **Prompt engineering as a measured axis** — too unstable to teach as a result.
- **Differentiable logic / neurosymbolic proper** — still MNIST-addition scale. One
  paragraph at most.
- **Hyperbolic embeddings** — the research reproduced the *ordering* (2x MAP at low
  dimension) but not the published absolutes, and Euclidean overtakes on mean rank by
  dimension 20. Not a drop-in upgrade, and not worth a notebook on that evidence.

## Risks

1. **Most numbers here are agent-produced and unverified.** `evidence.md` §10 is the only
   one re-run by hand so far. Everything cited above must be independently reproduced
   before a notebook states it as a result — that is the rule in `research/README.md`,
   and this lesson is the reason it exists.
2. **Generated corpora are not real corpora.** The synthetic tasks show the mechanism
   cleanly, which is their virtue and their limit. At least one section should land on
   real data, or the lesson proves only that the generator has the property it was
   written to have.
3. **Size.** Five notebooks with three new builds. If it slips, 04 is the one to defer —
   it stands alone and would survive as its own lesson.
