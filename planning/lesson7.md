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

> When you cannot train the model, all your leverage is in **what you feed it, what you
> fall back on, and whether you can tell the difference**. The prompt is not the design
> surface.

The lesson is a chain, not a list, because **evaluation and routing are the same
problem**. A router has to decide whether the cheap answer was good enough — that
decision *is* an evaluation, and it is the router's training signal. So the lesson
measures first and routes second, and the routing notebook cannot exist without the
evaluation notebook in front of it.

The closing line of the course, worth stating in the notebook: **routing is L3's mixture
of experts with the experts replaced by whole systems.** In L3 the router chose between
four tiny expert networks. Here the same router sits in front of a two-layer head and a
32B model. Same equation, five orders of magnitude apart.

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

### 02 — How do you know it is any good?  **[BUILT]**

The gap this lesson exists to close. Students experiment with LLMs constantly and judge
the output by looking at it. Eyeballing is not a metric, and it does not scale past ten
examples — but almost nobody teaches the alternative, so the habit persists.

The arc, cheapest first:

1. **Where ground truth exists**, use it: exact match, F1. Then show where it breaks — two
   correct answers phrased differently score zero against each other.
2. **Embedding similarity** as the obvious fix, and its measured failure: a paraphrase and
   a contradiction of the same sentence score about equally. Cheap to demonstrate on a
   cache, and it kills the "just use cosine" instinct properly.
3. **LLM-as-judge**, built with a local model — and then **the judge gets judged**, which
   is the part that never happens in practice:
   - agreement with human labels (Cohen's kappa, not raw accuracy)
   - **position bias**: present the same pair in both orders and measure how often the
     verdict flips. Minutes to run, and the number is usually shocking.
   - verbosity bias: does the longer answer win regardless of quality?
   - self-preference: does the model prefer its own output?
4. **A cheap classifier judge** trained on cached human labels, compared to the LLM judge
   on agreement *and* cost. Sometimes it is competitive, which is the course's argument
   again.

Punchline: **an unvalidated judge is eyeballing with extra steps.** The deliverable is
not "use LLM-as-judge", it is "measure your judge before you trust it".

Measured against `qwen3.6:35b` via ollama — 200 direct judgements, 119 pairs asked twice:
kappa 0.939 against ground truth, 2% unparseable (all of them the model answering "Mixed"),
accuracy 0.966 with the right answer listed first against 0.899 listed second, and **8.4%
of verdicts flip under a swap**. Verbosity bias came out *undetectable* (judge picked the
longer review 27.7% against a 29.4% base rate), so the notebook reports it as a confounded
metric rather than a finding. `build_judgements.py` regenerates in ~4 min; the 33 KB cache
is committed so students analyse on CPU in a second.

This notebook is also load-bearing for 03: the metric built here is the signal the router
is trained against.

### 03 — Build your own router

The constructive centre. Cheap path versus expensive path, on a corpus with a
`VectorCache`.

- **Cheap path**: a head on cached embeddings. Milliseconds, CPU.
- **Expensive path**: a local model via ollama. Seconds, and it needs the hardware.
- **The router**: a small model predicting *will the cheap path be right?* — trained
  against notebook 02's metric, on features the cheap path already produces (its own
  confidence, the embedding, query length, retrieval margin).

Measure the accuracy/cost Pareto: at what escalation rate do you keep 95% of the
expensive path's accuracy? And keep the controls honest, per L3's habit:

| arm | why it is there |
|---|---|
| always cheap | the floor |
| always expensive | the ceiling, and the cost you are trying to avoid |
| random escalation at the same rate | proves the router is doing something |
| **threshold on the cheap head's own confidence** | the obvious baseline, and it is often hard to beat |

That last row is the capacity-matched control transplanted into a systems setting. If a
trained router cannot beat "escalate when the cheap model is unsure", say so — same rule
as everywhere else in this course.

**Distillation lives here**, because it is how you make the cheap path good enough to
route to in the first place: the instructor runs the big model once over the training set
and publishes its outputs, soft probabilities included, as a cache; students train the
cheap head against those instead of against ground truth. Three rows and a cost column —
generalist zero-shot, head distilled from it, head trained on real labels. The third row
is the control that stops it being a magic trick: **when you have real labels,
distillation is usually worse than just using them.**

### 04 — The index is the model

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

### 05 — Memory over time

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

### 06 — Exercise

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

- **Evaluating generative output** — 02. The gap that made the lesson a chain instead of
  a list: students eyeball LLM output because nobody teaches them not to, and the metric
  built here is what the router in 03 is trained against.
- **Routing, with distillation inside it** — 03. Closes the course's argument, and is
  L3's MoE router with whole systems as the experts.
- **Index interventions** — 04.
- **Replay / continual learning** — 05. Already measured, tiny, practical.

Considered and not taken this round:

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
