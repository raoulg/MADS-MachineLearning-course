# Lesson 3 — What did that capacity buy you?

> Build plan. Delete this file once the lesson has been run.

## Why the old lesson has no subject

L3 was "here is ray". Ray now lands in L1's `04_search` and again in the search notebook
that arrived from L2, so by the time students reach lesson 3 they have used it twice. A
third introduction is not a lesson.

The subject it *should* have is the question every measurement in `evidence.md` keeps
answering the same way: **you added parameters — show that you got something for them.**
That is scaling laws, made empirical. This course can do what almost none can, because a
training run costs seconds: students *draw* the capacity/return curve on their own data
instead of reading about it.

It also gives an orphan a home. `5.13 — kan uitleggen wat scaling laws zijn` has been a
learning goal for years, parked in the attention lesson with no notebook behind it.

## What is already here

| file | state |
|---|---|
| `05_gates_highways_moe.ipynb` | arrived from L2, **measured and run**. The gating family + capacity-matched control |
| `06_search.ipynb` | arrived from L2, **measured and run**. Ray search over that family, params trained vs used |
| `01_hypertuner.ipynb` | 78 cells, `mads_datasets`, legacy `tune.run`, algorithm zoo with no through-line. **Rewrite** |
| `02_ray.ipynb` | 22 cells, `ExperimentAnalysis` (superseded result API). **Fold into the rewrite** |
| `05_hypertuning_exercise.md` | the graded assignment. Good bones, three dead references. **Update** |
| `hypertune.py`, `visualize.py` | the worked script the assignment points at. Check against ray 2.55 |
| `03_minimal_transfer_learning`, `04_transfer_learning_with_resnet` | **cut** — transfer learning is L1's premise |

## Proposed sequence

### 01 — The control question *(from `05_gates_highways_moe`, renumbered)*

Runs already. Gate, Highway, MoE and SparseMoE as three resolutions of one idea, each
measured against a **capacity-matched** control. The lesson's organising habit arrives in
notebook 01 rather than as an afterthought.

The evidence it carries is unusually good teaching material because it is negative and
self-correcting: a 3-seed check found `Gate` beating its control, a 5-seed × 3-epoch-count
check found them indistinguishable, and both are in the notebook (`evidence.md` §7).

### 02 — Scaling curves *(new — the only genuinely new build)*

Draw the three axes on cached vectors, where a point on the curve costs seconds:

- **accuracy vs parameters** — the head-capacity sweep, already measured: depth loses 5.2
  points at 15 sigma, 0/8 seeds (`evidence.md` §3)
- **accuracy vs training rows** — the DTD size curve 0.694 / 0.753 / 0.782 → 0.808
  (`evidence.md` §3b), with the shuffle trap as a live teaching moment
- **accuracy vs compute per run** — and the plateau. L5's opener now has a clean instance
  of this: unseen accuracy flattens after epoch 24 while seen accuracy climbs to 60
  (`evidence.md` §11), so the two lessons can cite each other

Closing idea: a scaling curve is how you decide **what to buy next** — more data, more
parameters, or more epochs — and on a frozen encoder the answer is usually none of the
three, which is why L1's baseline kept winning.

### 03 — What does this algorithm know? *(rewrite of `01_hypertuner`, absorbing `02_ray`)*

Not a zoo. One question, five answers, each motivated by a failure of the one before:

| | knows | fails at |
|---|---|---|
| grid | nothing | scales exponentially in dimensions |
| random | nothing, but samples better | wastes budget on bad regions |
| TPE | a model of the response surface | still runs every trial to completion |
| **population methods** (PSO, differential evolution) | keeps a *population*, so it explores jointly rather than one point at a time | no model of the surface; spends evaluations to get one |
| multi-fidelity (ASHA / Hyperband) | when to stop early | can rank on a prefix that misleads |

Population methods come from `8_extras/swarms`, where the optimisers currently sit with no
lesson to belong to. They are the one family that adds something students have not seen —
and on a cache they are entirely practical: a population of 100 is 100 evaluations of a
two-second trial, which is a couple of minutes on any laptop, no parallel cluster needed.
(PBT is the exception that genuinely wants concurrent workers, so treat it as a mention
rather than a section.)

Migrate to `tune.Tuner` + `tune.TuneConfig` + `tune.RunConfig` (ray 2.55, all under
`ray.tune`, no `ray.train` import). Drop `mads_datasets` — load from the hub or reuse an L1
cache so a trial costs seconds. Demote BayesOpt to a mention rather than carry
`bayesian-optimization==1.5.1` for one section.

### 04 — Budget and the Pareto front *(from `06_search`, renumbered)*

Runs already, and already carries the multi-objective material: `param_counts()` reporting
(total, active) per trial, and the measured tradeoff — 292,008 parameters trained, 43,368
(14.9%) used per input, which still lost to a dense 92,576-parameter trial
(`evidence.md` §9).

**To add: the budget question**, asked *before* ASHA is shown as its answer. Fixed compute
— 50 trials × 10 epochs, or 10 × 50?

**To add: the Pareto front as a concept, because it has to be introduced.** Students have
optimised one number all course. The trade-off game here is accuracy against *cost*, and
cost is not one thing:

| axis | what it buys you | what it costs |
|---|---|---|
| **parameters trained** | capacity | memory, storage, training time |
| **parameters used per input** | capacity *at inference* | latency, and the cloud bill |
| **training compute** | a better fit | time before you have anything |

Sparse routing exists precisely because the first two came apart — that is the whole point
of a mixture of experts, and L7 does the same trick again with whole systems instead of
expert layers.

The concept to teach: a model is **dominated** if some other model is at least as accurate
*and* at least as cheap on every axis you care about. Dominated models can be discarded
outright, with no judgement call. What survives is the **frontier**, where buying accuracy
always costs something — and on the frontier there is no "best model", only a choice of
where to sit, decided by a constraint you bring from outside (a latency budget, a memory
limit, a price per thousand calls).

`06_search`'s own numbers give a clean worked example, and it is a negative one: the
`sparse_moe` trial that trained 292k parameters and used 43k was beaten on accuracy by a
dense model with 92k parameters that used all of them. It is cheaper on one axis, more
expensive on another, and worse on accuracy — so on *this* task it is not on the frontier
at all. That is what a dominated point looks like, and students should meet one.

### 05 — Exercise *(update `05_hypertuning_exercise.md`)*

Keep the scientific-method cycle, the journal, the report rubric and the "don't use
hyperband when building a heatmap" warning — all still correct and all still good.

Fix what has rotted: it points at `03_minimal_transfer_learning` (cut), at
`2_convolutions/03_mlflow.ipynb` (folder removed), and at `mads_datasets`,
`configspace`/`hpbandster` and `bayesian-optimization` as environment advice. Retarget the
data suggestion at a published cache, and make the deliverable a **capacity curve with a
control**, not just a heatmap.

## The paper

**Kaplan et al. 2020, *Scaling Laws for Neural Language Models*** — read alongside
**Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models*** (Chinchilla).

The pairing is the point, and it is the best argument this course has for its own
standards. Kaplan's scaling law was believed and built on for two years. Hoffmann re-ran
the experiment with a better design — varying data and parameters together instead of
holding one near-fixed — and found a substantially different optimum, which changed how
frontier models were trained. A widely accepted empirical law was wrong because of how the
experiment was set up, and the correction came from measuring more carefully rather than
from a new idea.

That is the same failure this lesson teaches at laptop scale, and the same one this repo
has committed twice (`evidence.md` §7, §11).

Secondary: **Bergstra & Bengio 2012, *Random Search for Hyper-Parameter Optimization***,
for notebook 03 — the paper behind "grid search loses", and short.

## Dependencies this clears

L3 holds the last `mads_datasets` consumers in the course. When `01_hypertuner` is
rewritten, `mads-datasets` leaves `pyproject.toml`, and `bayesian-optimization` with it if
BayesOpt is demoted. `hyperopt` stays — it backs TPE in L1 and here.

## Risks

1. **02 is the only new notebook, and it is the one carrying the lesson's name.** If it
   does not produce a legible curve on cached vectors, the reframe is decoration. Build it
   first, not last.
2. **Population methods must still be measured, not assumed.** They are cheap enough on a
   cache (a population of 100 is 100 two-second trials), but cheap is not the same as
   better — check whether PSO or differential evolution actually beats TPE at equal
   evaluation budget here before writing a section that says it does. Equal budget is the
   control, exactly as elsewhere.
3. **The renumbering is cosmetic and the content is not.** `05_gates_highways_moe` and
   `06_search` are measured and running; do not rewrite them to fit a narrative.
