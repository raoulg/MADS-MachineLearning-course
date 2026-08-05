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

- **Time series** (L2) — no general-purpose vectorizer worth freezing; windowing and
  training end-to-end remains correct. Confirmed by measurement: a linear model beat
  N-BEATS on both pedestrian_counts and ETTh1.
- **Graphs** (L6) — structure is the signal; there is no per-node encoder to cache.
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
| **D5** | ray.tune from L1, TPE (`HyperOptSearch`) only. Algorithm zoo stays in L3. | One tool, used properly, beats five tools named. |
| **D6** | The L1 search space includes **encoder choice** alongside width/depth/noise/lr. | Measured: encoder moves accuracy 10–40 pts; head architecture moves it <1 pt and usually downward. §3.3 |
| **D7** | Nonlinearity is taught once, unifying dense and conv: `σ(Wx + b)` and `σ(W * x)`. | Convolution is a linear operator with weight sharing; the nonlinearity is the shared essential part. Lets L1 absorb the old L2 conv material. |
| **D8** | Old L1 + L2 notebooks move to `notebooks/legacy/`, not deleted. | Reference during transition; deleted once the replacements have been run. |
| **D9** | ~~L2 becomes "produce a cache + track experiments"~~ — **superseded by D14.** | Ray and mlflow both landed in L1 (notebook 01 §1.5, notebook 04) before this lesson was ever built, so a whole lesson re-teaching them was redundant on arrival. See `cuts`. |
| **D10** | Two-encoder fusion is taught as a **hypothesis the student tests**, and the notebook is honest that simpler won. | No free lunch. The deliverable is the research habit, not the verdict. See §1.4. |
| **D11** | `VectorCache.from_hub()` and `get_or_create_cache` move **into vectormesh**. | Blocking prerequisite — see §4. |
| **D12** | Caches are joined on `source_idx`, with an explicit alignment assertion. | Row order happens to match today; relying on that silently is how you get a model trained on mismatched pairs. |
| **D13** | **No notebook builds a full cache.** Every notebook downloads a published one. Building is taught on a ~200-image subsample that finishes on any hardware, and the full build lives in `make build-lesson1`. | Student hardware ranges from new MacBooks to CPU-only VMs. See §3.7. |
| **D14** | **L2 is redefined as time series**, absorbing the round-1 `3_recurrent_networks` rebuild (renamed `2_timeseries`). D9's "build your own cache" L2 is cut outright, not deferred. | §1.2 already named time series as the counterweight to L1; there is no reason to spend a lesson number on cache-mechanics students already practised in L1. See `cuts` and §6. |
| **D15** | The probabilistic-forecasting notebook moves off ETTh1-only onto **australian_electricity_demand** (Monash, CC-BY-4.0). | Not because ETTh1 is noisier point-to-point — measured, it is not (§3.8) — but because its difficulty is regime drift/non-stationarity, which is a harder, more confounded story to teach calibration on than a clean high-SNR series. ~~ETTh1 stays in the lesson elsewhere as the literature-comparable, harder case~~ — **superseded by D18**: ETTh1 was dropped from the graded sequence entirely. |
| **D16** | Gating (Gate/Highway/MoE) is split into its **own** L2 notebook, framed as general-purpose composition inspired by the GRU/Highway/MoE papers — not a time-series-only trick. | It was buried inside the old `04_composition.ipynb` §4.6 alongside unrelated feature-engineering content. Splitting it lets the notebook cite its sources properly and lets the closing search notebook (D17) treat "gate: yes/no" as a clean, nameable axis. Retargeted to `australian_electricity_demand` by D18 — see §3.9. The dense-MoE-only
gap this left (§1.4's "never present an architecture as correct because the idea is
elegant" cuts both ways: also don't leave "the real version isn't tested" as a permanent
caveat) closed 2026-08-05 — `vectormesh` 0.7.0 added `SparseMoE`, measured directly in
the notebook's new §5.5. See §4b. |
| **D17** | L2 closes with a ray.tune search notebook, mirroring L1 notebook 04 — search over basis choice, gate on/off, highway depth. | D5's "one search tool, used properly" only pays off if students exercise it twice in different modalities, not once and never again. |
| **D18** | L2 consolidated from **five** datasets (ETTh1, pedestrian_counts, sunspot, australian_electricity_demand, traffic_hourly) to **three** (sunspot, pedestrian_counts, australian_electricity_demand). ETTh1 dropped from the graded sequence entirely, kept only in the optional appendix `09_composition_extra`. | Direct user feedback, 2026-08-05: five datasets across nine notebooks was real sprawl, and ETTh1's six-covariate electricity-transformer domain added confusion an unfamiliar dataset didn't need to, on top of the dataset count itself. Daily sunspot counts take over ETTh1's structural role (§3.9) — a series where the naive baseline is hard to beat — while being a series students already met in MADS-DAV. |
| **D19** | Every epoch count in L2 must be justified by a checked convergence curve or a multi-seed, multi-epoch-count sweep — not copied from another notebook. | Direct user challenge, 2026-08-05, after the `EPOCHS=1` bug (§4a) was fixed by simply picking a different number: "why would 8 epochs be enough? who says it doesn't need 100?" Caught a real error doing this — an initial 3-seed check of `05_gates_highways_moe` at one epoch count found Gate beating its capacity-matched control; a proper 5-seed, 3-epoch-count check found that result was noise (§3.9). Both the wrong finding and the correction are on the record in the notebook, not silently fixed. |
| **D20** | `pedestrian_counts` is replaced by **`solar_10_minutes`** as `03_decomposition_and_nbeats`'s primary demo and `06_search`'s search dataset. D18's three-dataset count is preserved — the same swap is made in both places, not a fourth dataset added. | Direct user challenge, 2026-08-05: D18 left `03` without a single dataset in the graded sequence where N-BEATS's basis decomposition wins on accuracy (linear won on both `pedestrian_counts` and `sunspot`, §3.9) — a notebook whose whole premise is "the decomposition inside the architecture" never showed that premise earning anything beyond interpretability. The user explicitly rejected re-manufacturing a win via data scarcity (the old ETTh1 story) and asked for a series where N-BEATS wins on its merits, with ample data. Found by searching, not assumed — see §3.10. |

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

### 3.4d The DTD caches are grouped by label — subsampling must shuffle first

Found while building `lesson1.yml` notebook 05's data-size ablation, and it silently
invalidated the first version of that exercise.

| cache | first 940 train rows contain | why |
|---|---|---|
| `eurosat-*` | all 10 of 10 classes | built with `--n-train 6000`, which shuffles before selecting |
| `dtd-*` | **12 of 47 classes** | built with `--full`, which does not shuffle at all |

So `cache.dataset.select(range(n))` on DTD returns a **class-truncated** subset, not a
smaller sample of the same task. The failure is silent and looks like a learning curve:
470/940/1880 rows scored 0.121 / 0.238 / 0.437, which reads as "more data helps" but is
actually `n/3760` — the model was scoring ~100% on the only classes it had ever seen and 0%
on the rest. With `.shuffle(seed=0).select(range(n))` the same sizes give **0.694 / 0.753 /
0.782**, against 0.808 on the full split: a real, boring, correctly-shaped curve.

Two consequences:

- **Any notebook that subsamples a cache must shuffle first.** `lesson1.yml` notebook 05
  does, and says why — it is a good teaching moment about looking at the data before
  slicing it.
- **`make build-lesson1`'s `--full` path should shuffle**, so the artefact does not carry
  this trap for the next person. Not blocking L1 (the notebooks shuffle at use), but it
  belongs to `make build-lesson1`/`scripts/build_cache.py` maintenance directly — there is
  no longer a lesson whose job is cache-building (D14 cuts it). Logged here so whoever next
  touches `scripts/build_cache.py` finds it.

### 3.4e GaussianNoise does nothing measurable on DTD, at any training size

The ablation the above bug was hiding, rerun correctly (dinov2-small, 3 seeds, epochs
scaled so every size gets comparable gradient steps):

| train rows | sigma=0.0 | sigma=0.3 | Δ | train accuracy |
|---|---|---|---|---|
| 470 | 0.6940 | 0.6902 | −0.38 pt | 1.000 |
| 940 | 0.7527 | 0.7534 | +0.07 pt | 1.000 |
| 1880 | 0.7819 | 0.7837 | +0.18 pt | 1.000 |

§3.3's "+0.4 pt on DTD" was inside noise and should not have been written as a benefit.
Note the last column: the head reaches **100% training accuracy at every size**, so it is
memorising completely, and a regulariser still moves the test number by nothing. That is
the informative part — the ceiling is set by the frozen representation, not by head
capacity, so regularising the head cannot reach it. Consistent with §3.3 and §3.6.

### 3.4c Raw-width and projected fusion: a wash, not a loss (the open question, closed)

`lesson1.yml` notebook 04 carried an `open_question`: §3.4's fusion was measured with a
128-wide bottleneck per branch, and raw-width fusion was *impossible* before the 0.5.0
`Concatenate2D` fix. Measured now, DTD 47-class, 20 epochs, **8 seeds**, dinov2-small +
resnet-18:

| setup | mean | std | Δ vs baseline | sigma | beats baseline |
|---|---|---|---|---|---|
| **baseline `NeuralNet(384, 47)`** | **0.8098** | 0.0024 | — | — | — |
| baseline + a hidden layer (2× params) | 0.7579 | 0.0086 | **−5.19 pt** | −15.4 | 0/8 |
| fusion, raw widths (384+512) | 0.8142 | 0.0126 | +0.43 pt | +0.9 | 7/8 |
| fusion, weak stream projected to 384 | 0.8132 | 0.0087 | +0.33 pt | +1.0 | 7/8 |

Three things this changes, and one it confirms:

- **Raw-width fusion does not lose.** It wins 7 seeds out of 8, by an amount that is inside
  its own noise (~1 sigma). The honest verdict is *indistinguishable from the baseline*, at
  3× the parameters and **4–5× the seed variance** (std 0.013 vs 0.002).
- **The 128-bottleneck penalty in §3.4 was the bottleneck, not the fusion.** Squeezing a
  384-wide stream through 128 throws away more than the second stream adds. Written as
  "fusion loses" this was an over-generalisation of our own measurement — exactly what §1.4
  warns against, committed here.
- **3 seeds lied.** The first run of this experiment used 3 seeds and reported +0.83 pt for
  projected fusion; 8 seeds says +0.33. Neither is wrong; the first was under-powered. This
  is now a teaching cell in notebook 04, not a footnote.
- **Confirmed, and harder than before:** added *depth* on a frozen encoder loses badly and
  unambiguously — 5.2 points at 15 sigma, 0/8 seeds. §3.3's finding survives contact with a
  proper seed budget; the fusion finding did not.

Consequence for D10 and `lesson1.yml` §4.3: the section still poses fusion as a hypothesis
the student predicts and tests, but the notebook must **not** report "the baseline wins".
It reports a wash with a variance cost, which is a more useful result and a better argument
for measuring rather than believing.

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

### 3.8 ETTh1 is not obviously the noisiest time-series candidate — measured

D15 started from the instructor's impression that ETTh1 (used pervasively in the round-1
`3_recurrent_networks` build) is "chaotic, noisy" and looked for a calmer alternative for
the probabilistic-forecasting notebook. Measured before picking one, 2026-08-05, on
normalised series (z-scored on the full series): seasonal-naive residual RMSE at the
dataset's own daily period, and the fraction of spectral power in the top-5 FFT bins
(higher = more of the series' energy sits in a few clean periodic components):

| dataset | source | CoV | seasonal-naive RMSE (period) | spectral top-5/total |
|---|---|---|---|---|
| ETTh1 (OT) | `ETDataset/ett` | 0.62 | 0.34 (24h) | 0.77 |
| pedestrian_counts | Monash | 1.04 | 0.33 (24h) | 0.81 |
| **australian_electricity_demand** | Monash | **0.20** | 0.48 (48 half-hours) | 0.70 |
| traffic_hourly | Monash | 1.27 | 0.77 (24h) | 0.38 |
| solar_10_minutes | Monash | 1.30 | 0.56 (144 steps) | 0.65 |
| weather | Monash | 3.28 | 1.43 (144 steps) | — |

ETTh1 is not the noisiest of these by either metric — it is smoother point-to-point than
pedestrian_counts and has a cleaner spectrum than traffic_hourly. Its reputation for being
hard tracks a different property: known regime drift / non-stationarity between train and
test spans (the reason linear models like DLinear beat transformer forecasters on it in the
literature), not instantaneous variance. `australian_electricity_demand` is the genuinely
calmer series by the metric that matters for a Gaussian/quantile head — CoV three times
tighter than ETTh1 — while remaining a real Monash benchmark, not a synthetic dataset built
to look clean. See D15.

`sunspot` (Monash, daily, 1818–2020, single series, 73924 points) was also checked as an
N-BEATS candidate. Students already met it in MADS-DAV (`03.2-statistics-of-time.ipynb`):
statsmodels `seasonal_decompose` fails on it there (trend absorbs a >100yr cycle, the
residual keeps visible sine-wave structure), a Fourier reconstruction is shown overfitting,
and DAV's own closing notes name "finding a way to model the very long (+100 year) cycle"
as future work and hand it forward explicitly: "In the course on Machine Learning we will
go much deeper into this topic." L2 is that hand-off.

Verified by resampling to yearly means and counting how many repeats the 203-year span
actually contains of each named solar cycle, rather than assuming a fixed-period basis
simply fails:

| cycle | period | repeats in 203yr | verdict |
|---|---|---|---|
| Schwabe | ~11yr | 18.5 | plenty — fits a fixed-period basis confidently, same register as pedestrian_counts |
| Gleissberg | ~87yr | 2.3 | visible, not pin-downable — an 11yr-rolling-max envelope over the yearly series shows real modulation (227→182→110→142→174→191→269→220→174, sampled every 20yr) but two-and-a-bit repeats won't fix the period precisely |
| Suess/de Vries | ~210yr | 0.97 | not resolvable at all — under one full cycle |

So rather than posing this purely as "a fixed-period basis fails here," the notebook adds a
**second** basis period (~87yr) alongside the Schwabe one and shows the fit improve — a
real, usable multi-scale decomposition that answers DAV's own suggested improvement — then
names the ~210yr cycle as the case genuinely out of reach given the data span. "How much
data would you need" becomes the concrete takeaway in place of "this doesn't work." See
`lesson2.yml` for how this is used.

### 3.9 D18 consolidation — sunspot's daily rotation, and a correction caught mid-build

D18 (§2) drops ETTh1 from L2's graded sequence and moves daily sunspot counts into its
structural role. Two things had to be verified before that swap was defensible, and one
mistake was made and caught while building it — recorded here rather than silently fixed,
per the standing rule.

**The clean daily tail, and a real rotation period.** `sunspot_daily.py` starts at
1848-12-22 (the last date with a missing day anywhere ahead of it) — 62,617 consecutive
daily points, 171.4 years. Autocorrelation on that span has a local maximum at lag 26-28
days (0.77, above its lag-14 and lag-60 neighbours, 0.68 and 0.70); the FFT's short-period
peaks cluster at 26.8-28.2 days. This is the Sun's ~27-day synodic rotation — a genuine
physical quasi-period, distinct from any calendar season, checked rather than assumed.
Day-to-day persistence is still much stronger (acf(1d)=0.96), so naive still beats
seasonal-naive(27d) — the same shape of result ETTh1 gave (§3.8), now without ETTh1's
six-covariate domain. Separately, the single largest peak in the *entire* daily spectrum
(measured over the 151-year training span) sits at ~11 years — the Schwabe cycle is
visible even at daily resolution, ~160x the power of the 27-day peak, and far too slow for
any window this lesson's windowing notebooks hold — which is exactly why the yearly
zoom-out in §3.7 exists as a second look at the same series.

**The `03_decomposition_and_nbeats` §3.6 contrast does not reproduce ETTh1's data-scarcity
flip.** ETTh1's version had N-BEATS beating linear because ETTh1's window only produced
~1,400 training windows — too few for the linear layer's parameters to fit without
overfitting. Sunspot at the same `W=168,H=24` window produces ~9,187 windows, and linear
wins outright, same as pedestrian. The seasonality/trend ratio still drops honestly (1.071
pedestrian → 0.608 sunspot) and the headroom over naive still shrinks (77% → 19%,
explained by sunspot's high `acf(1d)`) — but "N-BEATS wins when data is scarce" is no
longer demonstrated anywhere in the lesson's graded sequence. The rewritten notebook
reports what is actually true rather than forcing the old flip onto new numbers.

**A wrong finding, caught by checking harder — D19.** Building `05_gates_highways_moe` on
`australian_electricity_demand`, a first check (3 seeds, one epoch count) found `Gate`
beating its capacity-matched `deeper` control by a real-looking margin (0.157 vs 0.176). A
follow-up check (5 seeds, three epoch counts: 50/90/130) found `Gate` and `deeper`
statistically indistinguishable at every one of them — the first result was noise from an
under-sampled check, exactly the trap this course's own "3 seeds lied" finding (§3.4c)
warns about. The corrected, honest result: `deeper` and `gate` tie (0.159 vs 0.160 at
`EPOCHS=90`), `highway` and `moe` cost 1.8x and 2.1x the parameters without a margin that
clears their own seed noise — the same null result ETTh1's version gave, on a series that
should have given gating a better chance. Both the wrong number and the correction are
written into the notebook's §5.2, not silently replaced.

**Contamination check, verified not assumed.** `08_foundation_models`'s procedure only
works if the series is plausibly in a foundation model's pretraining mixture.
`Salesforce/lotsa_data`'s file listing contains a `sunspot_with_missing` config, confirming
the same contamination risk ETTh1 carried transfers to sunspot. The frozen-encoder
collapse (§8.4) replicates qualitatively with real sunspot numbers: mean-pooled embedding
MAE ratio 1.855 (worse than naive), last-patch 1.910 (worse still), embedding+raw 0.925
(worse than raw alone at 0.796) — same three-hypothesis elimination, same conclusion, new
series.

### 3.10 D20 — a genuine N-BEATS win, found by search rather than assumed away

Direct user challenge, 2026-08-05, to `03_decomposition_and_nbeats` as §3.9 left it: a
notebook whose entire premise is "put the decomposition inside the architecture" had no
dataset, anywhere in the graded sequence, where that architecture actually won on accuracy
— linear beat N-BEATS on both `pedestrian_counts` and `sunspot` (§3.9). The user explicitly
rejected re-manufacturing a win via data scarcity (the old ETTh1 story, ~1,400 windows,
really an overfitting story about linear rather than a decomposition story) and asked for a
series where N-BEATS wins **on its merits**, with enough data that the win cannot be
dismissed as an artefact.

**Checked, not guessed — every candidate below run at a window count comfortably above the
linear model's own parameter count, so none of these results can be a scarcity flip:**

| dataset | window (W, H, period) | pooled windows | result |
|---|---|---|---|
| `nn5_daily` | 35, 7, 7 (weekly) | 3,394 (8 series) | linear wins: 0.6717 vs generic 0.6995 |
| `traffic_hourly` | 336, 24, 168 (weekly) | 6,780 (4 series) | linear wins: 0.2787 vs generic 0.3750; seasonal-naive(168)=0.2192 beats both |
| `london_smart_meters` | 336, 24, 168 (weekly) | 9,093 (4 series) | linear wins: 0.5935 vs generic 0.6529 |
| `tourism_monthly` | 24, 12, 12 (annual) | 2,420 (20 series) | linear wins clearly: 0.5705 vs generic 0.8211 (generic overfits badly on test: 2.67 vs 1.01) |
| **`solar_10_minutes`** | 288, 144, 144 (daily) | **103,700 (20 series)** | **N-BEATS generic wins: 0.2177 vs linear 0.2523** |

Solar power is the one series checked where the daily period is close to a physical
certainty rather than a statistical regularity — exactly zero overnight, a smooth hump
during the day, governed by the sun's position rather than aggregate human behaviour. That
is the mechanistic reason a period-144 Fourier basis beats an unconstrained 41,616-parameter
linear map even with 103,700 training windows (2.5x the linear model's own parameter
count): `nn.Linear(288,144)` has to learn "step *t* resembles step *t*-144" independently at
144 different output positions; the seasonality stack cannot help assuming it once, because
every output is built from one 20-number `theta` reused across a single continuous Fourier
phase. Full account, with the epoch/seed robustness check (8/12/20 epochs × 5 seeds, gap
holds at 0.030-0.035 throughout — D19 discipline), in notebook 03 §3.4.

**D20**: `pedestrian_counts` is replaced by `solar_10_minutes` as `03`'s primary demo and
`06_search`'s search dataset (both notebooks used `pedestrian_counts`; the same swap in
both keeps D18's three-dataset count rather than adding a fourth). `pedestrian_counts` is
retired from the graded sequence entirely — no notebook in the current 01-08 list uses it.

**A correction inside the correction: re-measuring §3.9's sunspot contrast at the new,
larger window.** `03` §3.6 runs the identical architecture "unchanged, at the same W,H"
the primary demo used — now `W=288,H=144` rather than the old `W=168,H=24`. At this window,
N-BEATS generic's nominal edge on sunspot (0.3225 vs linear's 0.3270) is **not** a real
result: the gap (0.0045) is smaller than N-BEATS's own seed-to-seed std (0.0099). This is a
sharper version of §3.9's finding ("linear wins outright" on sunspot) — the honest read at
this larger window is a **statistical tie**, not a linear win. Recorded because the window
changed and the number was re-measured, not carried over from the smaller-window run.

**`06_search`'s result changed shape too, not just its dataset name.** The original
`pedestrian_counts` search (main_principles.md, superseded) found `gate` edging `plain`'s
own best trial by 0.0028 — inside the 2-std noise band that check exists to catch, read as
"no clear win." Re-run on `solar_10_minutes` (30 trials, same 5-mechanism/5-axis space):
`gate` wins at 0.2079 against `plain`'s own best trial of 0.2214 — a 0.0135 gap, over 6x
the 5-seed baseline's own std (0.0021) — and **every** routing mechanism's best trial beat
`plain`'s best trial (`highway` 0.2102, `moe` 0.2095, `sparse_moe` 0.2099), which itself
had never happened on `05`'s electricity window or the earlier `pedestrian_counts` search.
`sparse_moe`'s best trial (`n_experts=2`) ties dense `moe` at 2.4x fewer trained parameters
(44,626 vs 106,454). Full table in notebook 06 §6.3.

## 4. Blocking prerequisites

Ordered. Nothing in L1 can be finished before these land.

| # | Item | Where | Status |
|---|---|---|---|
| **P1** | `VectorCache.from_hub(repo, split=...)` — `VectorCache.load()` is local-disk only; every notebook currently open-codes `snapshot_download` + `load`. | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P2** | Promote `notebooks/4_attention/cache_utils.py` (`get_or_create_cache`, encoder-in-the-tag naming, metadata-mismatch raise, `datasets.disable_caching()`) into vectormesh proper. A course notebook folder is the wrong home for the shared helper. | vectormesh | todo |
| **P3** | `VectorCache.join(other, on="source_idx")` — fusion needs two caches as one dataset with two columns; today that is a hand-rolled `add_column` plus a `with_format(None)` dance that silently fails when the format is torch. | vectormesh | **coded**, branch `feat/from-hub-join` (D12) |
| **P4** | Build + publish **EuroSAT** and **DTD** caches, 3 encoders each, with `source_idx`. | `make build-lesson1` / `make publish-lesson1` | **done** 2026-08-04 |
| **P6** | Fix `Concatenate2D`'s binding axis (`"batch dim"` → `"batch _"`) so branches of different widths concatenate, as its docstring already promises. §3.5 | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P7** | sdist shipped 50 MB in 0.4.4, almost all of it `references/` — four copyrighted papers republished on PyPI. Restricted to source/tests/docs: 50 MB → 48 KB. | vectormesh | **coded**, branch `feat/from-hub-join` |
| **P5** | Release vectormesh 0.5.0 (P1, P3, P6, P7) and bump the floor to `vectormesh>=0.5.0` in the course `pyproject.toml`. | both | **done** 2026-08-04 |

Known API limit, not blocking: `CollateParallel` takes exactly **two** columns. Three-way
fusion needs a library change; L1 stays at two.

## 4a. Round 3 additions (vectormesh 0.6.0)

Triggered by a detailed review pass on the round-2 notebooks (2026-08-05): tone (several
passages talked to the *author* about the *previous round* rather than to a student —
cut), missing visuals in several places, and one real bug (see P7a). vectormesh work
landed on branch `feat/cache-properties-neuralnet-depth`, merged and **published as
0.6.0** the same day; the course `pyproject.toml` floor is bumped to match.

| # | Item | Where | Status |
|---|---|---|---|
| **P2** | Promote `notebooks/4_attention/cache_utils.py`'s `get_or_create_cache` into vectormesh proper, as `VectorCache.get_or_create`. Flagged in round 2 (§4 above) and not done then because D13 meant nothing in L1 called it. Round 3's optional `01c_build_your_own_cache.ipynb` is the second, decisive reason. Kept deliberately separate from `create()` rather than merged into it: `create()` is `scripts/build_cache.py`'s method and several vectormesh docs/demo notebooks call it expecting an unconditional fresh build; `get_or_create` adds the orthogonal idempotent contract without changing `create()`'s existing meaning for anyone relying on it. | vectormesh | **done** 2026-08-05 |
| **P6a** | `VectorCache.num_classes` / `VectorCache.dim` properties — shortcuts for `features["label"].num_classes` / `metadata[<col>]["hidden_size"]`, which every L1 notebook re-derived by hand. Both raise `VectorMeshError` on the ambiguous cases (no label column, non-`ClassLabel` label, >1 vector column after a `join()`). | vectormesh | **done** 2026-08-05 |
| **P7a** | `NeuralNet(hidden_size: int \| list[int], out_size)` — fixes a real bug, not a style preference. `04_search.ipynb` and `05_exercises.ipynb` both built "add a hidden layer" by composing two `NeuralNet`s (`NeuralNet(dim, width)` then `NeuralNet(width, n_classes)`); `NeuralNet` ends in a bare `Linear` with no trailing activation, so that composition puts an unactivated `Linear -> Linear` seam in the middle — exactly the "depth without nonlinearity buys nothing" collapse §1.4/notebook-02 §2.1 proves. A bare int reproduces the old two-layer behaviour bit-for-bit (checked: every `NeuralNet(` call site across every lesson passes a bare int, so this is purely additive); `hidden_size=[dim, w1, ...]` names the whole input→hidden width chain in one call, with a `GELU` between every pair of `Linear`s so the seam cannot happen by construction. Very likely part of why §3.3's "proj→width→head" experiments below lost points — the composition itself was leaking capacity, not only the width choice; not re-measured, flagged for anyone revisiting §3.3. | vectormesh | **done** 2026-08-05 |

All three verified against a real published cache (`pttrn-io/eurosat-dinov2-small`), not
just synthetic test data, and against the full non-integration vectormesh test suite (86
passed). `notebooks/4_attention/cache_utils.py` becomes a thin re-export so lesson 5 is
untouched.

Also fixed at the notebook level, no library change needed (both confirmed by reading
`vectormesh/components/connectors.py` and `vectormesh/data/cache.py`, not assumed):
`Concatenate2D` already accepts a raw two-tensor tuple, so `04_search.ipynb`/
`05_exercises.ipynb`'s `Parallel([nn.Identity(), nn.Identity()])` before it was inert —
removed; `VectorCache.join()` already validates alignment and raises on mismatch (D12), so
the notebooks' manual `assert` after `.join()` was re-checking something the library
guarantees — removed.

## 4b. Round 4 addition (vectormesh 0.7.0) — SparseMoE

`05_gates_highways_moe.ipynb`'s dense `MoE` section carried an honest caveat: it tests
Jacobs & Jordan's (1991) dense blend, not the sparse top-k routing (Shazeer et al. 2017;
Switch Transformer, Fedus et al. 2021) that is why production MoE scales — dense MoE runs
every expert on every input, so it structurally cannot show capacity and compute
decoupling. Rather than leave the caveat standing, `vectormesh` gained a genuinely sparse
component and the notebook now measures the claim directly.

| # | Item | Where | Status |
|---|---|---|---|
| **P8** | `SparseMoE(experts, hidden_size, out_size, k=1)` — top-`k` routed mixture of experts. Routes via a `Linear` router, then a real gather/scatter: flatten the `(n, k)` routing decisions into `n*k` (example, expert, weight) assignments, sort by expert id so each expert's assignments are one contiguous slice (one `argsort`, not an `O(N)` scan per expert), run each expert only on its own slice, scatter the weighted outputs back with `index_add_`. An expert that received zero examples this batch is absent from the loop entirely — confirmed by a test asserting its parameters get no gradient at all (not a masked-to-zero one). `k=N` is checked to match dense `MoE`'s output exactly (same router/expert weights), tying the new path to the already-trusted one. 10 new tests, `tests/test_gating.py`. | vectormesh | **done** 2026-08-05, released 0.7.0 |

### 4b.1 Sparse vs dense, measured (not just claimed)

Benchmarked on `05_gates_highways_moe.ipynb`'s own data — a batch of 64 of its own
`(W=192,)` windows, projected to `hidden=64`, `k=1`, 30 reps after 5 warmup calls:

| N experts | sparse ms/fwd | dense ms/fwd | params |
|---|---|---|---|
| 4 | 0.129 | 0.117 | 16,900 |
| 16 | 0.204 | 0.224 | 67,600 |
| 64 | 0.250 | 0.535 | 270,400 |
| 256 | 0.262 | 1.783 | 1,081,600 |

From N=4 to N=256, parameters grow 64x. `SparseMoE`'s forward time grows only ~2x
(routing overhead — one `argsort` plus a per-active-expert Python loop — not added
compute, since every example still passes through exactly one expert). Dense `MoE` grows
~15-17x, close to proportional with N as it must be, since it evaluates every expert on
every example regardless of capacity. Dense is briefly *faster* at small N (routing
machinery costs more than it saves with nothing to route around); the crossover is
around N=16-64, and dense is 6-7x slower than sparse by N=256 for identical parameter
counts. This is the capacity/compute decoupling claim, measured on this course's own
data rather than asserted from the papers.

### 4b.2 Corrected: this doesn't answer whether routing forecasts better — that got checked too

The first version of this section stopped at the timing table above and left "does
sparse routing forecast better" untested — a real gap, caught by direct user pushback
("wasn't this about seeing if sparse actually gives better performance?"), not by this
plan's own review. Fixed by adding `sparse_moe (N=4, k=1)` and `sparse_moe (N=64, k=1)`
as two more trained-and-scored rows in `05_gates_highways_moe.ipynb`'s existing 5-seed,
`EPOCHS=90` table (§3.9's D19-settled point), not a separate speed-only demo:

| model | val MAE | params | vs dense `moe` |
|---|---|---|---|
| dense `moe` (N=4) | 0.16475 ± 0.01009 | 32,372 | — |
| `sparse_moe` (N=4, k=1) | 0.16960 ± 0.01129 | 32,372 (matched) | inside seed noise, not clearly worse |
| `sparse_moe` (N=64, k=1) | 0.17038 ± 0.00443 | 285,872 (8.8x moe, 18.5x plain) | indistinguishable from the N=4 sparse row |

Routing does not beat blending at matched capacity here, and 16x more capacity (same
active compute) bought nothing — both `sparse_moe` rows are the two worst trained models
in the table apart from `plain` itself. Checked *why*, not just left as a shrug: for one
seed (seed=0, 90 epochs), the trained 64-expert router used only **15 of 64 experts** on
the 361 validation windows (top expert 42% of examples, top five 90%) — router collapse,
a known real failure mode of top-1 routing with no load-balancing auxiliary loss (which
`SparseMoE` doesn't have, unlike production Switch/Mixtral). Named honestly in the
notebook (§5.4) as the likely cause, with a new exercise (E6) asking students to add a
load-balancing loss and re-check.

None of this reopens §4b.1's compute claim — the capacity/compute decoupling is still
real and still measured. It just means this specific forecasting task's answer to
"is sparse routing worth it" is no, for a checkable reason, same as every other gating
mechanism this notebook has tested (D19, §3.9) — one more real negative result, not an
exception carved out for the newest architecture.

### 4b.3 Also fixed: §5.2's epoch-count table gained an actual learning curve

Second direct user pushback, same session: "normally I would visualise the train-test
learning curves... why not visualise it like any human would?" `05_gates_highways_moe`
§5.2 previously only printed mean±std MAE at three fixed epoch counts (30/60/90) — real
evidence, but not the standard first diagnostic (is it still improving, has it
plateaued, is train pulling away from val). Added: a train/val MAE-vs-epoch plot
(seed=0, `gate` and `deeper`, matching `02_basemodels.ipynb`'s own existing
single-seed-curve convention) ahead of the multi-seed table. Both curves flatten by
~epoch 20-30 with a stable ~0.025 train/val gap that does not widen (no runaway
overfitting) — but the val curve visibly bounces epoch-to-epoch by more than the gap
between the two models, which is the visual version of exactly why the seed-and-
epoch-count check right after it exists.

### 4b.4 Extended: `06_search` gains sparse_moe, n_experts, expert_hidden, and the params-trained-vs-used tradeoff

Third direct user request, same session: search over how many experts and how wide each
expert should be, keep it set up for students to reuse on other datasets even if it
doesn't win here, and make the trained-vs-used parameter tradeoff and the specialisation
intuition explicit rather than implicit. `06_search.ipynb`'s existing search (mechanism,
hidden, n_experts, lr — n_experts already existed but only for dense `moe`, capped at 8)
gained: `sparse_moe` (k=1) as a mechanism choice; `n_experts` widened to
`{2,4,6,8,10,16}`; a new `expert_hidden` axis (`{16,32,64}`) that makes each expert a
genuine bottleneck 2-layer network (`Linear -> GELU -> Linear`) rather than
`05_gates_highways_moe`'s single-`Linear` experts, so the search can trade many small
experts against few large ones — the "N=10, but smaller" question directly, not one
point in that space. `param_counts()` reports (total, active) parameters per trial,
carried through `tune.report` into the results dataframe. N_TRIALS bumped 24 → 30 for
the larger space.

Measured, 30 trials, 148s: `sparse_moe` did not win, and did worse than every other
mechanism's own worst trial — only 4 of 30 trials (HyperOptSearch appears to have
deprioritised it early), ranging 0.2793 (best) to 0.3644 (worst), the widest spread of
any mechanism. Its best config (`n_experts=16, expert_hidden=64, hidden=128`) is the
concrete tradeoff this addition exists to make visible: **292,008 parameters trained,
43,368 (14.9%) used per input** — real, measured capacity/compute decoupling — that
still did not beat dense `moe`'s much smaller 92,576-parameter trial (mae 0.2706, every
parameter used every time). More capacity, spent more cheaply, was not automatically
*better* capacity here. Separately, the overall search winner shifted from the earlier
24-trial run's `plain` (0.2624) to `gate` (0.2596) — a 0.0028 edge, smaller than the
2-std band the baseline check exists to catch, and every trial trains from one fixed
seed, so this is flagged in-notebook (§6.3, E3) as unresolved rather than as a reversal.

The specialisation question this addition was written to make explicit — does pooling 8
pedestrian-count locations give a router something real to divide up — is not settled by
this run: TPE never spent much budget in the "many small experts" corner before moving
on. A new exercise (E4) asks students to test the hypothesis on a dataset with more
obvious heterogeneity (their own choice) rather than re-spending trials on this one,
since the search space itself (mechanism, hidden, n_experts, expert_hidden, lr) is
written to be reusable as-is.

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
| `imdb-granite-…` | text, chunked | granite-embedding-small | L4 |

---

## 5. Cuts register

| Cut | Was | Why |
|---|---|---|
| MNIST / Fashion-MNIST as the L1 spine | `1_pytorch_intro/05_classification` | D2 — solved a hardware problem the cache solves better |
| Hand-rolled data generators | `1_pytorch_intro/02_datagenerators` | Superseded by `Collate` + `DataLoader`; the streaming case is a legacy note |
| Hand-rolled dataloader (full notebook) | `1_pytorch_intro/03_dataloader` | Compressed to one section of the L1 training notebook |
| Double for-loop hyperparameter search | L1/L2 exercises | D5 — replaced by ray.tune from the start |
| `mads_datasets` | L3 notebooks, `pyproject.toml` | Everything loads from the HF hub now; the dependency is dead weight |
| `tune.run` legacy API | `3_tuning_networks/hypertune.py` | ray 2.55 ships `tune.Tuner` + `tune.TuneConfig` + `tune.RunConfig`; no `ray.train` import needed |
| Separate "transfer learning" notebooks | `3_tuning_networks/03`, `04` | Transfer learning *is* the L1 premise now; a whole lesson restating it is redundant |
| "Build your own cache + mlflow" as L2 (D9) | `lesson2.yml` round 2, `2_convolutions/02_pretrained.ipynb` + `04_exercises.md` | D14 — ray and mlflow both already land in L1 (notebook 01 §1.5, notebook 04) before this lesson would run; a whole lesson re-teaching them is redundant on arrival, not merely stale. `notebooks/2_convolutions/` moved to `legacy/` 2026-08-05 and the now-empty directory removed. |

**Cut, do not stage.** If material no longer reflects practice, remove it — do not invent
a scenario that keeps old teaching code alive.

---

## 6. Lesson map after the refactor

| Lesson | Spine | Vectorized? |
|---|---|---|
| L1 | Cache → nonlinearity → training → ray search | yes (EuroSAT / DTD) |
| L2 | Time series — windowing, decomposition + N-BEATS, probabilistic forecasting, gating/highway/MoE, ray search | **no** (D-rationale §1.2) |
| L3 | The search algorithms themselves; the full ray zoo | yes |
| L4 | Attention, long documents, chunked text | yes (IMDB / legal) |
| L5 | Self-supervised — where representations come from | yes |
| L6 | Graphs | **no** |

Folders renamed to match 2026-08-05: `3_recurrent_networks` → `2_timeseries`,
`4_tuning_networks` → `3_tuning_networks`, `5_attention` → `4_attention`,
`6_unsupervised` → `5_unsupervised`, `7_graphs` → `6_graphs`. `2_convolutions` removed
(cut, see `cuts` register above). `pyproject.toml`'s `extra-paths` updated to match.

Per-lesson detail in `lesson<N>.yml`. L1 and L2 are both fully specified — L2 because this
round rewrites it wholesale (D14–D17), not because it is new: the round-1
`3_recurrent_networks` rebuild supplies most of its actual notebook content, reordered and
partly re-targeted. L3–L6 record only what this round changes: L4 and L5 were rebuilt in
round 1 and need only change notes; L3 and L6 are round-3 rebuild work.
