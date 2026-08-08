# Evidence

Every number here was produced on this machine, not recalled. Several curriculum
decisions are not what was first intended because of them. **Anything added must be a
number someone actually ran**, with enough seeds to have the right to the conclusion.

Measured 2026-08-04/05 unless noted, vectormesh 0.4.4–0.7.0.

---

## 1. The representation is the model

Oxford Flowers-102, 4000 train / 1000 test, published caches, `NeuralNet(dim, 102)` head:

| encoder | dim | accuracy |
|---|---|---|
| dinov2-small | 384 | **0.9932** |
| mobilenet_v2 | 1280 | 0.9082 |
| resnet-18 | 512 | 0.9025 |

Total runtime for all three, CPU: **~9 seconds.**

Checked for leakage, because 99.3% on 102 classes deserves suspicion:

- train and test come from **different source splits** (`source_idx` 1–7168 vs 0–1019)
- test is balanced, 8–11 images/class; chance = 1.0%
- near-duplicates: 4/1000 above cosine 0.99, 45/1000 above 0.95 — cannot carry the number
- **resnet-18 gets 90% through the identical rows and head.** A leak would lift every
  encoder, not one.

The decisive statistic:

> **1-nearest-neighbour on raw DINOv2 vectors, zero training: 0.9910.**

The head contributes ~0.2 points. The best demo in the course for "the representation is
the model" — and the reason flowers is a single demo cell and cannot be a tuning target.

---

## 2. Dataset ceilings, and where the encoder axis is sharp

dinov2-small, 2000 train, two-line head:

| dataset | classes | accuracy | role |
|---|---|---|---|
| oxford-flowers | 102 | 0.9932 | saturated — demo cell only |
| **eurosat** | 10 | **0.9447** | **L1 demo** — impressive, not a lookup |
| resisc45 | 45 | 0.8209 | alternative |
| **dtd** (textures) | 47 | **0.7849** | **L1 exercises** — out-of-domain for an object-centric encoder |
| stanford_cars | 196 | 0.6034 | hard; 2000 train over 196 classes is near few-shot |

The encoder axis is **not** equally sharp everywhere. Full published caches, two-line head:

| encoder | eurosat (6000) | dtd (3760) | flowers (4000) |
|---|---|---|---|
| dinov2-small | **0.9502** | **0.8033** | **0.9932** |
| resnet-18 | 0.9380 | 0.6589 | 0.9025 |
| mobilenet_v2 | 0.9258 | — | 0.9082 |
| **spread** | **2.4 pt** | **14.4 pt** | **9.1 pt** |

**EuroSAT is a good demo but a poor search target** — its encoder axis is nearly flat, so
a search there shows *nothing* moving, which is not the intended lesson. L1's search
notebook therefore runs on DTD, where the 14-point spread makes "the encoder dominates"
legible. EuroSAT keeps notebooks 01–03, where the point is that the thing works at all.

DTD at 0.7849 with an object-centric encoder is also precisely the situation where you
would train your own representation — the hook L5 opens on.

---

## 3. Capacity added after a frozen encoder is wasted

Oxford Flowers, resnet-18 cache, sweeping the head:

| head | 5/class | full 4000 |
|---|---|---|
| **baseline `NeuralNet(512, 102)`** | **0.7502** | **0.9025** |
| + GaussianNoise σ=0.05 | 0.7602 | 0.9006 |
| proj→64→head | 0.5467 | 0.8197 |
| proj→256→head | 0.6250 | 0.8412 |
| proj→1024→head | 0.6432 | 0.8314 |
| Skip block | 0.7063 | 0.8887 |

Every added layer **loses**, by up to 8 points. The obvious confound is ruled out — jointly
sweeping lr on the full dataset, the baseline wins at **every** learning rate:

| head | lr=1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| baseline | 0.8635 | 0.8938 | **0.9025** | 0.8971 |
| proj→256→head | 0.7939 | 0.8420 | 0.8412 | 0.7992 |

**Confirmed at a proper seed budget.** DTD 47-class, 20 epochs, **8 seeds**, dinov2-small
+ resnet-18:

| setup | mean | std | Δ vs baseline | sigma | beats baseline |
|---|---|---|---|---|---|
| **baseline `NeuralNet(384, 47)`** | **0.8098** | 0.0024 | — | — | — |
| baseline + a hidden layer (2× params) | 0.7579 | 0.0086 | **−5.19 pt** | −15.4 | 0/8 |
| fusion, raw widths (384+512) | 0.8142 | 0.0126 | +0.43 pt | +0.9 | 7/8 |
| fusion, weak stream projected to 384 | 0.8132 | 0.0087 | +0.33 pt | +1.0 | 7/8 |

Depth loses badly and unambiguously: 5.2 points at 15 sigma, 0 seeds out of 8.

### 3a. Fusion is a wash, not a loss — and three seeds lied

An earlier measurement with a 128-wide bottleneck per branch showed fusion losing 5 times
out of 5:

| setup | best single | fusion (128 bottleneck) | Δ |
|---|---|---|---|
| flowers, dinov2+resnet | 0.9932 | 0.9902 | −0.30 |
| fashion-mnist, dinov2+mobilenet | 0.9113 | 0.9055 | −0.58 |
| flowers, resnet+mobilenet | 0.9082 | 0.8945 | −1.37 |
| flowers 5/class | 0.9912 | 0.9750 | −1.62 |
| flowers 2/class | 0.9604 | 0.9266 | −3.38 |

**That penalty was the bottleneck, not the fusion.** Squeezing a 384-wide stream through
128 throws away more than the second stream adds. At raw widths (table above) fusion wins
7 seeds of 8 by an amount inside its own noise: the honest verdict is *indistinguishable
from the baseline*, at 3× the parameters and **4–5× the seed variance** (std 0.013 vs
0.002). Written as "fusion loses", the first result was an over-generalisation of our own
measurement — exactly what `principles.md` §2 warns against, committed here.

The first run of the 8-seed experiment used **3 seeds** and reported +0.83 pt for
projected fusion; 8 seeds says +0.33. Neither is wrong; the first was under-powered. This
is a teaching cell in L1's search notebook, not a footnote.

The consequence: the fusion section poses a hypothesis the student predicts and tests, and
reports a wash with a variance cost — a more useful result than "the baseline wins", and a
better argument for measuring rather than believing.

### 3b. The DTD caches are grouped by label — subsampling must shuffle first

| cache | first 940 train rows contain | why |
|---|---|---|
| `eurosat-*` | all 10 of 10 classes | built with `--n-train 6000`, which shuffles before selecting |
| `dtd-*` | **12 of 47 classes** | built with `--full`, which does not shuffle at all |

`cache.dataset.select(range(n))` on DTD returns a **class-truncated** subset, not a smaller
sample of the same task. The failure is silent and looks like a learning curve: 470/940/1880
rows scored 0.121 / 0.238 / 0.437, which reads as "more data helps" but is actually
`n/3760` — the model scored ~100% on the only classes it had seen and 0% on the rest. With
`.shuffle(seed=0).select(range(n))` the same sizes give **0.694 / 0.753 / 0.782**, against
0.808 on the full split: a real, boring, correctly-shaped curve.

Consequences: any notebook that subsamples a cache shuffles first and says why (a good
teaching moment about looking at the data before slicing it), and `scripts/build_cache.py`'s
`--full` path should shuffle so the artefact stops carrying the trap.

### 3c. GaussianNoise does nothing measurable on DTD, at any training size

The ablation the bug above was hiding, rerun correctly (dinov2-small, 3 seeds, epochs
scaled so every size gets comparable gradient steps):

| train rows | σ=0.0 | σ=0.3 | Δ | train accuracy |
|---|---|---|---|---|
| 470 | 0.6940 | 0.6902 | −0.38 pt | 1.000 |
| 940 | 0.7527 | 0.7534 | +0.07 pt | 1.000 |
| 1880 | 0.7819 | 0.7837 | +0.18 pt | 1.000 |

The "+0.4 pt on DTD" reported earlier was inside noise and should not have been written as
a benefit. The informative column is the last one: the head reaches **100% training
accuracy at every size**, so it memorises completely, and a regulariser still moves the
test number by nothing. The ceiling is set by the frozen representation, not by head
capacity, so regularising the head cannot reach it.

**Implication for search space design:** a search over head width/depth/fusion will
rediscover the two-line baseline. That is a fine lesson only if designed for — put encoder
choice in the space, where the 10–40 pt differences live, and let the architecture axes
demonstrate that capacity added after a frozen encoder is nearly always wasted.

---

## 4. Encoding throughput, and why students do not do it

200 DTD images, Apple silicon:

| encoder | CPU | MPS | speedup |
|---|---|---|---|
| dinov2-small | 81 img/s | 245 img/s | 3.0× |
| resnet-18 | 95 img/s | 481 img/s | 5.1× |

This is a *fast* CPU. An older laptop or a basic cloud VM lands nearer 10–30 img/s. The
whole lesson-1 build (eurosat 8000 + dtd 5640 rows × 3 encoders) took **~90 s here** and
would take **most of an hour** on a weak CPU-only VM — repeated by every student, to
reproduce a byte-identical artefact that already exists. Hence the published-cache rule in
`principles.md` §3, and the ~200-image opt-in demo at ~2.5 s here / ~20 s on a weak VM.

---

## 5. Time series — dataset characterisation

Normalised series (z-scored on the full series): seasonal-naive residual RMSE at the
dataset's own daily period, and the fraction of spectral power in the top-5 FFT bins
(higher = more energy in a few clean periodic components).

| dataset | source | CoV | seasonal-naive RMSE (period) | spectral top-5/total |
|---|---|---|---|---|
| ETTh1 (OT) | `ETDataset/ett` | 0.62 | 0.34 (24h) | 0.77 |
| pedestrian_counts | Monash | 1.04 | 0.33 (24h) | 0.81 |
| **australian_electricity_demand** | Monash | **0.20** | 0.48 (48 half-hours) | 0.70 |
| traffic_hourly | Monash | 1.27 | 0.77 (24h) | 0.38 |
| solar_10_minutes | Monash | 1.30 | 0.56 (144 steps) | 0.65 |
| weather | Monash | 3.28 | 1.43 (144 steps) | — |

ETTh1 is **not** the noisiest candidate by either metric — smoother point-to-point than
pedestrian_counts, cleaner spectrum than traffic_hourly. Its reputation tracks a different
property: known regime drift between train and test spans (why DLinear beats transformer
forecasters on it in the literature), not instantaneous variance. That makes it a
confounded place to teach calibration, so `australian_electricity_demand` — CoV three
times tighter, still a real Monash benchmark — carries the probabilistic material.

ETTh1 was subsequently dropped from the graded sequence entirely: five datasets across nine
notebooks was real sprawl, and a six-covariate electricity-transformer domain added
confusion an unfamiliar dataset did not need.

### 5a. Sunspot — a real rotation period, verified

`sunspot_daily.py` starts at 1848-12-22 (last date with a missing day ahead of it) —
62,617 consecutive daily points, 171.4 years. Autocorrelation has a local maximum at lag
26–28 days (0.77, above its lag-14 and lag-60 neighbours at 0.68 and 0.70); FFT short-period
peaks cluster at 26.8–28.2 days. This is the Sun's ~27-day synodic rotation, a genuine
physical quasi-period, checked rather than assumed. Day-to-day persistence is still much
stronger (acf(1d)=0.96), so naive beats seasonal-naive(27d).

The largest peak in the *entire* daily spectrum sits at ~11 years — the Schwabe cycle,
visible even at daily resolution, ~160× the power of the 27-day peak, and far too slow for
any window the lesson holds. Hence the yearly zoom-out as a second look at the same series.

How many repeats does a 203-year span actually contain of each named solar cycle:

| cycle | period | repeats in 203yr | verdict |
|---|---|---|---|
| Schwabe | ~11yr | 18.5 | plenty — fits a fixed-period basis confidently |
| Gleissberg | ~87yr | 2.3 | visible, not pin-downable — an 11yr-rolling-max envelope shows real modulation (227→182→110→142→174→191→269→220→174, sampled every 20yr) but two-and-a-bit repeats will not fix the period |
| Suess/de Vries | ~210yr | 0.97 | not resolvable — under one full cycle |

So the notebook adds a **second** basis period (~87yr) alongside the Schwabe one, shows the
fit improve, and names the ~210yr cycle as genuinely out of reach given the span. "How much
data would you need" replaces "this doesn't work". Students met this series in MADS-DAV,
where `seasonal_decompose` fails on it and the closing notes hand the long-cycle problem
forward explicitly; L2 is that hand-off.

### 5b. A genuine N-BEATS win, found by searching for one

A notebook whose premise is "put the decomposition inside the architecture" had no dataset
where that architecture won on accuracy — linear beat N-BEATS on both `pedestrian_counts`
and `sunspot`. Manufacturing a win via data scarcity (the ETTh1 story: ~1,400 windows,
really an overfitting story about linear) was rejected. Every candidate below ran at a
window count comfortably above the linear model's own parameter count, so none of these can
be a scarcity flip:

| dataset | window (W, H, period) | pooled windows | result |
|---|---|---|---|
| `nn5_daily` | 35, 7, 7 | 3,394 (8 series) | linear wins: 0.6717 vs generic 0.6995 |
| `traffic_hourly` | 336, 24, 168 | 6,780 (4 series) | linear wins: 0.2787 vs generic 0.3750; seasonal-naive(168)=0.2192 beats both |
| `london_smart_meters` | 336, 24, 168 | 9,093 (4 series) | linear wins: 0.5935 vs generic 0.6529 |
| `tourism_monthly` | 24, 12, 12 | 2,420 (20 series) | linear wins clearly: 0.5705 vs generic 0.8211 |
| **`solar_10_minutes`** | 288, 144, 144 | **103,700 (20 series)** | **N-BEATS generic wins: 0.2177 vs linear 0.2523** |

Solar power is the one series checked where the daily period is close to a physical
certainty rather than a statistical regularity — exactly zero overnight, a smooth hump
during the day, governed by the sun's position rather than aggregate human behaviour. That
is the mechanistic reason a period-144 Fourier basis beats an unconstrained 41,616-parameter
linear map even with 103,700 training windows (2.5× the linear model's parameter count):
`nn.Linear(288,144)` must learn "step *t* resembles step *t*−144" independently at 144
output positions, while the seasonality stack cannot help assuming it once — every output is
built from one 20-number `theta` reused across a single continuous Fourier phase. Robustness
checked at 8/12/20 epochs × 5 seeds; the gap holds at 0.030–0.035 throughout.

At this larger window, N-BEATS generic's nominal edge on sunspot (0.3225 vs linear 0.3270)
is **not** a real result — the 0.0045 gap is smaller than N-BEATS's own seed-to-seed std
(0.0099). The honest read on sunspot is a **statistical tie**, re-measured at the new
window rather than carried over from the smaller one.

---

## 6. A time-series foundation model's embeddings are worse than the raw window

The contamination risk is real and was verified, not assumed: `Salesforce/lotsa_data`'s
file listing contains a `sunspot_with_missing` config, so sunspot carries the same
pretraining-overlap risk ETTh1 does.

Frozen-encoder collapse, real sunspot numbers, MAE ratio vs naive:

| representation | MAE ratio | verdict |
|---|---|---|
| raw window | 0.796 | best |
| embedding + raw | 0.925 | worse than raw alone |
| mean-pooled embedding | 1.855 | worse than naive |
| last-patch embedding | 1.910 | worst |

The rule this produces — **does the encoder compress or expand?** — is the sharpest test of
L1's thesis anywhere in the course, and the reason a frozen-encoder story does not transfer
to time series by analogy.

---

## 7. Gating ties its capacity-matched control — and a three-seed result that was wrong

Building the gating notebook on `australian_electricity_demand`, a first check (3 seeds,
one epoch count) found `Gate` beating its capacity-matched `deeper` control by a
real-looking margin (0.157 vs 0.176). A follow-up (5 seeds, three epoch counts: 50/90/130)
found `Gate` and `deeper` **statistically indistinguishable at every one of them** — the
first result was noise from an under-sampled check, exactly the trap §3a warns about.

The corrected result at `EPOCHS=90`: `deeper` and `gate` tie (0.159 vs 0.160); `highway`
and `moe` cost 1.8× and 2.1× the parameters without a margin clearing their own seed noise.
Both the wrong number and the correction stay in the notebook.

**Every epoch count must be justified by a checked convergence curve or a multi-seed,
multi-epoch-count sweep — never copied from another notebook.** The learning curve is the
standard first diagnostic and belongs ahead of the table: train/val curves flatten by epoch
20–30 with a stable ~0.025 gap that does not widen, but the val curve bounces
epoch-to-epoch by more than the gap between the two models — the visual version of why the
seed check exists.

---

## 8. Sparse vs dense MoE — capacity and compute do decouple

Batch of 64 `(W=192,)` windows, projected to `hidden=64`, `k=1`, 30 reps after 5 warmups:

| N experts | sparse ms/fwd | dense ms/fwd | params |
|---|---|---|---|
| 4 | 0.129 | 0.117 | 16,900 |
| 16 | 0.204 | 0.224 | 67,600 |
| 64 | 0.250 | 0.535 | 270,400 |
| 256 | 0.262 | 1.783 | 1,081,600 |

From N=4 to N=256 parameters grow 64×. `SparseMoE`'s forward time grows only ~2× (routing
overhead, not added compute — every example still passes through exactly one expert). Dense
`MoE` grows ~15–17×, close to proportional with N as it must be. Dense is briefly *faster*
at small N; the crossover is around N=16–64, and dense is 6–7× slower by N=256 at identical
parameter counts. This is the capacity/compute decoupling claim, measured on the course's
own data rather than asserted from the papers.

### 8a. It does not forecast better, and the reason is checkable

5 seeds, `EPOCHS=90`:

| model | val MAE | params | vs dense `moe` |
|---|---|---|---|
| dense `moe` (N=4) | 0.16475 ± 0.01009 | 32,372 | — |
| `sparse_moe` (N=4, k=1) | 0.16960 ± 0.01129 | 32,372 (matched) | inside seed noise |
| `sparse_moe` (N=64, k=1) | 0.17038 ± 0.00443 | 285,872 (8.8× moe) | indistinguishable from N=4 |

Routing does not beat blending at matched capacity, and 16× more capacity at the same
active compute bought nothing. Checked *why* rather than shrugged: at seed 0, the trained
64-expert router used only **15 of 64 experts** across 361 validation windows (top expert
42% of examples, top five 90%) — router collapse, a known failure mode of top-1 routing
without a load-balancing auxiliary loss, which `SparseMoE` does not have (unlike
Switch/Mixtral). Named in the notebook, with an exercise asking students to add one and
re-check.

This does not reopen §8's compute claim. It means this task's answer to "is sparse routing
worth it" is **no, for a checkable reason** — one more real negative result, not an
exception carved out for the newest architecture.

---

## 9. Params trained vs params used — the tradeoff, made visible

Search over mechanism / hidden / n_experts / expert_hidden / lr, 30 trials, 148 s, on
`solar_10_minutes`. `param_counts()` reports (total, active) per trial.

`sparse_moe` did not win, and did worse than every other mechanism's own worst trial —
TPE gave it only 4 of 30 trials, ranging 0.2793 to 0.3644, the widest spread of any
mechanism. Its best config (`n_experts=16, expert_hidden=64, hidden=128`) is the concrete
tradeoff: **292,008 parameters trained, 43,368 (14.9%) used per input** — real, measured
capacity/compute decoupling — which still did not beat dense `moe`'s much smaller
92,576-parameter trial (MAE 0.2706, every parameter used every time). More capacity, spent
more cheaply, was not automatically *better* capacity.

On the same search, every routing mechanism's best trial beat `plain`'s best trial —
`gate` 0.2079, `moe` 0.2095, `sparse_moe` 0.2099, `highway` 0.2102 against `plain`'s
0.2214 — a 0.0135 gap for `gate`, over 6× the 5-seed baseline's own std (0.0021). This had
not happened on the electricity window, where gating tied throughout (§7). The mechanism
that wins is a property of the series, not of the mechanism.

The specialisation question — does pooling 20 solar sites give a router something real to
divide up — is **not settled**: TPE never spent much budget in the "many small experts"
corner. The search space is written to be reusable as-is, and an exercise asks students to
test it on a dataset with more obvious heterogeneity rather than re-spending trials here.

---

## 10. You cannot rank encoders by the geometry of their embedding cloud

Measured 2026-08-08 on the published EuroSAT caches, CPU, seconds. Independently re-run
rather than taken from the research note; script in the session scratchpad,
`verify_rankme.py`, to be checked in with the notebook that uses it.

RankMe (`exp` of the entropy of the normalised singular-value spectrum) against a
closed-form ridge probe on the same rows:

| encoder | dim | RankMe | probe | RankMe after whitening | probe after whitening |
|---|---|---|---|---|---|
| dinov2-small | 384 | 265.1 | **0.9460** | 384.0 | 0.9467 |
| resnet-18 | 512 | 328.4 | 0.9180 | 512.0 | 0.9180 |
| mobilenet_v2 | 1280 | 822.3 | 0.8873 | 1280.0 | 0.8873 |

**RankMe ranks these encoders exactly backwards.** And it is not simply reading off the
ambient dimension, though that is part of it — pure Gaussian noise scores higher than
every real encoder at matched width (380.9 / 506.4 / 1243.5 at 384 / 512 / 1280).

The decisive control is whitening. It drives RankMe to `dim` for all three — its
theoretical maximum — while the probe moves by at most 0.0007. That is the argument, and
it is structural rather than empirical: **a linear probe is invariant to invertible
linear maps and RankMe is not**, so RankMe cannot be measuring what the probe measures.
Any unsupervised metric with the same non-invariance fails the same way.

The lesson is not "RankMe is bad". It answers a different question — *has this encoder
collapsed?* — which is a training diagnostic for one model, not a selection criterion
across models. Fitting the actual probe costs about a second per cache and answers the
question you asked. Further metrics, and two that do correlate (intrinsic dimension,
label-free; and same-minus-different-class cosine, which needs labels and is therefore a
probe with extra steps), in `research/embedding_space_measurement.md` — **not yet
independently verified.**

---

## 11. A supervised encoder specialises, but only if you train it long enough

EuroSAT at 32x32, small conv encoder (DIM=128), five classes seen / five held out, linear
probe on each half. 3 seeds, one run per seed probed at checkpoints. Measured 2026-08-08.

| epochs | seen | unseen | gap |
|---|---|---|---|
| 4 | 0.8848 ± 0.0009 | 0.8873 ± 0.0021 | **-0.0025 ± 0.0024** |
| 8 | 0.9102 ± 0.0028 | 0.9036 ± 0.0080 | +0.0066 ± 0.0096 |
| 16 | 0.9256 ± 0.0021 | 0.9043 ± 0.0046 | +0.0212 ± 0.0066 |
| 24 | 0.9350 ± 0.0017 | 0.9094 ± 0.0080 | +0.0256 ± 0.0065 |
| 40 | 0.9455 ± 0.0040 | 0.9128 ± 0.0084 | +0.0326 ± 0.0124 |
| 60 | 0.9579 ± 0.0037 | 0.9088 ± 0.0010 | **+0.0491 ± 0.0044** |

Seen accuracy climbs the whole way, 0.885 to 0.958. **Unseen accuracy plateaus near 0.91
after epoch 24.** Past that point every epoch buys accuracy on the training classes and
nothing on the others — supervised specialisation, visible from outside the model.

**The epoch count changed the sign of the conclusion.** The first draft of L5's opener
trained for 12 epochs, measured a gap of +0.007 against a seed std of ±0.010, and wrote
that supervised representations transfer for free. At 60 epochs the gap is +0.049 at
11 sigma. The 12 was chosen, not justified, and the sweep only happened because the
instructor challenged it. Both the wrong reading and the correction are in the notebook.

For contrast, on the same data the autoencoder scores ~0.86 on the unseen half having used
**zero** labels, against supervised's ~0.91 from ~3,100. The case for self-supervision
here is the label budget plus the plateau, not a transfer failure.

---

## 12. Published caches (hf.co/pttrn-io)

Every one carries `label`, `source_idx`, `embed` and a dataset card pinning the
source-dataset and encoder revisions.

| dataset | rows | encoders | measured two-line-head accuracy |
|---|---|---|---|
| `eurosat-*` | 6000 / 2000 | dinov2-small, resnet-18, mobilenet_v2 | 0.9502 / 0.9380 / 0.9258 |
| `dtd-*` | 3760 / 1880 | dinov2-small, resnet-18, mobilenet_v2 | 0.8033 / 0.6589 / — |
| `oxford-flowers-*` | 4000 / 1000 | dinov2-small, resnet-18, mobilenet_v2 | 0.9932 / 0.9025 / 0.9082 |
| `fashion-mnist-*` | 4000 / 1000 | dinov2-small, mobilenet_v2 | 0.9113 / 0.8678 |
| `imdb-granite-…` | text, chunked | granite-embedding-small | L4 |

Build and publish with `make build-lesson1` / `make publish-lesson1`.
