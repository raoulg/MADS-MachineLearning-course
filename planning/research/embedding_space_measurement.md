# Training-free geometry of frozen-encoder embeddings

Can a student rank encoders by a training-free geometric metric computed on the cached
embedding matrix, and does that ranking match linear-probe accuracy?

Everything below labelled MEASURED was produced by running code against the actual
`pttrn-io` caches on this machine. Claims taken from papers are labelled PAPER and are
never used as evidence for a ranking.

## 0. Setup and gotchas (MEASURED)

Each cache is loaded train+test, `.shuffle(seed=0)` **first** (the DTD caches are stored
grouped by label, so `select(range(n))` truncates classes), then `embed` is converted to
a float64 `(n, dim)` matrix. Spectral metrics use all rows; pairwise and kNN metrics use
a fixed random subsample of 2000 rows. All nine caches complete in well under a minute
on CPU once downloaded.

Repo names differ from the brief: the mobilenet caches are
`pttrn-io/<dataset>-mobilenet_v2_1.0_224`, and **`pttrn-io/dtd-mobilenet_v2_1.0_224`
does exist** (the brief lists DTD as dinov2/resnet only).

A closed-form ridge probe (one-hot targets, alpha=1, centred + globally scaled) was run
as a consistency check on the ranking target:

| dataset | encoder | course probe | ridge probe (MEASURED) |
|---|---|---|---|
| eurosat | dinov2-small | 0.9502 | 0.9490 |
| eurosat | resnet-18 | 0.9380 | 0.9340 |
| eurosat | mobilenet_v2 | 0.9258 | 0.9125 |
| dtd | dinov2-small | 0.8033 | 0.7755 |
| dtd | resnet-18 | 0.6589 | 0.6500 |
| dtd | mobilenet_v2 | (not measured before) | 0.6351 |
| flowers | dinov2-small | 0.9932 | 0.9910 |
| flowers | resnet-18 | 0.9025 | 0.8310 |
| flowers | mobilenet_v2 | 0.9082 | 0.8500 |

The ridge probe reproduces the course ranking on all three datasets, so it is a valid
cheap stand-in, and DTD's third encoder can be included. Target orders:

- eurosat: dinov2 > resnet-18 > mobilenet_v2
- dtd: dinov2 > resnet-18 > mobilenet_v2
- flowers: dinov2 > mobilenet_v2 > resnet-18

## 1. Raw numbers (MEASURED)

Spectral metrics on the **centred** matrix. `RankMe` = exp(Shannon entropy of the
normalized singular values). Note this is *the same formula* as Roy & Vetterli effective
rank; RankMe adds an eps and applies it to a minibatch of an SSL encoder. Reporting both
separately would be reporting the same number twice.

| cache | dim | RankMe | RankMe/dim | stable rank | part. ratio | 95%-var dims | PC1 var frac |
|---|---|---|---|---|---|---|---|
| eurosat-dinov2-small | 384 | 265.2 | 0.690 | 5.51 | 18.4 | 183 | 0.181 |
| eurosat-resnet-18 | 512 | 328.5 | 0.642 | 4.93 | 16.4 | 197 | 0.203 |
| eurosat-mobilenet_v2 | 1280 | 823.0 | 0.643 | 7.39 | 29.3 | 514 | 0.135 |
| dtd-dinov2-small | 384 | 298.6 | 0.778 | 18.61 | 68.9 | 234 | 0.054 |
| dtd-resnet-18 | 512 | 362.1 | 0.707 | 14.64 | 56.8 | 244 | 0.068 |
| dtd-mobilenet_v2 | 1280 | 946.9 | 0.740 | 24.58 | 122.3 | 705 | 0.041 |
| flowers-dinov2-small | 384 | 248.9 | 0.648 | 13.96 | 44.5 | 155 | 0.072 |
| flowers-resnet-18 | 512 | 340.4 | 0.665 | 12.70 | 43.1 | 213 | 0.079 |
| flowers-mobilenet_v2 | 1280 | 825.7 | 0.645 | 15.19 | 58.4 | 530 | 0.066 |

| cache | mean cos | cos same-class | cos diff-class | cos gap | alignment | uniformity | TwoNN | MLE k=20 |
|---|---|---|---|---|---|---|---|---|
| eurosat-dinov2-small | 0.384 | 0.582 | 0.362 | 0.220 | 0.836 | -2.197 | 19.0 | 14.4 |
| eurosat-resnet-18 | 0.563 | 0.723 | 0.545 | 0.177 | 0.554 | -1.620 | 21.4 | 13.9 |
| eurosat-mobilenet_v2 | 0.524 | 0.660 | 0.509 | 0.151 | 0.680 | -1.788 | 26.6 | 16.9 |
| dtd-dinov2-small | 0.085 | 0.357 | 0.079 | 0.279 | 1.285 | -3.497 | 12.2 | 8.9 |
| dtd-resnet-18 | 0.452 | 0.568 | 0.450 | 0.119 | 0.863 | -2.130 | 15.0 | 12.5 |
| dtd-mobilenet_v2 | 0.269 | 0.378 | 0.267 | 0.111 | 1.244 | -2.866 | 17.6 | 17.1 |
| flowers-dinov2-small | 0.089 | 0.691 | 0.081 | 0.610 | 0.618 | -3.405 | 13.9 | 5.6 |
| flowers-resnet-18 | 0.698 | 0.802 | 0.696 | 0.106 | 0.395 | -1.167 | 17.9 | 15.6 |
| flowers-mobilenet_v2 | 0.492 | 0.658 | 0.490 | 0.168 | 0.683 | -1.949 | 20.5 | 16.1 |

Alignment = mean squared distance between same-class pairs of L2-normalized vectors
(lower = better, Wang & Isola). Uniformity = log E exp(-2 d^2) over all pairs
(lower = more uniform). Cos gap = mean same-class cosine minus mean different-class
cosine. TwoNN is fitted on the lower 90% of the log-ratio CDF with zero-distance
duplicates dropped: 4 duplicate points in dtd-resnet-18 and 2 in
flowers-mobilenet_v2 make r2/r1 infinite and must be removed or the estimate is NaN.

## 2. Does each metric reproduce the probe ranking? (MEASURED)

Score = number of correctly ordered encoder pairs out of 3 per dataset, 9 total.
Chance is 4.5/9.

| metric | direction assumed | eurosat | dtd | flowers | total |
|---|---|---|---|---|---|
| cos gap (uses labels) | higher better | 3/3 | 3/3 | 3/3 | **9/9** |
| TwoNN intrinsic dim | lower better | 3/3 | 3/3 | 2/3 | 8/9 |
| MLE intrinsic dim | lower better | 2/3 | 3/3 | 2/3 | 7/9 |
| mean pairwise cosine | lower better | 2/3 | 2/3 | 3/3 | 7/9 |
| uniformity | lower better | 2/3 | 2/3 | 3/3 | 7/9 |
| RankMe / dim | higher better | 2/3 | 2/3 | 1/3 | 5/9 |
| stable rank | higher better | 1/3 | 1/3 | 2/3 | 4/9 |
| participation ratio | higher better | 1/3 | 1/3 | 2/3 | 4/9 |
| **RankMe (raw)** | higher better | **0/3** | **0/3** | **1/3** | **1/9** |
| alignment | lower better | 1/3 | 1/3 | 1/3 | 3/9 |

Two results stand out and both are negative in the useful sense:

**Raw RankMe is almost exactly backwards.** 1/9 correct pairs — a metric that scored 0/9
would be a perfect inverted predictor. On every dataset it ranks
mobilenet_v2 > resnet-18 > dinov2-small, which is the reverse of the probe on eurosat
and dtd and near-reverse on flowers. The previous partial run's signal is **confirmed
and it replicates on all three datasets.**

**Alignment alone is worse than chance (3/9).** It says resnet-18 has the best space on
all three datasets, because resnet-18's embeddings are the most anisotropic
(mean pairwise cosine 0.56 / 0.45 / 0.70) so *every* pair is close, same-class pairs
included. Alignment without a same-vs-different baseline measures the cone, not the
clustering. Subtracting the baseline (that is exactly what `cos gap` does) takes it
from 3/9 to 9/9.

## 3. Is "RankMe backwards" just the dimension confound? (MEASURED)

Three controls.

**(a) Gaussian null.** RankMe of an iid N(0,1) matrix with n=6000:

| dim | RankMe of pure noise | RankMe/dim | TwoNN of pure noise |
|---|---|---|---|
| 384 | 380.9 | 0.992 | 108.1 |
| 512 | 506.5 | 0.989 | 128.4 |
| 1280 | 1244.4 | 0.972 | 208.0 |

Pure noise scores higher on RankMe than every real encoder. RankMe is bounded above by
the ambient dimension and structureless data nearly saturates it, so a raw RankMe
comparison across 384 / 512 / 1280 dimensions is close to a comparison of the
dimensions. (TwoNN of noise is 108-208, an order of magnitude above the 9-27 the real
caches produce, so the caches are genuinely low-dimensional manifolds.)

**(b) PCA-truncate every encoder to its top 384 components.** This is *not* a neutral
control — dinov2 loses nothing, mobilenet loses 70% of its directions, and the discarded
tail is exactly where its flat spectrum lived — but it is the obvious first thing to try:

| cache | RankMe@384 | stable rank@384 | TwoNN@384 | ridge probe@384 |
|---|---|---|---|---|
| eurosat-dinov2-small | 265.2 | 5.51 | 20.14 | 0.9490 |
| eurosat-resnet-18 | 262.6 | 4.88 | 20.18 | 0.9340 |
| eurosat-mobilenet_v2 | 294.5 | 6.85 | 23.35 | 0.9085 |
| dtd-dinov2-small | 298.6 | 18.61 | 10.73 | 0.7755 |
| dtd-resnet-18 | 291.8 | 14.44 | 14.68 | 0.6351 |
| dtd-mobilenet_v2 | 330.1 | 21.17 | 14.87 | 0.6410 |
| flowers-dinov2-small | 248.9 | 13.96 | 13.39 | 0.9910 |
| flowers-resnet-18 | 275.1 | 12.56 | 18.77 | 0.8070 |
| flowers-mobilenet_v2 | 304.7 | 13.94 | 17.45 | 0.8170 |

RankMe goes from 1/9 to **4/9** — it stops being an inverted predictor and becomes an
uninformative one (chance is 4.5/9). TwoNN stays at **8/9**.

**(c) Random-project every encoder to 384 dimensions.** A random Gaussian projection is
the neutral dimension match: it does not preferentially keep high-variance directions.

| cache | RankMe@rp384 | TwoNN@rp384 | ridge probe@rp384 |
|---|---|---|---|
| eurosat-dinov2-small | 214.7 | 19.36 | 0.9495 |
| eurosat-resnet-18 | 226.3 | 20.04 | 0.9370 |
| eurosat-mobilenet_v2 | 281.8 | 25.25 | 0.8895 |
| dtd-dinov2-small | 242.3 | 10.93 | 0.7739 |
| dtd-resnet-18 | 250.5 | 15.00 | 0.6356 |
| dtd-mobilenet_v2 | 315.0 | 17.29 | 0.6186 |
| flowers-dinov2-small | 203.4 | 12.94 | 0.9910 |
| flowers-resnet-18 | 237.3 | 18.65 | 0.8000 |
| flowers-mobilenet_v2 | 289.5 | 19.71 | 0.7750 |

With the dimensions genuinely matched, **RankMe scores 0/9 and TwoNN scores 9/9.**
RankMe is a *perfect inverted predictor* of the probe, and intrinsic dimension is a
perfect one. Random projection to 384 costs the probe almost nothing (0.9490 -> 0.9495,
0.9910 -> 0.9910), so the ranking target is unchanged.

**(d) Within-encoder control, no projection at all.** Comparing one encoder across the
three datasets holds the dimension exactly fixed:

| encoder | dtd | flowers | eurosat |
|---|---|---|---|
| dinov2-small: RankMe / probe | 298.6 / 0.776 | 248.9 / 0.991 | 265.2 / 0.949 |
| resnet-18: RankMe / probe | 362.1 / 0.650 | 340.4 / 0.831 | 328.5 / 0.934 |
| mobilenet_v2: RankMe / probe | 946.9 / 0.635 | 825.7 / 0.850 | 823.0 / 0.913 |

All three rows are strictly anti-monotone: within a fixed dimension, higher RankMe means
lower probe accuracy, 9/9 pairs. (Caveat: across datasets the accuracy differences are
mostly task difficulty, not encoder quality, so this shows RankMe is not tracking
separability rather than that it inverts encoder quality.)

**Verdict on the previous run's signal: confirmed, and it is not merely a dimension
artifact.** The dimension confound is real and large — it is what makes raw RankMe
*monotone* in dim — but after a neutral dimension match the anti-correlation is not
weakened, it is sharpened to 0/9. Both halves are worth teaching.

## 4. Why no unsupervised geometric metric can work (MEASURED)

This is the result that turns a pile of tables into a lesson. A linear probe is invariant
to any invertible linear map of the embedding space: if `W` is the probe and `A` is
invertible, `WA^-1` on `XA` produces the same logits. None of these metrics is.

`eurosat-dinov2-small`, identical data, three transforms, ridge probe with alpha=1e-3:

| transform | RankMe | stable rank | mean cos | alignment | uniformity | cos gap | TwoNN | probe |
|---|---|---|---|---|---|---|---|---|
| raw | 265.2 | 5.51 | +0.388 | 0.838 | -2.196 | +0.215 | 21.04 | **0.9490** |
| random rotation Q | 265.2 | 5.51 | +0.382 | 0.845 | -2.210 | +0.218 | 19.92 | **0.9490** |
| invertible A (QD, lognormal D) | 123.8 | 3.99 | +0.432 | 0.772 | -1.935 | +0.203 | 16.60 | **0.9490** |
| whitening | 383.2 | 383.0 | +0.000 | 1.953 | -3.973 | +0.026 | 135.65 | **0.9485** |

`eurosat-resnet-18`:

| transform | RankMe | stable rank | mean cos | alignment | uniformity | cos gap | TwoNN | probe |
|---|---|---|---|---|---|---|---|---|
| raw | 328.5 | 4.93 | +0.561 | 0.553 | -1.629 | +0.181 | 20.08 | **0.9340** |
| random rotation Q | 328.5 | 4.93 | +0.561 | 0.548 | -1.627 | +0.183 | 20.55 | **0.9340** |
| invertible A | 133.0 | 5.56 | +0.565 | 0.573 | -1.513 | +0.165 | 15.69 | **0.9340** |
| whitening | 512.1 | 512.0 | +0.001 | 1.930 | -3.961 | +0.038 | 121.41 | **0.9340** |

Read the last row of each table. **Whitening simultaneously maximizes every unsupervised
metric on the list** — RankMe hits the ambient dimension, stable rank hits the ambient
dimension, mean pairwise cosine hits zero (perfect isotropy, cone gone), uniformity hits
its floor — and the probe accuracy does not move by a thousandth. Whitening also makes
resnet-18 (512.1) beat dinov2 (383.2) on RankMe by an even wider margin than before,
while leaving the accuracy gap exactly where it was.

So: **the answer to "does whitening change the probe result" is no, and that is the whole
argument.** Any quantity that whitening can move to its optimum while the probe stands
still cannot be a measure of linear separability.

Two honest corrections this control forces:

- **TwoNN is not principled either.** Intrinsic dimension is invariant to invertible
  linear maps only in the infinite-sample, noiseless limit. Empirically it is not:
  whitening drives it 21 -> 136, because whitening rescales near-null noise directions up
  to unit variance and fills the space. Its 8-9/9 record is an empirical regularity on
  *natural, unmodified* encoder outputs, not a theorem. It is the best label-free metric
  measured here and it is still breakable.
- **The supervised `cos gap` is breakable the same way** (0.215 -> 0.026 under whitening,
  probe unchanged). Its 9/9 is likewise an empirical result on natural spaces.

## 5. Per-metric summary

| metric | measures | cost on 6000x1280 | ranking match | verdict |
|---|---|---|---|---|
| RankMe / effective rank | flatness of the singular-value spectrum; detects dimensional collapse | one SVD, ~1 s | 1/9 raw, 0/9 dim-matched | **teach as the negative result.** Inverted predictor; confounded by dim; maximized by whitening |
| stable rank | sum(s^2)/s_max^2 | free after the SVD | 4/9 | too noisy; whitening sets it to `dim` |
| participation ratio | (sum lam)^2 / sum lam^2 | free after the SVD | 4/9 | same failure as stable rank; carries no extra information |
| PC1 variance fraction | dominance of the top direction | free after the SVD | tracks dim inversely | descriptive only |
| alignment (Wang & Isola) | mean same-class distance | O(n^2 d) on a 2000 subsample, ~0.3 s | 3/9 | **worse than chance alone**; only meaningful against the different-class baseline |
| uniformity (Wang & Isola) | log E exp(-2 d^2) over all pairs | same pass | 7/9 | duplicates the anisotropy signal; picks dinov2 first every time, muddles 2nd/3rd |
| mean pairwise cosine (cone) | anisotropy | same pass | 7/9 | good *descriptive* hook, not a ranker; the cone is real (resnet 0.70 on flowers) but the probe does not care |
| cos gap (same minus different) | class separation | same pass | **9/9** | best predictor measured, but it uses labels, so it is a cheap probe, not a training-free one |
| TwoNN | intrinsic dimension from 2 nearest neighbours | kNN on 2000 points, ~0.2 s | 8/9 raw, 9/9 dim-matched | **best label-free metric measured**; needs duplicate handling and 90% CDF truncation |
| MLE (Levina-Bickel, k=20) | intrinsic dimension | same kNN | 7/9 | noisier than TwoNN, no advantage |

Numerics that actually bit: `dtd-resnet-18` has 4 exact duplicate embeddings and
`flowers-mobilenet_v2` has 2; without dropping them TwoNN returns NaN. The CDF fit must
be truncated (90% used here) or the tail dominates. And the DTD `.shuffle(seed=0)`
gotcha is real — without it a 2000-row subsample of DTD covers roughly the first 25 of
47 classes and every label-aware number is wrong.

## 6. Notebook proposal

**Title: what you cannot learn from the shape of an embedding cloud.**

Cell-by-cell, all CPU, all from `VectorCache.from_hub`, no training:

1. Load the three eurosat caches. Print `dim`. Note 384 / 512 / 1280 — plant the confound
   before anyone computes anything.
2. Implement RankMe in four lines (`s = svd(X - X.mean(0)); p = s/s.sum(); exp(entropy(p))`).
   Compute it on the three caches. **Students should find 265 / 328 / 823 — the exact
   reverse of the probe ranking they already measured in the previous lesson.** Let that
   sit.
3. Compute RankMe of `np.random.randn(6000, d)` for d in 384/512/1280: 381 / 507 / 1244.
   Pure noise wins. This is where the confound becomes visible without being told.
4. Dimension-match with a random projection to 384 and recompute: 215 / 226 / 282, still
   backwards, and the probe barely moves. The confound was real but it was not the
   explanation.
5. The whitening cell. Whiten each cache; recompute RankMe (-> `dim` for everyone),
   stable rank (-> `dim`), mean pairwise cosine (-> 0.000); rerun the probe (unchanged to
   four decimals). **This is the punchline cell.** Every unsupervised metric goes to its
   theoretical optimum and nothing about the task gets easier.
6. State the invariance argument in one markdown cell: the probe is invariant to
   invertible linear maps; these metrics are not; therefore they cannot measure what the
   probe measures. Spectral metrics answer "has this encoder collapsed?", which is a
   *training* diagnostic for a *single* model, not a *selection* criterion across models.
7. Two things that do work, with their limits: TwoNN (label-free, 8/9, and honestly
   flagged as empirical — show it break under whitening) and cos gap (9/9 but uses
   labels, so it is a cheap probe with extra steps). Optional exercise: run both on DTD
   and flowers to confirm they hold, and on the deliberately-broken whitened space to
   confirm they do not.
8. Exercise: hand students `dtd-*` without the `.shuffle(seed=0)` and let the
   class-truncated subsample produce a nonsense cos gap. Cheap, memorable lesson about
   how cached data is stored.

**Honest takeaway to put in the closing cell.** There is no free lunch: you cannot rank
frozen encoders for your task by looking at the geometry of their output cloud. The
metrics that look most principled (RankMe, stable rank, uniformity) rank these encoders
backwards or at chance, and can be driven to their optimum by a linear transform that the
downstream classifier is blind to. The metrics that do work either need labels (cos gap —
at which point just fit the probe, it takes seconds on a cache) or are empirical
regularities with no invariance guarantee (intrinsic dimension). Fitting the actual
linear probe on the cached embeddings costs about a second per cache and answers the
question you actually asked. **Measure the thing you care about.**

This is a good fit for the course's negative-results stance: the notebook produces a
crisp, reproducible, falsifiable failure rather than a leaderboard number.

## 7. Provenance: verified vs. recalled

**Verified by running code this session** (all of section 0-5): every number in every
table; the RankMe inversion on eurosat, dtd and flowers; the Gaussian null; the PCA and
random-projection dimension controls; the within-encoder control; the invariance /
whitening experiment; the duplicate-point and DTD-shuffle numerics; the ridge probe
agreeing with the course's trained-probe ranking on all three datasets.

**Recalled from training data, NOT re-read or fetched this session** (no paper was
opened; treat as attribution, not as evidence):

- RankMe, Garrido et al. 2023 — proposed for label-free *hyperparameter and checkpoint
  selection within one SSL method*, where the failure mode being detected is dimensional
  collapse. My reading of the measurements above is that it is being used out of
  distribution here: none of these three encoders is collapsed, and past collapse a
  flatter spectrum stops being good news. I did not verify what the paper claims about
  cross-architecture use.
- Alignment and uniformity, Wang & Isola 2020 — defined for contrastive learning with
  *augmentation* pairs as positives. Using class labels as positives, as done here, is an
  adaptation, not what the paper measured.
- Effective rank, Roy & Vetterli 2007 — same exp-entropy-of-singular-values formula as
  RankMe.
- TwoNN, Facco et al. 2017; MLE intrinsic dimension, Levina & Bickel 2004.

No claim in sections 2-5 depends on any of these attributions being right.
