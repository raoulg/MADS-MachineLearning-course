# Evidence for the lesson 2 slide preview

The figures support goals 2.1 (skip), 2.6 (gate), 2.7 (inductive bias) and 2.11
(defend a design). Their measurements are in `measurements.json`.

## Reproduction

Render the figures from the saved measurements:

```sh
uv run scripts/lesson2_evidence.py
```

Measure the fixed designs on the full train and validation splits and render:

```sh
uv run --with wfdb scripts/lesson2_evidence.py --measure
```

The raw MIT-BIH files must be in `artefacts/sources/mitdb/` (the source of
`scripts/build_heartbeat_cache.py`). The evidence script reads the model definitions
from `02_state_and_gates.ipynb` and uses the published vectormesh installation.

The deep comparison uses 10 epochs, the gate comparison 12; both have three repeated
runs. Each run's score is validation macro recall averaged over its final three epochs.
The plotted whiskers show the sample standard deviation of those run scores. They are
not confidence intervals. The comparisons use the same training protocol and data,
not equal computation. Validation holds records 118, 124, 201, 207, 215 and 223.
The dataset revision is recorded in the JSON. Test is not loaded.

The initial spread is the standard deviation across 256 training beats, averaged over
positions and channels. The gradient is the norm over a block's parameter gradients
for that batch before training. Neither is a repeated-training estimate. The two gate
maps come from the first trained gated model, on one N and one V validation beat;
white cells mask padding or features whose magnitude is in the lowest quartile.

## The S example

The signal is raw MLII from MIT-BIH record 209, sampled at 360 Hz, from 1510 to 1520
seconds. The target R annotation is sample 545731, symbol `A` (atrial premature beat),
which the course's AAMI mapping groups into S with `a`, `J` and `S`.
Record 209 is in training, outside the six validation records. Its public header
describes a 62-year-old man; no clinical diagnosis is inferred from that metadata.

The interval preceding the target is 179 samples, or 0.4972 s. The median of the three
preceding intervals is 229 samples, or 0.6361 s. This is a nearby comparison in one
selected example, not an estimated patient's normal rhythm or a classification rule.
The normal comparison is the immediately preceding N annotation. Both dataset rows
come directly from `scripts/mitbih_beats.py:record_beats`, including resampling,
window-level normalisation, the R-starting segment and zero padding. The marked
pre-R region locates omitted context; it does not label a particular bump as an
abnormal P wave. A single example cannot establish that morphology never identifies S.

S is a group of beat annotations, not a single disease. An atrial premature beat can
reach the ventricles through the normal conduction path and have a QRS resembling
the surrounding normal beats; a premature or differently shaped P wave can precede
it. The three-class model receives an R-starting row and thus omits that beat's
pre-R context. The observation motivates a hypothesis about input information; poor
S recall by itself does not establish its sole cause.

## Sources and attribution

- Moody G, Mark R. [MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/), DOI 10.13026/C2F305. Signal and annotation data under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
- PhysioNet [annotation definitions](https://archive.physionet.org/physiobank/annotations.shtml) and [record directory](https://physionet.org/physiobank/database/html/mitdbdir/records.htm).
- de Chazal P, O'Dwyer M, Reilly RB. [Automatic classification of heartbeats using ECG morphology and heartbeat interval features](https://pubmed.ncbi.nlm.nih.gov/15248536/). IEEE TBME 51(7), 2004, DOI 10.1109/TBME.2004.827359. Supports considering morphology and intervals together, not a promised result for a new course design.
- [A Detector for Premature Atrial and Ventricular Complexes](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.678558/full), Frontiers in Physiology, 2021, DOI 10.3389/fphys.2021.678558: Figure 3 compares N/S/V morphology; the discussion separates RR disturbances, P-wave changes and QRS morphology.

The open investigation slide offers intervention points, not an assigned solution.
Hard-negative N/S comparisons, auxiliary timing prediction, subtype or patient-balanced
sampling, R-peak jitter robustness and selective referral are untested hypotheses for
class discussion. Model choices are selected on training/validation patients; S recall
is read alongside S precision and false alarms on N.
