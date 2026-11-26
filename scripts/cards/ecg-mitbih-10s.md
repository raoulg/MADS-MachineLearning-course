---
license: odc-by
tags:
  - vectormesh
  - vector-cache
  - ecg
  - time-series
---

# ECG, MIT-BIH Arrhythmia: raw 10 s windows `(1250, 1)`

The ECG recordings of the MIT-BIH Arrhythmia Database, uncut into beats, as a vectormesh
cache: every row is 10 seconds of one patient's MLII lead at 125 Hz, with the
cardiologists' beat and rhythm annotations that fall inside it. The cache is **split by
patient**: no patient has windows in both train and test. There is no encoder: every row
holds the signal itself, ready for a sequence model.

It is the companion of
[`pttrn-io/heartbeats-mitbih-raw`](https://huggingface.co/datasets/pttrn-io/heartbeats-mitbih-raw),
which holds single beats cut from these same windows. Use this one when a model needs
more than one beat: several beats at once, the intervals between them, or the rhythm.

Built for the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

## What a row is

One window of 10 s: 1250 steps at 125 Hz of the MLII lead, in millivolt, not scaled. A
record is about 30 minutes, so it gives 180 windows, which follow each other without gap
or overlap; the last 5.6 s of every record are left out. The beats inside a window are
lists, one entry per beat, in time order:

| column | type | |
|---|---|---|
| `ecg` | float32 `(1250, 1)`, a fixed-shape `Array2D` | the signal in mV, as (sequence, channel) |
| `record` | int | the PhysioNet record: one patient |
| `start` | int | the window's first step, at 125 Hz in the record |
| `source_idx` | int | the row's position in its split |
| `r_peaks` | list of int | each beat's annotated R-peak, a step 0 to 1249 in the window |
| `r_peaks_360` | list of int | the same R-peak, as a sample at 360 Hz in the record |
| `beat_class` | list of int | each beat's AAMI class: 0 N, 1 S, 2 V, 3 F, 4 Q |
| `symbol` | list of str | each beat's annotation symbol: `N`, `L`, `R`, `A`, `V`, ... |
| `rhythm` | list of str | the rhythm in effect at each beat: `(N`, `(AFIB`, `(B`, ... |

A window holds a median of 12 to 13 beats. `r_peaks_360` equals `r_peak` in
`heartbeats-mitbih-raw`, so a beat there finds its window here. Within a split the rows of
one record are next to each other and in time order: row `i + 1` starts where row `i`
ends, as long as `record` is the same.

A `DataLoader` over the cache gives `X` of shape `(batch, 1250, 1)`.

| | |
|---|---|
| model_tag | `raw`: the steps unchanged, no encoder |
| hidden_size | 1 (one channel) |
| tensordtype | 2, `(sequence, channel)`, every sequence 1250 steps |
| rows | 3.960 train (22 records) / 3.960 test (22 records) |
| download | 42 MB train / 42 MB test |
| built with | vectormesh 0.14.0, 2026-11-16 |
| licence | ODC-By 1.0 |

### Why windows of 10 s, not whole records

A whole record is a row of 225.000 steps, 22 rows per split: a DataLoader would hand out
a few patients per batch, and every model would first have to cut them up. A window of
fixed shape is what a `VectorCache` and a `DataLoader` handle well: it decodes as one
float32 block, it shuffles, and it batches without padding. Ten seconds is the window the
beats of `heartbeats-mitbih-raw` are cut from (Kachuee et al. 2018), long enough for a
dozen beats and the intervals between them. A longer stretch is a few consecutive rows of
one record glued together (see below), so nothing is lost by the cut.

The amplitudes are left in mV rather than scaled per window: a scale per window would
make two consecutive windows disagree at their seam. Scale as your model needs.

## Annotations

All beats are kept, of all five AAMI classes. The beats per class:

| label | class | train | test |
|---|---|---|---|
| 0 | N, normal and other non-ectopic beats | 45.725 | 44.114 |
| 1 | S, supraventricular ectopic | 931 | 1.833 |
| 2 | V, ventricular ectopic | 3.782 | 3.213 |
| 3 | F, fusion of ventricular and normal | 414 | 388 |
| 4 | Q, unclassifiable | 8 | 7 |
| | total | 50.860 | 49.555 |

These are more than in `heartbeats-mitbih-raw` (45.831 / 44.578), which leaves out F and
Q and every beat whose segment runs past its window. In this split nearly all F beats of
train come from one patient (record 208) and nearly all of test from one other (213).

The rhythm annotations mark where a rhythm begins; `rhythm` gives each beat the one in
effect. Normal sinus rhythm `(N` covers 84% of the beats of train and 73% of test; atrial
fibrillation `(AFIB` is next, with 3.262 beats in 212 windows of train and 7.849 beats in
604 windows of test. Other rhythms include `(B` (ventricular bigeminy), `(T` (trigeminy),
`(AFL` (atrial flutter), `(SVTA`, `(VT`, `(SBR` and `(NOD`; the PhysioNet page lists them
all. A window's labels are yours to define from these lists: "holds a V beat", "is in
AFIB", the classes of its beats in order.

Eleven windows of train hold no annotated beat (record 207, during ventricular flutter,
whose waves are not annotated as beats); their lists are empty.

## Load it

```python
import torch
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.data import Collate

REVISION = "main"  # pin a commit for anything graded
train = VectorCache.from_hub("pttrn-io/ecg-mitbih-10s-raw", split="train", revision=REVISION)
test = VectorCache.from_hub("pttrn-io/ecg-mitbih-10s-raw", split="test", revision=REVISION)
train.metadata  # read this before writing any model code


def holds_v(row):  # one example of a window label: 1 when the window holds a V beat
    return {"target": int(2 in row["beat_class"])}


def labelled(cache):
    # map in plain Python format, then hand only the fixed-shape columns to torch
    return (
        cache.dataset.with_format(None)
        .map(holds_v)
        .with_format("torch", columns=["ecg", "target"])
    )


collate = Collate(embedding_col="ecg", target_col="target", padder=torch.stack)
loader = DataLoader(labelled(train), batch_size=32, shuffle=True, collate_fn=collate)
X, y = next(iter(loader))  # X (32, 1250, 1), y (32,)
```

Longer stretches, the R-R intervals of a window:

```python
import numpy as np

rows = train.dataset.with_format(None)
ecg = train.dataset.with_format("numpy", columns=["ecg"])[:]["ecg"]
records = np.asarray(rows["record"])


def stretch(i, n):
    """n consecutive windows from row i on, as one (n * 1250, 1) signal."""
    assert (records[i : i + n] == records[i]).all(), "the stretch crosses a record"
    return np.concatenate(ecg[i : i + n])


thirty_seconds = stretch(0, 3)            # (3750, 1)
rr = np.diff(rows[5]["r_peaks"]) / 125    # seconds between the beats of row 5
```

This is a `save_to_disk` layout, not parquet: `load_dataset()` will **not** read it,
`VectorCache.load()` and `VectorCache.from_hub()` will.

## Validation

There is **no validation split**. Carve one from train **by record**, so that validation
holds patients the model has not seen, as test does; leave test alone until the end:

```python
val_records = [118, 124, 201, 207, 215, 223]  # as in heartbeats-mitbih-raw
trainset = labelled(train)
fit_idx = np.flatnonzero(~np.isin(records, val_records)).tolist()
val_idx = np.flatnonzero(np.isin(records, val_records)).tolist()
fit, val = trainset.select(fit_idx), trainset.select(val_idx)
```

Do not split windows at random: the windows of one patient look alike, and a random split
measures how well a model recognises the patient.

## Provenance

1. **MIT-BIH Arrhythmia Database** on PhysioNet (<https://physionet.org/content/mitdb/1.0.0/>,
   doi:10.13026/C2F305): 48 two-channel ambulatory ECG recordings of about 30 minutes at
   360 Hz, with beat and rhythm annotations by cardiologists.
2. **The split** is the inter-patient split of de Chazal, O'Dwyer and Reilly (2004): the
   22 records of DS1 train, the 22 of DS2 test. The four records with a pacemaker
   (102, 104, 107, 217) are in neither.
3. **A window** is the MLII lead resampled from 360 Hz to 125 Hz (`scipy.signal.resample_poly`
   over the whole record), cut every 1250 steps from the first sample. The R-peaks are
   the annotated beats (every annotation with an AAMI class) at their sample rounded to
   125 Hz; a beat belongs to the window its rounded R-peak falls in. Annotations that mark
   no beat (rhythm changes, noise, signal quality) are not beats; the rhythm changes give
   `rhythm`.

## How it was built

`scripts/build_ecg_window_cache.py`, with `scripts/mitbih_beats.py`, in
<https://github.com/raoulg/MADS-MachineLearning-course> (`make build-ecg-windows`, which
downloads the records from PhysioNet). The build checks every cached window and its beat
lists against the record, and cuts the beats of `heartbeats-mitbih-raw` again from the
cached windows to check they come out the same.

## Licence and attribution

Contains information from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), which is made
available under the
[ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/) (Open Data
Commons Attribution License v1.0). This cache is a derived database of it and is shared
under the same licence.

When you use it, cite:

- Moody, G., & Mark, R. (2005). MIT-BIH Arrhythmia Database (version 1.0.0). PhysioNet.
  RRID:SCR_007345. <https://doi.org/10.13026/C2F305>
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and
  Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
  <http://ecg.mit.edu/george/publications/mitdb-embs-2001.pdf>
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A.,
  Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical
  research. Nature Health. <https://doi.org/10.1038/s44360-026-00096-z>. Available from:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC13641116/>
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features. IEEE Trans Biomed Eng 51(7):1196-1206
  (2004). <https://doi.org/10.1109/TBME.2004.827359>
- Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification:
  A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).
  <https://arxiv.org/abs/1805.00794>
