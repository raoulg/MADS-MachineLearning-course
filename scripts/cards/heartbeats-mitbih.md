---
license: odc-by
tags:
  - vectormesh
  - vector-cache
  - ecg
  - time-series
---

# Heartbeats, MIT-BIH Arrhythmia: raw `(187, 1)` sequences

Single ECG heartbeats from the MIT-BIH Arrhythmia Database, labelled with one of three
AAMI heartbeat classes (N, S, V), as a vectormesh cache, **split by patient**: no patient has beats
in both train and test. There is no encoder: every row holds the signal itself, ready for
a sequence model.

Built for the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

## What a row is

One heartbeat: 187 steps sampled at 125 Hz (about 1.5 s), amplitudes scaled to [0, 1],
and zero-padded at the end when the beat is shorter than 187 steps. The trailing zeros
are padding, not signal.

| column | type | |
|---|---|---|
| `beat` | float32 `(187, 1)`, a fixed-shape `Array2D` | the signal as (sequence, channel) |
| `label` | int | 0 N, 1 S, 2 V |
| `record` | int | the PhysioNet record the beat comes from: one patient |
| `r_peak` | int | the beat's annotated R-peak, as a sample at 360 Hz in that record |
| `source_idx` | int | the row's position in its split |

The beat itself is the vector, so there is nothing to look up elsewhere: the cache holds
the whole signal. `record` and `r_peak` find a beat back in the PhysioNet record.

A `DataLoader` over the cache gives `X` of shape `(batch, 187, 1)`.

| | |
|---|---|
| model_tag | `raw`: the steps unchanged, no encoder |
| hidden_size | 1 (one channel) |
| tensordtype | 2, `(sequence, channel)`, every sequence 187 steps |
| rows | 45.831 train (22 records) / 44.578 test (22 records) |
| download | 67 MB train / 65 MB test |
| built with | vectormesh 0.14.0, 2026-11-16 |
| licence | ODC-By 1.0 |

## Classes

| label | class | train | test |
|---|---|---|---|
| 0 | N, normal and other non-ectopic beats | 41.492 | 39.991 |
| 1 | S, supraventricular ectopic | 851 | 1.645 |
| 2 | V, ventricular ectopic | 3.488 | 2.942 |
| | total | 45.831 | 44.578 |

The classes are strongly unbalanced: N is about 90% of both splits, S under 4%. Accuracy
alone says little here; look at per-class recall or a confusion matrix.

The AAMI classes F (fusion of ventricular and normal) and Q (paced or unclassifiable) are
left out. In this split nearly all F beats of train come from one patient (347 of 385, in
record 208) and nearly all of test from one other (338 of 362, in record 213): a recall on
F would measure one patient. Q is almost absent once the paced records are out.

Test holds about twice as many S beats as train, as in the split it follows, and three
quarters of them come from one patient (record 232). A model that
learned S from 851 beats of 22 patients meets the S beats of 22 others: this is the
hard part of the problem, and the reason the split is by patient.

## Load it

```python
import torch
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.data import Collate, OneHot

train = VectorCache.from_hub("pttrn-io/heartbeats-mitbih-raw", split="train")
test = VectorCache.from_hub("pttrn-io/heartbeats-mitbih-raw", split="test")
train.metadata  # read this before writing any model code

onehot = OneHot(num_classes=3, label_col="label", target_col="onehot")
collate = Collate(embedding_col="beat", target_col="onehot", padder=torch.stack)


def labelled(cache):
    # map in plain Python format: in torch format every row is converted to a tensor
    # and back, about five times slower
    return cache.dataset.with_format(None).map(onehot).with_format("torch")


loader = DataLoader(labelled(train), batch_size=64, shuffle=True, collate_fn=collate)
X, y = next(iter(loader))  # X (64, 187, 1), y (64, 3)
```

Pin `revision=` to a commit for anything graded. This is a `save_to_disk` layout, not
parquet: `load_dataset()` will **not** read it, `VectorCache.load()` and
`VectorCache.from_hub()` will.

## Validation

There is **no validation split**. Carve one from train **by record**, so that validation
holds patients the model has not seen, as test does; leave test alone until the end:

```python
import numpy as np

labelled_train = labelled(train)
records = np.asarray(train.dataset.with_format(None)["record"])
valid_records = [118, 124, 201, 207, 215, 223]  # six patients with N, S and V among them
train_idx = np.flatnonzero(~np.isin(records, valid_records)).tolist()
valid_idx = np.flatnonzero(np.isin(records, valid_records)).tolist()
trainset = labelled_train.select(train_idx)
validset = labelled_train.select(valid_idx)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate)
testloader = DataLoader(
    labelled(test), batch_size=64, shuffle=False, collate_fn=collate
)
```

## Provenance

1. **MIT-BIH Arrhythmia Database** on PhysioNet (<https://physionet.org/content/mitdb/1.0.0/>,
   doi:10.13026/C2F305): two-channel ambulatory ECG recordings with beat annotations by
   cardiologists.
2. **The split** is the inter-patient split of de Chazal, O'Dwyer and Reilly (2004): the
   22 records of DS1 train, the 22 of DS2 test. The four records with a pacemaker
   (102, 104, 107, 217) are in neither.
3. **A beat is cut** the way Kachuee, Fazeli and Sarrafzadeh (2018) cut theirs: the MLII
   lead resampled to 125 Hz, in windows of 10 s, each window scaled to [0, 1]; from every
   annotated R-peak a segment of 1.2 times the window's median R-R interval, zero-padded
   to 187 steps. The R-peaks and classes are the cardiologists' annotations, mapped to the
   AAMI classes; F and Q beats count for the window's median interval but are not cut. A
   beat whose segment runs past the end of its window is left out, about one in ten.

The Kaggle files of Kachuee et al. ("ECG Heartbeat Categorization Dataset") hold beats cut
in the same way but split per beat, so that beats of one patient are in both train and
test; this cache is cut from the records instead, so that the split can follow the
patients. Its average beat per class correlates 0.95 to 0.98 with theirs.

## How it was built

`scripts/build_heartbeat_cache.py` and `scripts/mitbih_beats.py` in
<https://github.com/raoulg/MADS-MachineLearning-course> (`make build-heartbeats`, which
downloads the records from PhysioNet). The build checks every cached beat and label
against the beat it cut.

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
