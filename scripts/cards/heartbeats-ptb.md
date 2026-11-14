---
license: odc-by
tags:
  - vectormesh
  - vector-cache
  - ecg
  - time-series
---

# Heartbeats, PTB Diagnostic ECG: raw `(187, 1)` sequences

Single ECG heartbeats from the PTB Diagnostic ECG Database, labelled normal or abnormal
(myocardial infarction), as a vectormesh cache. There is no encoder: every row holds
the signal itself, ready for a sequence model.

Built for the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

## What a row is

One heartbeat: 187 steps sampled at 125 Hz (about 1.5 s), amplitudes scaled to [0, 1],
and zero-padded at the end when the beat is shorter than 187 steps. The trailing zeros
are padding, not signal.

| column | type | |
|---|---|---|
| `beat` | float32 `(187, 1)`, a fixed-shape `Array2D` | the signal as (sequence, channel) |
| `label` | int | 0 normal, 1 abnormal |
| `source_idx` | int | the row in the course's source files (see provenance) |

The beat itself is the vector, so there is nothing to look up elsewhere: the cache holds
the whole signal. `source_idx` only records which row of the course's source files a
beat came from.

A `DataLoader` over the cache gives `X` of shape `(batch, 187, 1)`.

| | |
|---|---|
| model_tag | `raw`: the steps unchanged, no encoder |
| hidden_size | 1 (one channel) |
| tensordtype | 2, `(sequence, channel)`, every sequence 187 steps |
| rows | 11.641 train / 2.911 test |
| download | 18 MB train / 4 MB test |
| built with | vectormesh 0.14.0, 2026-11-14 |
| licence | ODC-By 1.0 |

## Classes

| label | class | train | test |
|---|---|---|---|
| 0 | normal | 3.240 | 806 |
| 1 | abnormal | 8.401 | 2.105 |
| | total | 11.641 | 2.911 |

The classes are unbalanced, about 28% normal to 72% abnormal in both splits.

The splits are not fully disjoint: 2 test beats equal a train beat, and train holds 5
beats that repeat another train beat.

## Load it

```python
import torch
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.data import Collate, OneHot

train = VectorCache.from_hub("pttrn-io/heartbeats-ptb-raw", split="train")
test = VectorCache.from_hub("pttrn-io/heartbeats-ptb-raw", split="test")
train.metadata  # read this before writing any model code

onehot = OneHot(num_classes=2, label_col="label", target_col="onehot")
collate = Collate(embedding_col="beat", target_col="onehot", padder=torch.stack)


def labelled(cache):
    # map in plain Python format: in torch format every row is converted to a tensor
    # and back, about five times slower
    return cache.dataset.with_format(None).map(onehot).with_format("torch")


loader = DataLoader(labelled(train), batch_size=64, shuffle=True, collate_fn=collate)
X, y = next(iter(loader))  # X (64, 187, 1), y (64, 2)
```

Pin `revision=` to a commit for anything graded. This is a `save_to_disk` layout, not
parquet: `load_dataset()` will **not** read it, `VectorCache.load()` and
`VectorCache.from_hub()` will.

## Validation

There is **no validation split**. Carve one from train, and leave test alone until the end:

```python
from sklearn.model_selection import train_test_split

trainset = labelled(train)
labels = train.dataset.with_format(None)["label"]
fit_idx, val_idx = train_test_split(
    list(range(len(labels))), test_size=0.2, stratify=labels, random_state=42
)
fit, val = trainset.select(fit_idx), trainset.select(val_idx)

fitloader = DataLoader(fit, batch_size=64, shuffle=True, collate_fn=collate)
valloader = DataLoader(val, batch_size=64, shuffle=False, collate_fn=collate)
testloader = DataLoader(
    labelled(test), batch_size=64, shuffle=False, collate_fn=collate
)
```

## Provenance

1. **PTB Diagnostic ECG Database** on PhysioNet (<https://physionet.org/content/ptbdb/1.0.0/>,
   doi:10.13026/C28C71): 15-lead ECG recordings from the Physikalisch-Technische
   Bundesanstalt, Berlin.
2. **Kachuee, Fazeli and Sarrafzadeh (2018)** cut the recordings into single beats,
   downsampled to 125 Hz, scaled to [0, 1] and zero-padded to a fixed length.
3. **Kaggle, "ECG Heartbeat Categorization Dataset"**
   (<https://www.kaggle.com/datasets/shayanfazeli/heartbeat>) publishes those beats as
   `ptbdb_normal.csv` (4.046 beats) and `ptbdb_abnormal.csv` (10.506 beats), with no
   train/test split.
4. **raoulg/mads-exam-24** (a private course repository, commit `6652bd8f`) holds them
   as `data/heart_train.parq` and `data/heart_test.parq`: both classes together, shuffled
   and split into train and test. `source_idx` is a row's position in those files.

This cache keeps every row of both files, in file order.

## How it was built

`scripts/build_heartbeat_cache.py` in
<https://github.com/raoulg/MADS-MachineLearning-course>, which checks every cached beat
and label against its source row.

## Licence and attribution

Contains information from the
[PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/), which is made
available under the
[ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/) (Open Data
Commons Attribution License v1.0). This cache is a derived database of it and is shared
under the same licence. The Kaggle page states no licence of its own; ODC-By applies as
the licence of the source database.

When you use it, cite:

- Bousseljot R, Kreiseler D, Schnabel, A. Nutzung der EKG-Signaldatenbank CARDIODAT der
  PTB über das Internet. Biomedizinische Technik, Band 40, Ergänzungsband 1 (1995) S 317
  <https://www.deepdyve.com/lp/de-gruyter/nutzung-der-ekg-signaldatenbank-cardiodat-der-ptb-ber-das-internet-uemKpjIFzM>
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A.,
  Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical
  research. Nature Health. <https://doi.org/10.1038/s44360-026-00096-z>. Available from:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC13641116/>
- Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification:
  A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).
  <https://arxiv.org/abs/1805.00794>
