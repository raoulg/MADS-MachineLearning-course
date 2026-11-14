---
license: odc-by
tags:
  - vectormesh
  - vector-cache
  - ecg
  - time-series
---

# Heartbeats, MIT-BIH Arrhythmia: raw `(187, 1)` sequences

Single ECG heartbeats from the MIT-BIH Arrhythmia Database, labelled with one of five
AAMI heartbeat classes, as a vectormesh cache. There is no encoder: every row holds
the signal itself, ready for a sequence model.

Built for the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

## What a row is

One heartbeat: 187 steps sampled at 125 Hz (about 1.5 s), amplitudes scaled to [0, 1],
and zero-padded at the end when the beat is shorter than 187 steps. The trailing zeros
are padding, not signal.

| column | type | |
|---|---|---|
| `beat` | float32 `(187, 1)` | the signal as (sequence, channel) |
| `label` | int | 0 N, 1 S, 2 V, 3 F, 4 Q |
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
| rows | 87.554 train / 21.892 test |
| download | 133 MB train / 33 MB test |
| built with | vectormesh 0.13.0, 2026-11-14 |
| licence | ODC-By 1.0 |

## Classes

| label | class | train | test |
|---|---|---|---|
| 0 | N, normal and other non-ectopic beats | 72.471 | 18.118 |
| 1 | S, supraventricular ectopic | 2.223 | 556 |
| 2 | V, ventricular ectopic | 5.788 | 1.448 |
| 3 | F, fusion of ventricular and normal | 641 | 162 |
| 4 | Q, paced or unclassifiable | 6.431 | 1.608 |
| | total | 87.554 | 21.892 |

The classes are strongly unbalanced: N is 83% of both splits, F under 1%. Accuracy
alone says little here; look at per-class recall or a confusion matrix.

## Load it

```python
import torch
from torch.utils.data import DataLoader
from vectormesh import VectorCache
from vectormesh.data import Collate, OneHot

train = VectorCache.from_hub("pttrn-io/heartbeats-mitbih-raw", split="train")
test = VectorCache.from_hub("pttrn-io/heartbeats-mitbih-raw", split="test")
train.metadata  # read this before writing any model code

onehot = OneHot(num_classes=5, label_col="label", target_col="onehot")
collate = Collate(embedding_col="beat", target_col="onehot", padder=torch.stack)


def labelled(cache):
    # map in plain Python format: in torch format every (187, 1) row is converted to a
    # tensor and back, which takes minutes on a full split
    return cache.dataset.with_format(None).map(onehot).with_format("torch")


loader = DataLoader(labelled(train), batch_size=64, shuffle=True, collate_fn=collate)
X, y = next(iter(loader))  # X (64, 187, 1), y (64, 5)
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

1. **MIT-BIH Arrhythmia Database** on PhysioNet (<https://physionet.org/content/mitdb/1.0.0/>,
   doi:10.13026/C2F305): two-channel ambulatory ECG recordings with beat annotations by
   cardiologists.
2. **Kachuee, Fazeli and Sarrafzadeh (2018)** cut the recordings into single beats,
   downsampled to 125 Hz, scaled to [0, 1] and zero-padded to a fixed length.
3. **Kaggle, "ECG Heartbeat Categorization Dataset"**
   (<https://www.kaggle.com/datasets/shayanfazeli/heartbeat>) publishes those beats as
   `mitbih_train.csv` (87.554 beats) and `mitbih_test.csv` (21.892 beats), with the
   class mapping N 0, S 1, V 2, F 3, Q 4.
4. **raoulg/mads-exam-24** (a private course repository, commit `6652bd8f`) holds them
   as `data/heart_big_train.parq` and `data/heart_big_test.parq`, with the same row
   counts as the Kaggle files. `source_idx` is a row's position in those files.

This cache keeps every row of both files, in file order.

## How it was built

`scripts/build_heartbeat_cache.py` in
<https://github.com/raoulg/MADS-MachineLearning-course>, which checks every cached beat
and label against its source row.

## Licence and attribution

Contains information from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), which is made
available under the
[ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/) (Open Data
Commons Attribution License v1.0). This cache is a derived database of it and is shared
under the same licence. The Kaggle page states no licence of its own; ODC-By applies as
the licence of the source database.

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
- Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification:
  A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).
  <https://arxiv.org/abs/1805.00794>
