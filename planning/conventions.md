# Course conventions

Masterplan task **P0.1**. One place for the decisions that eight lessons would otherwise
each answer differently. If a notebook disagrees with this file, one of the two is wrong —
fix it, do not add a variant.

Last updated: 2026-07-30.

---

## 1. Where caches live

**One directory, at the repo root: `artefacts/`.** Already in `.gitignore` (`artefacts/**`),
so a cache never lands in a commit.

Notebooks reach it the same way from anywhere in the tree:

```python
from pathlib import Path

ARTEFACTS = Path("../../artefacts").resolve()
```

Two `..` because every notebook lives at `notebooks/<lesson>/`. Do not invent
`notebooks/*/tmp/artefacts`, `assets/`, or `../artefacts` — those all exist in the current
tree and are the reason this file exists.

Raw *downloads* are a different thing and stay under the HF cache. Pass
`cache_dir="../../data/hf"` to `load_dataset` only when a notebook has a reason to keep its
download separate; otherwise let `datasets` use its own default (`~/.cache/huggingface`),
which is shared across lessons and therefore downloads Fashion-MNIST once for the whole
course.

### 1a. When a lesson gets a cache at all

**Use a `VectorCache` if it helps — reduces compute, makes training easier, or improves
performance. Not because a cache is mandated.** (Owner, 2026-08-01.) The reason the course
works with caches is that they are the smart move for expensive, reusable encoders — not a
box every lesson has to tick.

L3 (timeseries) is the standing counter-example: there is no good general-purpose timeseries
vectorizer yet, and measured caching there cost time and space rather than saving it — a
cached window is **40x larger** than the raw window it came from, and a plain linear model on
raw values beat a head trained on cached TimesFM embeddings (lesson3.yml C9-C11). Before
adding a cache to a lesson, check: does the encoder *compress* the input (images, text — the
pattern works) or *expand* it (this modality, today — it doesn't)?

## 2. How a cache is named

`VectorCache.create` builds the folder name itself:

```
{YYYYMMDDHHMMSS}_{dataset_tag}_{column_name}
20260607154702_imdb_train_embed
```

So the only thing a notebook chooses is `dataset_tag`, and the convention is
**`{dataset}_{split}`** — `imdb_train`, `fashion_test`, `flowers_valid`. The encoder is
already recorded in `column_name` (`embed`, `regex_features`, `granite-small`) and in full in
`metadata.json`.

Timestamps mean a rebuild never silently overwrites the cache a lesson was written against.
That is the point; do not strip them.

## 3. How a notebook gets a cache

In this order:

1. **Download the published one** — `snapshot_download` from the hub, then `VectorCache.load`.
   This is the path every student takes.
2. **Build it** — `VectorCache.create`, in a clearly marked optional branch, for the student
   who has the hardware and wants to change the encoder.

The load must not assume a cache is already on disk. `next(path.glob(...))` raises
`StopIteration` on an empty glob, and `StopIteration` is not a `VectorMeshError` — that is the
exact bug in the current `05_classfication.ipynb` cell 24, and it fires for every student on
first run. Use the standard snippet:

```python
from huggingface_hub import snapshot_download
from vectormesh import VectorCache

path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=ARTEFACTS / NAME)
cache = VectorCache.load(Path(path))
```

Hub repo ids are settled in masterplan **D3**/**Q8** and are not to be improvised per notebook.

## 4. Dataloading

Masterplan **D7**, and it needs mltrainer >= 0.3.2 (**P0.6**) to be true:

```python
DataLoader(cache, batch_size=32, shuffle=True, collate_fn=Collate(...), num_workers=0)
```

- `num_workers=0` in notebooks. The measured spread between 0 and 4 workers is 0.13 ms/batch —
  invisible next to any training step, and workers compete with training compute on a 4-core
  student VM. Use `num_workers=2, persistent_workers=True` in `scripts/`, where a run is long
  enough for it to matter.
- Do **not** call `with_format("torch")` when a `Collate` already stacks. Measured *slower*.
- `streaming=True` only where streaming is the lesson (`L1/03_dataloader`) or the data
  genuinely does not fit.
- A `.py` script with `num_workers>0` needs its entry point under
  `if __name__ == "__main__":` or the workers crash on spawn. Notebooks are unaffected.

## 5. Tracking

**MLflow, one tracker per notebook, never two** (masterplan **D10**).

```python
settings = TrainerSettings(..., reporttypes=[ReportTypes.MLFLOW])
```

Store: `mlflow.set_tracking_uri("sqlite:///" + str(Path("../../mlruns.db").resolve()))` — one
db at the repo root, so runs from different lessons are comparable and there is one thing to
delete. `mlruns.db` is covered by `*.db` in `.gitignore`.

TENSORBOARD stays available as the zero-setup local option and is what `06_exercises` uses,
but it is not what notebooks *teach*. `ReportTypes.TOML` rides on backlog B1; do not add new
uses.

### 5a. Metrics: vectormesh over mltrainer

`MASE`/`MAE`/`Accuracy` exist in both `mltrainer.metrics` and `vectormesh.components.metrics`.
**Prefer vectormesh's, wherever the two have the same capability** (owner, 2026-08-01). This
is an interim rule, not a permanent split: the stated direction is that `mltrainer.Trainer`
may eventually fold into vectormesh too, so "which library" is expected to keep resolving
toward vectormesh over time, not stay a two-library situation forever. `Trainer` itself stays
`mltrainer.Trainer` until/unless that happens — this rule is about metrics only, for now.

## 6. Shapes

Every notebook names the rung it is standing on, using the vocabulary in
[notebooks/1_pytorch_intro/tensor-ladder.md](../notebooks/1_pytorch_intro/tensor-ladder.md) —
`(D)`, `(C, D)`, `(B, D)`, `(B, C, D)`, `(B, K)`. Introduced in `L1/01_tensors`;
**referenced, not re-explained**, everywhere after. The page lives next to the lesson-1
notebooks rather than in `references/`, so a student reading `01_tensors.ipynb` does not have
to go looking for it.

Keep the two levels apart when you write, because students conflate them constantly:

- **formal** — rank and shape, all any pytorch op ever sees. A 2D tensor is a 2D tensor.
- **semantic** — what the axes *mean*. `(C, D)` and `(B, D)` are formally the same object.

Say which level you are arguing on. "Can a CNN take a timeseries?" is a formal question with a
yes, and treating it as a semantic one ("a CNN is an image model") is the single most common
wrong answer in this course.

## 7. Naming and spelling

- Notebooks: `NN_lowercase_with_underscores.ipynb`, numbered in teaching order, exercises last.
- British/American spelling is not policed, but "artefacts" is the spelling used for the
  directory, because that is what is on disk.
- The typo `classfication` is being fixed as each notebook is touched, not in a sweep.
