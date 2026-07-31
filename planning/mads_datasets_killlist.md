# mads_datasets kill list

**The rule (masterplan D2):** `mads_datasets` is removed *per site*, and only once that
site has a replacement that has been **run**. `mads-datasets[torch]` stays in
`pyproject.toml` until the last row below is green. No repo-wide sweep.

**Status legend**

| | meaning |
|---|---|
| ⬜ | not started |
| 🔶 | replacement written, **not yet run** — does not count |
| ✅ | replaced, notebook executed top to bottom, output inspected |

Last updated: 2026-07-31. **Lessons 1 and 2 are green**, both verified against the
published vectormesh 0.4.0 / mltrainer 0.3.4.

---

## 1. The 14 sites reduce to 5 real problems

Most rows are the same swap repeated. Solve these five once and the table below is
mechanical:

| # | Problem | Replacement | Where it bites |
|---|---|---|---|
| A | `DatasetFactoryProvider.create_factory(...)` + `download_data()` | `load_dataset("<hub id>")` + `DatasetSchema.infer` | 11 sites |
| B | `factory.create_datastreamer(...)` → `train.stream()` | `DataLoader(dataset, collate_fn=Collate(...))`, **num_workers=0** (masterplan D7) | 6 sites |
| C | `mads_datasets.base.BaseDatastreamer` used *directly* as teaching material | **deleted, not relocated** (masterplan D5). `DataLoader`, plus `load_dataset(..., streaming=True)` where the does-not-fit-in-RAM point needs making. Goal 1.5 is **kept** — only the implementation changes (D6) | L1/03, L3/03 |
| D | `datatools.window(...)` | **moves into vectormesh** (D5) — it is the `(C, D)`-producing operation the library is built around, and VM.1 needs it anyway. The L3 notebook still hand-rolls it first, then imports the library version | L3/03, L3/04, L5/03 |
| E | `datatools.get_file(...)` | `huggingface_hub.hf_hub_download`, or `urllib` for a one-off URL | L5/08 |

### The mltrainer preprocessors map 1:1 onto vectormesh Collate

Worth knowing before starting: **mltrainer does not depend on mads_datasets** (checked its
metadata — no such requirement), and its preprocessors are plain collate functions over
`list[tuple]`:

| mltrainer | vectormesh equivalent |
|---|---|
| `BasePreprocessor` — `torch.stack(X), torch.stack(y)` | `Collate(padder=torch.stack)` |
| `PaddedPreprocessor` — `pad_sequence(X, batch_first=True)` | `Collate(padder=DynamicPadding())` |

So the streamer→DataLoader move (problem B) is not a rewrite, it is renaming the same
function and gaining a documented shape contract. Say that out loud in the notebooks.

### Which loader (masterplan D7)

Plain torch `DataLoader` with `num_workers=0`. HF `datasets` ships no loader of its own —
`Dataset` and `IterableDataset` are both meant to be *fed to* a torch DataLoader. Measured
on the real 25k IMDB regex cache: 0.31 ms/batch at `num_workers=0`, 0.18 at 4 workers, and
`with_format("torch")` is *slower* (0.41) because the Collate already stacks. At those
numbers the choice is invisible next to any training step, so pick the simplest. Use
`streaming=True` only where streaming is the lesson (L1/03) or the data genuinely does not
fit.

### ✅ mltrainer fixed (masterplan P0.6) — landed 2026-07-30, needs publishing

`Trainer` now holds its dataloader iterator and re-`iter`s it only on `StopIteration`
(`_next_batch`, `src/mltrainer/trainer.py`), with four regression tests in
`tests/test_trainer_iteration.py`. Verified: pre-fix a `shuffle=False` loader yielded batch
`[0, 0, 0, 0]`, post-fix `[0, 16, 32, 48]`. Released as **0.3.2** on PyPI (2026-07-30) and the course floor is raised to match, so this
holds for a student on a fresh `uv sync`. Original description below.

#### The original problem

`Trainer.trainbatches` and `Trainer.evalbatches` call `next(iter(self.dataloader))` **inside
the step loop**. That is correct for an *iterator* — which `BaseDatastreamer.stream()`
returns, and which the `traindataloader: Iterator` annotation asks for — but a torch
`DataLoader` is an *iterable*, so every step builds a fresh iterator:

- **train** (`shuffle=True`): a reshuffled first batch each step. Trains by luck, and
  re-permutes the full index every step.
- **valid** (`shuffle=False`): the *same first batch*, `valid_steps` times. Validation
  metrics describe one batch, not the split.

Every problem-B row below turns a working iterator into a silently-degraded iterable, so
**P0.6 lands first**. It is also the precondition for "HF/torch dataloaders will perform
better" being true: with `num_workers>0`, the current code respawns the worker pool per
batch.

---

## 2. The table

### Lesson 1 — pytorch intro

| Site | Uses | Replacement | Status |
|---|---|---|---|
| `05_classfication.ipynb` cell 2 | `import mads_datasets`, FASHION factory | dead code — deleted 2026-07-30. Notebook is now `07_vectors.ipynb` | ✅ |
| `03_dataloader.ipynb` cells 2, 7, 17, 48 | FLOWERS factory + `download_data()`, `FileTypes`, `BaseDatastreamer` | Done 2026-07-30. Image-folder half cut per D8; rebuilt as an iterator lesson on `nkirschi/oxford-flowers`: memory-map + lazy decode, then `yield` + shuffled index + wrapping pointer, then DataLoader, then `streaming=True` on FineWeb for goal 1.5. **52 → 30 cells.** Runs in 17s | ✅ |
| `05_image_classfication.ipynb` cell 2, 19 | FASHION factory + `create_datastreamer` → 4 streamers | Done 2026-07-30, renamed `06_classification.ipynb`. `zalando-datasets/fashion_mnist` + DataLoader with an explicit PIL→`(1,28,28)` float conversion. The "which one is faster" comparison became DataLoader (1.4 ms/batch) vs HF streaming (2.1 ms/batch) — **and the `%timeit next(iter(loader))` idiom was fixed too**, it was timing iterator construction. Runs in 65s | ✅ |
| `06_exercises.ipynb` cell 0, 1 | `DatasetType` import; markdown telling students to add `mads_datasets` to their own repo | Done 2026-07-30, renamed `09_exercises.ipynb`. Dependency advice now names `datasets`, `vectormesh` and `mltrainer >= 0.3.2`, and points the "source to study" hint at `BaseVectorizer`/`VectorCache`. Runs in 64s | ✅ |

### Lesson 2 — convolutions

| Site | Uses | Replacement | Status |
|---|---|---|---|
| `01_convolutions.ipynb` cell 2 (was `02_`) | FASHION factory + streamers; commented-out FLOWERS variant | Done 2026-07-30. `zalando-datasets/fashion_mnist` + `DataLoader` with the same explicit PIL→`(1,28,28)` conversion as L1/06_classification, `num_workers=0` per conventions §4. The commented-out FLOWERS variant is gone — the frozen-encoder notebook covers flowers properly now. Runs in 31s | ✅ |
| `03_mlflow.ipynb` cell 1, 2, 5 | FASHION factory; markdown selling the mads_datasets package | Done 2026-07-30, rewritten around the L2 comparison (masterplan D9/D10); re-verified 07-31. The "let's use the mads_datasets package" paragraph is gone; the notebook now reuses the cache `02_pretrained` wrote, and runs on **flowers** because fashion has no crossover to show (lesson2.yml C9). MLflow is the only report type | ✅ |
| `03_mlflow.py` — **was not on this list** | `get_fashion_streamers()`: FASHION factory + `BasePreprocessor` + `create_datastreamer` | Done 2026-07-30. Rewritten around `load_dataset` + `DataLoader`. This is the one place in the course that uses `num_workers=2, persistent_workers=True` and carries the `if __name__ == "__main__"` spawn-guard comment — i.e. it is where conventions §4's script/notebook split is demonstrated | ✅ |

> The `.py` row above was missing from the original table, which counted **14 sites** by scanning
> notebooks only. Running `grep -rln mads_datasets notebooks/` on 2026-07-30 turned up **three**
> non-notebook sites, none of them listed:
>
> | Site | Status |
> |---|---|
> | `2_convolutions/03_mlflow.py` | ✅ migrated (row above) |
> | `4_tuning_networks/hypertune.py` | ⬜ add to the L4 section |
> | `6_unsupervised/src/settings.py` | ⬜ add to an L6 section, which does not exist yet |
>
> So the real count is **17 sites, not 14**, and the two open ones must be green before
> `mads-datasets[torch]` can leave `pyproject.toml`. Also note `4_tuning_networks/05_hypertuning_exercise.md`
> mentions the package in prose — no import, but it needs rewording with the L4 rows.

### Lesson 3 — recurrent networks

| Site | Uses | Replacement | Status |
|---|---|---|---|
| `03_basemodels.ipynb` cell 4 | SUNSPOTS factory + `download_data()` | fetch the SILSO monthly txt directly (urllib + pandas). **Not** Monash — tested: its `sunspot` config is the *daily* series (73.864 points from 1818) and it is script-based, so `load_dataset` fails on datasets 5.0. Monash becomes the exercise catalog instead. See masterplan P2.1 | ⬜ |
| `03_basemodels.ipynb` cell 16 | `datatools.window`, `factory.settings.window_size` | problem D — hand-roll the windowing; `window_size` becomes an explicit notebook variable instead of a hidden factory setting, which is an improvement | ⬜ |
| `03_basemodels.ipynb` cell 20 | `class SunspotDataset(DatasetProtocol)` | a plain `torch.utils.data.Dataset`, or build the windows into an HF `Dataset` | ⬜ |
| `03_basemodels.ipynb` cell 47 | `BaseDatastreamer` in type hints of the naive-baseline class | mechanical: retype to `DataLoader` | ⬜ |
| `04_excercise.ipynb` cell 0, 2 | `datatools`, GESTURES factory + streamers | **blocked** — needs the GESTURES replacement decision (masterplan P2.2) | ⬜ |

### Lesson 4 — tuning networks

| Site | Uses | Replacement | Status |
|---|---|---|---|
| `01_hypertuner.ipynb` cell 12 | GESTURES factory + streamers, inside the ray trainable | **blocked** on P2.2. Note: the factory call is inside the tune function, so it runs per trial — a cached dataset fixes a real inefficiency here, not just an import | ⬜ |
| `03_minimal_transfer_learning.ipynb` cell 5 | `Path.home()/".cache/mads_datasets/hymenoptera_data"` — reads a directory that mads_datasets happened to fill | no import to remove, but the path is a dependency in disguise: it silently assumes another notebook ran. Replace with an explicit hub dataset (masterplan lists dog-food / rock-paper-scissors as substitutes) | ⬜ |
| `04_transfer_learning_with_resnet.ipynb` cell 1, 4 | `!pip install mads_datasets`, FLOWERS factory + streamers | `nkirschi/oxford-flowers`; fix the colab pip line too | ⬜ |

### Lesson 5 — attention

| Site | Uses | Replacement | Status |
|---|---|---|---|
| `03_Attention_gestures.ipynb` cell 0, 2 | `datatools`, GESTURES factory + streamers | **blocked** on P2.2 | ⬜ |
| `05_imdb_preprocessing.ipynb` cell 2 | IMDB factory → `create_dataset()` | `stanfordnlp/imdb` (already proven in L1/05) | ⬜ |
| `06_imdb_attention.ipynb` cell 0, ~36 | IMDB factory → streamers | same | ⬜ |
| `07_metric.ipynb` cell 4 | FASHION factory → `create_dataset()` | `zalando-datasets/fashion_mnist` | ⬜ |
| `08_excercise.ipynb` cell 1, 17 | `datatools.get_file` for the UCI EEG `.arff`; markdown telling students to study mads_datasets source and build their own datafactory | problem E; and **rewrite the markdown exercise** — "implement your own datafactory" becomes "implement your own `BaseVectorizer`", which is a strictly better exercise and points at code we still ship | ⬜ |

---

## 3. Ordering

Per masterplan R8, migrate a **whole lesson or none of it** — do not start a cohort with a
lesson where two notebooks disagree about how data is loaded.

1. **L1** — 6 sites, all drop-in except problem C. Unblocks phase_1.
2. **L2** — 2 sites, both FASHION. Trivial once L1 is done.
3. **L5** — 5 sites; 3 are drop-in (IMDB, FASHION), 2 blocked on gestures.
4. **L4** — 3 sites; the hymenoptera path is the interesting one.
5. **L3** — 5 sites, the most design work (windowing, DatasetProtocol) and one blocked.
6. **Blocked group** — L3/04, L4/01, L5/03 all fall out of one decision (P2.2). Do them in
   a single pass, on the same day, with the same dataset.

## 4. Definition of done (the dependency drop)

- [ ] every row above is ✅
- [ ] `grep -rn "mads_datasets" notebooks/ README.md` returns nothing
- [ ] `mads-datasets[torch]` removed from `pyproject.toml`, `uv lock` regenerated
- [ ] `uv sync` from scratch in a clean venv, then every notebook runs (notebooktester is
      already a dev dependency)
- [ ] README's "I have separated the management of datasets…" section rewritten
- [ ] `references/leerdoelen/les1.md` goal 1.5 (datastreamer advantages) still has a
      notebook that demonstrates it
