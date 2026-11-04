# Cuts

What was removed and why, so it does not come back by inertia. A cut recorded here is a
decision, not an accident — if you want to reverse one, argue with the reason.

**Cut, do not stage.** If material no longer reflects practice, remove it — do not invent
a scenario that keeps old teaching code alive. Two entries below exist because that rule
was broken once and had to be applied twice.

---

## Datasets and framing

| Cut | Was | Why |
|---|---|---|
| MNIST / Fashion-MNIST as the L1 spine | `05_classification` | Chosen years ago to solve a weak-hardware problem; a published cache solves it better on real data |
| Oxford Flowers as a tuning target | L1 search | Saturated — 0.9910 with 1-NN and no training at all (`evidence.md` §1). Survives as one demo cell |
| EuroSAT as the search dataset | L1 search | Encoder axis spread is 2.4 pt, so a search there shows nothing moving. DTD's 14.4 pt spread is where the lesson is legible (`evidence.md` §2) |
| ETTh1 | L2 graded sequence | Five datasets across nine notebooks was sprawl, and a six-covariate electricity-transformer domain added confusion an unfamiliar dataset did not need |
| `pedestrian_counts` | L2 | Replaced by `solar_10_minutes`, the one series where N-BEATS wins on its merits with ample data (`evidence.md` §5b). The same swap in both notebooks that used it, so the dataset count stays at three |

## Notebooks

| Cut | Was | Why |
|---|---|---|
| Hand-rolled data generators | `1_pytorch_intro/02_datagenerators` | Superseded by `Collate` + `DataLoader`; the streaming case is a legacy note |
| Hand-rolled dataloader (full notebook) | `1_pytorch_intro/03_dataloader` | Compressed to one section of the L1 training notebook |
| Double for-loop hyperparameter search | L1/L2 exercises | Replaced by ray.tune from lesson 1 |
| "Build your own cache + track experiments" as a whole lesson | planned L2, `2_convolutions/02_pretrained` + `04_exercises.md` | Ray and mlflow both land in L1 before this lesson would run — redundant on arrival, not merely stale |
| Separate transfer-learning notebooks | `3_tuning_networks/03`, `04` | Transfer learning **is** L1's premise; a lesson restating it teaches the thing the course has already replaced |
| `09_composition_extra` | L2 appendix | Existed only to keep ETTh1 alive after ETTh1 was cut from the graded sequence. Staging, by definition |
| `03_Attention_gestures`, `05_imdb_preprocessing`, `06_imdb_attention` | L4 | `mads_datasets` + `mltrainer.rnn_models`, all at the abstraction level the course moved off. Gestures also duplicate L2's modality |
| `07_metric` | L4 | L1's training notebook already separates loss from metric. The one non-redundant idea — thresholding under cost asymmetry — moves to L6 rather than dying |
| `08_excercise` | L4 | Tied to the cut IMDB pipeline; replaced by an exercise on long documents |
| `5_unsupervised/src/` | L5 | `autoencoder.py`, `vae_embeddings.py`, `show_vae.py`, `query_embeddings.py`, `settings.py`. An autoencoder is `Serial(NeuralNet([d,64,32]), NeuralNet([32,64,d]))` — the module directory teaches plumbing the course has abstracted away |
| GNNs as a graded lesson | `6_graphs` | Three thin notebooks and a heavyweight dependency, the least connected material in the course. Demoted to `8_extras/graphs`; the *structural* graph material students actually need comes back in L7 as retrieval infrastructure with a job |

## Dependencies

| Cut | Was | Why |
|---|---|---|
| `mads_datasets` | L3, L4 notebooks, `pyproject.toml` | Everything loads from the HF hub now; L3 and L4 were its last consumers |
| `tune.run` legacy API | `3_tuning_networks/hypertune.py` | ray 2.55 ships `tune.Tuner` + `tune.TuneConfig` + `tune.RunConfig`, all under `ray.tune`, no `ray.train` import |
| `bayesian-optimization==1.5.1` | `pyproject.toml` | A pinned dependency carried for one notebook section. BayesOpt is demoted to a mention in L3's algorithm survey |
| `torch-geometric` (required) | `pyproject.toml` | Moves to an optional dependency group with the GNN material |

---

## Known limits, not cuts

- `CollateParallel` takes exactly **two** columns. Three-way fusion needs a library
  change; L1 and L4 stay at two.
