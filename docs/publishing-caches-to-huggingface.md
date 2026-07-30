# Publishing a vector cache to the HuggingFace hub

> **Why.** A `VectorCache` is a deliverable, not a temp file. Whoever has the GPU time and
> the judgement to pick an encoder runs the embedding pass **once**, and everyone else — a
> colleague, a downstream project, 30 students on CPU laptops — trains heads against that
> artefact. The hub is the cheapest way to hand it over: public URL, versioned, resumable
> downloads, no server to maintain.

This document is written against `huggingface_hub` 1.20 and `datasets` 5.0 (the versions
pinned in this repo). The CLI is `hf`; the older `huggingface-cli` still exists as a
deprecated alias.

Everything below is run through `uv run` so you get the project's environment:

```bash
uv run hf version
```

---

## 1. What is actually on disk

`VectorCache.create()` calls `Dataset.save_to_disk()` and then writes its own
`metadata.json` next to it. A finished cache folder looks like this:

```
20260730131253_imdb_train_regex_features/
├── data-00000-of-00001.arrow   36 MB   the vectors + surviving columns
├── dataset_info.json                   arrow schema, written by `datasets`
├── state.json                          split/shard bookkeeping, written by `datasets`
└── metadata.json                       written by vectormesh — the contract
```

`metadata.json` is the part that matters to whoever downloads it:

```json
{
  "regex_features": {
    "vectormesh_version": "0.2.0",
    "model_tag": "regex_vectorizer",
    "vectorizer_type": "RegexVectorizer",
    "tensordtype": 1,
    "hidden_size": 43,
    "context_size": null,
    "chunk_sizes": null
  },
  "features": ["text", "label", "regex_features"],
  "created_at": "2026-07-30T13:12:53.959983",
  "num_observations": 25000
}
```

Two consequences for publishing:

1. **Upload the folder as a whole.** The `.arrow` file is meaningless without
   `state.json`, and `metadata.json` is meaningless without both.
2. **`load_dataset("user/repo")` will not work on it.** A `save_to_disk` folder is not
   the parquet layout that `load_dataset` expects. You download the folder and call
   `VectorCache.load(path)`. See [§6](#6-the-alternative-push_to_hub) if you also want a
   hub-viewer-friendly copy.

### Keep it small before you upload

Drop the raw column you just embedded — that is what `remove_columns` is for:

```python
VectorCache.create(..., remove_columns=["image"])   # pixels are ~100x the vectors
```

A cache people cannot download is not a cache people will use.

---

## 2. One-time setup

### Account or organisation?

A personal account (e.g. `pttrnio`) works fine and is one click. An **organisation** is
worth it if you want co-teachers to be able to re-upload a cache, or if you want the
course caches to survive independently of one person's account. You can move a repo to an
org later, but the URL changes and every notebook that hardcodes it breaks — so decide
before you publish the first one.

### Log in

Create a token with **write** access at <https://huggingface.co/settings/tokens>, then:

```bash
uv run hf auth login
```

```bash
uv run hf auth whoami
```

The token is stored in `~/.cache/huggingface/token`. In CI or on a VM, set the
`HF_TOKEN` environment variable instead of logging in interactively.

---

## 3. Naming and repo layout

One dataset repo per **(dataset × encoder)** pair, with one folder per split inside it:

```
pttrnio/imdb-granite-small-r2-cache
├── README.md            <- the dataset card (see §5)
├── train/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   ├── state.json
│   └── metadata.json
└── test/
    └── ...
```

Why per encoder and not per dataset: the encoder *is* the thing that makes the vectors
what they are. `imdb-cache` will collide with itself the moment you try a second model,
and nobody downstream can tell which one they got. Put the model in the name and the full
model id in the card.

Drop the timestamp prefix from the folder name when you upload — the hub commit history
already dates it. Keep the timestamped folder locally as your own record.

---

## 4. Upload

### With the CLI

Create the repo (public by default; add `--private` while you are still testing):

```bash
uv run hf repos create pttrnio/imdb-granite-small-r2-cache --type dataset
```

Upload one split. The three arguments are `REPO_ID`, `LOCAL_PATH`, `PATH_IN_REPO`:

```bash
uv run hf upload pttrnio/imdb-granite-small-r2-cache ./artefacts/20260730131253_imdb_train_embed train --type dataset --commit-message "granite-small-r2 embeddings, train split"
```

```bash
uv run hf upload pttrnio/imdb-granite-small-r2-cache ./artefacts/20260730131253_imdb_test_embed test --type dataset --commit-message "granite-small-r2 embeddings, test split"
```

For anything above a few GB use the resumable variant instead — it retries and picks up
where it stopped, which matters on a hotel wifi or a VM that gets reaped:

```bash
uv run hf upload-large-folder pttrnio/eurosat-dinov2-cache ./artefacts/eurosat_train --type dataset
```

### From Python

Useful at the end of a `scripts/build_cache.py` run, so the artefact is published by the
same code that built it:

```python
from pathlib import Path
from huggingface_hub import HfApi

REPO = "pttrnio/imdb-granite-small-r2-cache"
api = HfApi()
api.create_repo(REPO, repo_type="dataset", exist_ok=True)

for split, folder in [("train", trainpath), ("test", testpath)]:
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=Path(folder),
        path_in_repo=split,
        commit_message=f"{split} split, granite-embedding-small-english-r2",
    )
```

`.arrow` files are picked up by git-LFS automatically; you do not configure anything.

---

## 5. The dataset card

`README.md` in the repo root is the first thing a consumer reads, and for a cache it has
to answer more than usual — they cannot inspect the raw data, because you removed it.
Minimum:

~~~markdown
---
license: mit
tags: [vectormesh, vector-cache, embeddings]
---

# IMDB — granite-embedding-small-english-r2 cache

Frozen embeddings of `stanfordnlp/imdb`, for training small heads without a GPU.

| | |
|---|---|
| source dataset | `stanfordnlp/imdb` (revision `e6281661`) |
| encoder | `ibm-granite/granite-embedding-small-english-r2` (revision `...`) |
| hidden_size | 384 |
| tensordtype | 2 — `(chunks, dim)` per row |
| context_size / stride | 512 / 51 |
| chunk_sizes | median 1, p95 3, max 27 |
| rows | 25.000 train / 25.000 test |
| built with | vectormesh 0.2.0, 2026-07-30 |
| licence | inherits the source dataset's licence |

## Load it

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from vectormesh import VectorCache

path = snapshot_download("pttrnio/imdb-granite-small-r2-cache", repo_type="dataset")
train = VectorCache.load(Path(path) / "train")
```

## How it was built

`scripts/create_cache_imdb.py` in <link to the repo>. ~12 minutes on one A100.
~~~

**Pin the encoder revision.** "granite-small" is not reproducible; a model id plus a
commit hash is. The same goes for the source dataset revision — community datasets get
silently re-uploaded.

**Licence.** A cache of someone else's data inherits their terms. Embeddings are a
derived work, and "research use only" on the source means research use only on the cache.
If the source licence does not permit redistribution, keep the repo private and share by
invitation, or do not publish it at all.

---

## 6. Download and load

For consumers — this is the snippet that belongs at the top of every cache-consuming
notebook:

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from vectormesh import VectorCache

path = Path(snapshot_download(
    "pttrnio/imdb-granite-small-r2-cache",
    repo_type="dataset",
    revision="main",          # pin a commit hash for a graded assignment
))

traincache = VectorCache.load(path / "train")
testcache = VectorCache.load(path / "test")
traincache.metadata          # read this before writing any model code
```

`snapshot_download` caches into `~/.cache/huggingface/hub`, so the second call is free and
the notebook is safe to re-run. To get the files in a specific folder instead, pass
`local_dir=...`, or use the CLI:

```bash
uv run hf download pttrnio/imdb-granite-small-r2-cache --type dataset --local-dir ./artefacts/imdb-granite
```

Only need one split? `snapshot_download(..., allow_patterns="train/*")`.

---

## 7. The alternative: `push_to_hub`

`Dataset.push_to_hub()` writes **parquet** instead of arrow. Trade-offs:

| | `upload_folder` (§4) | `push_to_hub` |
|---|---|---|
| `VectorCache.load()` | works | does not — needs the `save_to_disk` layout |
| `load_dataset()` | does not work | works |
| hub dataset viewer | no preview | previews in the browser |
| `metadata.json` | travels with the folder | must be uploaded separately |

So: `upload_folder` for caches that vectormesh will load, `push_to_hub` when you want a
browsable dataset for people who are not using vectormesh. Publishing both into one repo
(parquet at the root, the arrow folders under `cache/`) is possible and is what you want
if the cache doubles as teaching material — at the cost of storing the vectors twice.

If you go the `push_to_hub` route, upload the metadata alongside it or the cache stops
being a contract:

```python
dataset.push_to_hub(REPO, split="train")
HfApi().upload_file(
    path_or_fileobj=folder / "metadata.json",
    path_in_repo="metadata.json",
    repo_id=REPO,
    repo_type="dataset",
)
```

---

## 8. Checklist before you publish

- [ ] Raw columns dropped (`remove_columns`) — the folder is as small as it can be
- [ ] `metadata.json` present in every split folder
- [ ] Encoder id **and revision** recorded in the card
- [ ] Source dataset id, revision, and licence recorded — and redistribution is allowed
- [ ] The load snippet in the card was copy-pasted into a fresh kernel and it worked
- [ ] `traincache.metadata` matches what the card claims (dim, rows, tensordtype)
- [ ] Local artefact folders are gitignored in the repo that built them

---

## Related

- `vectormesh/docs/03-data-layer.md` — what `VectorCache.create/load/extend` do
- `vectormesh/docs/01-core-concepts.md` §1.7 — the cache as a contract
- <https://huggingface.co/docs/huggingface_hub/guides/upload> — upstream upload guide
