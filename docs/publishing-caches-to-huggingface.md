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

### Decide what the raw column is worth

`remove_columns` drops the column you just embedded:

```python
VectorCache.create(..., remove_columns=["image"])   # pixels are ~100x the vectors
```

This is not automatic — weigh the size against what the column buys downstream:

- **Pixels: drop them.** Images are ~100x their embeddings, and nobody debugs a model by
  reading a tensor of RGB values. A cache people cannot download is not a cache people
  will use.
- **Text: keep it.** IMDB's `text` column is ~25 MB against ~150 MB of vectors, and it is
  what lets a student pull up the reviews their model got wrong. Error analysis on a cache
  is impossible without the thing being classified, and dropping it to save a sixth of the
  download is a bad trade.

The rule is the ratio, not the principle: keep the raw column when it is cheap relative to
the vectors *and* a human would want to read it.

### When it is too big to keep: ship a pointer

Dropping the pixels does not have to mean giving up error analysis. Keep an **index back into
the source dataset** instead of the data itself — 8 bytes a row against ~5 KB for the vector,
so the cost is nil:

```python
# BEFORE any shuffle/select: attach each row's position in the source split
full = ds["train"].add_column("source_idx", list(range(len(ds["train"]))))
train = full.shuffle(seed=42).select(range(4000))

VectorCache.create(..., dataset=train, remove_columns=["image"])   # source_idx survives
```

Then "row 5 looks wrong, let me see it" is two lines:

```python
source = load_dataset("nkirschi/oxford-flowers", revision=SHA)["test"]
source[cache[5]["source_idx"]]["image"]
```

Three things make this work, and all three are easy to get wrong:

- **`add_column`, not `.map()`.** `add_column` bolts a column onto the Arrow table without
  reading the image column. A `.map()` re-encodes every image to disk — on 4000 photographs
  that costs far more than the embedding pass does.
- **Attach it before you shuffle.** After `.shuffle().select()`, cache row 5 is not source
  row 5 (measured: it was source row 5208). Index the source with the position instead of
  the stored `source_idx` and you get a real image of the *wrong* item, with nothing to warn
  you.
- **Pin the source revision.** A pointer is only as stable as what it points at. If the
  source is re-uploaded in a different order every published cache silently starts pointing
  at the wrong rows. Record the git sha in the dataset card and pass `revision=` when
  loading — `main` is a moving target.

So the full rule: **keep the raw column when it is cheap, ship a pointer when it is not, and
pin the revision whenever you ship a pointer.**

---

## 2. One-time setup

### Where the course caches live

**<https://huggingface.co/pttrn-io>** — an organisation, created 2026-07-31, empty until the
first cache lands. Settled in masterplan D12; do not publish course caches anywhere else.

An organisation rather than a personal account because co-teachers need to be able to
re-upload a cache, and the caches should outlive any one account. The Hub has no
convert-user-to-organisation path, and moving a repo later changes its URL and breaks every
notebook that hardcodes it — so this had to be decided before the first publish, and now is.
Note the hyphen: `pttrn-io`, not `pttrnio` (that handle is a personal user account, and users
and orgs share one namespace).

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
pttrn-io/imdb-granite-embedding-small-english-r2
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
and nobody downstream can tell which one they got.

**Name it `{dataset}-{model_tag}`, mirroring the encoder tag exactly** — everything after
the `/` in the model id, unabbreviated. So `ibm-granite/granite-embedding-small-english-r2`
over `stanfordnlp/imdb` gives:

```
pttrn-io/imdb-granite-embedding-small-english-r2
```

It is long, and that is the price of the repo id being self-describing. Abbreviations
("granite-small") are where reproducibility quietly dies: two people shorten the same tag
two different ways, and neither repo says which checkpoint it actually holds. The exact tag
means the repo id and `metadata.json["<column>"]["model_tag"]` can be compared by eye.
The full id still goes in the card, with a revision.

Drop the timestamp prefix from the folder name when you upload — the hub commit history
already dates it. Keep the timestamped folder locally as your own record.

---

## 4. Upload

### The short way

`scripts/publish_cache.py` does everything below — derives the repo id from the cache's own
`metadata.json`, creates the repo, uploads both splits, and generates the dataset card in §5
with the revisions filled in. There is a `Makefile` target per course cache:

```bash
make publish-flowers ARGS=--dry-run
```

Drop `ARGS` to actually upload, or `make publish-all` for every cache. `make list-caches`
shows which artefact folders the targets picked — they take the newest match, so you never
paste a timestamp.

The rest of this section is what those targets do, for when you are publishing something
that is not a course cache.

### With the CLI

Create the repo (public by default; add `--private` while you are still testing):

```bash
uv run hf repos create pttrn-io/imdb-granite-embedding-small-english-r2 --type dataset
```

Upload one split. The three arguments are `REPO_ID`, `LOCAL_PATH`, `PATH_IN_REPO`:

```bash
uv run hf upload pttrn-io/imdb-granite-embedding-small-english-r2 ./artefacts/20260730131253_imdb_train_embed train --type dataset --commit-message "granite-small-r2 embeddings, train split"
```

```bash
uv run hf upload pttrn-io/imdb-granite-embedding-small-english-r2 ./artefacts/20260730131253_imdb_test_embed test --type dataset --commit-message "granite-small-r2 embeddings, test split"
```

For anything above a few GB use the resumable variant instead — it retries and picks up
where it stopped, which matters on a hotel wifi or a VM that gets reaped:

```bash
uv run hf upload-large-folder pttrn-io/eurosat-dinov2-cache ./artefacts/eurosat_train --type dataset
```

### From Python

Useful at the end of a `scripts/build_cache.py` run, so the artefact is published by the
same code that built it:

```python
from pathlib import Path
from huggingface_hub import HfApi

REPO = "pttrn-io/imdb-granite-embedding-small-english-r2"
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

path = snapshot_download("pttrn-io/imdb-granite-embedding-small-english-r2", repo_type="dataset")
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
    "pttrn-io/imdb-granite-embedding-small-english-r2",
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
uv run hf download pttrn-io/imdb-granite-embedding-small-english-r2 --type dataset --local-dir ./artefacts/imdb-granite
```

Only need one split? `snapshot_download(..., allow_patterns="train/*")`.

> **The trap in that line.** `allow_patterns` that matches nothing is **not an error**.
> `snapshot_download` prints `Fetching 0 files`, returns the path, and exits 0 — so a
> pattern naming your *local* tag (`imdb_train/*`) instead of the folder that is actually
> in the repo (`train/*`) looks identical to a cache that was never published. If your
> loader falls back to building on failure, that typo silently costs hours of encoding.
>
> Two habits make it loud:
>
> ```python
> path = Path(snapshot_download(...)) / split
> if not path.exists():
>     raise FileNotFoundError(f"{REPO} has no {split}/ folder")
> ```
>
> and keep `VectorCache.load()` *outside* the `try` that guards the download. A cache that
> downloaded but will not load is a broken artefact — you want the traceback, not a rebuild.

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

- [ ] Raw columns consciously kept or dropped (§1) — pixels out, readable text in
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
