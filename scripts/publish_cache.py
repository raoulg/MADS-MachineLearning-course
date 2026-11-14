"""Publish a VectorCache to the HuggingFace hub, with a dataset card generated from its metadata.

    uv run python scripts/publish_cache.py \
        --dataset fashion-mnist \
        --source zalando-datasets/fashion_mnist \
        --train artefacts/20260731105730_fashion_train_mobilenet_v2_1.0_224_embed \
        --test  artefacts/20260731105751_fashion_test_mobilenet_v2_1.0_224_embed

The repo id is derived, not typed: `{org}/{dataset}-{encoder}`, where the encoder is the
model tag verbatim after the "/". One repo per (dataset x encoder) — see
docs/publishing-caches-to-huggingface.md §3.

The encoder is read from metadata.json rather than parsed out of the folder name, so this
keeps working when vectormesh starts naming the folders itself (raoulg/vectormesh#1).

A cache with no hub source dataset and no encoder (raw signals, say) has nothing for the
generated card to describe. Pass --card PATH to upload that hand-written card instead;
--source is then optional:

    uv run python scripts/publish_cache.py \
        --dataset heartbeats-ptb --card scripts/cards/heartbeats-ptb.md \
        --train artefacts/<newest *_heartbeats-ptb_train_raw_beat> \
        --test  artefacts/<newest *_heartbeats-ptb_test_raw_beat>

Add --dry-run to see the repo id and the card without touching the hub.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ORG = "pttrn-io"

# What a VectorCache folder holds. Anything else in it, such as the `cache-*.arrow`
# files `datasets.map` leaves next to a loaded cache, stays local.
CACHE_FILES = ["data-*.arrow", "dataset_info.json", "state.json", "metadata.json"]


def read_metadata(folder: Path) -> tuple[dict, str, str]:
    """Return (metadata, embedding column, model tag) for a cache folder."""
    if not folder.is_dir():
        # Usually a Makefile glob that matched nothing, i.e. the cache was never built
        # or was built under a different encoder.
        sys.exit(f"{folder} is not a directory — has the cache been built?")
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"{folder} has no metadata.json — is it a VectorCache folder?")
    meta = json.loads(meta_path.read_text())

    columns = [k for k, v in meta.items() if isinstance(v, dict) and "model_tag" in v]
    if not columns:
        sys.exit(f"{meta_path} has no column carrying a model_tag")
    if len(columns) > 1:
        # A cache can hold several vectorizers (regex_features + embed). The encoder that
        # names the repo is ambiguous then, so make the caller choose rather than guess.
        sys.exit(f"{folder} has multiple vector columns {columns}; pass --column")
    column = columns[0]
    return meta, column, meta[column]["model_tag"]


def chunk_summary(chunk_sizes: dict | None) -> str:
    """median / p95 / max over a {chunk_count: n_rows} histogram."""
    if not chunk_sizes:
        return "n/a — one vector per row"
    hist = {int(k): v for k, v in chunk_sizes.items()}
    total = sum(hist.values())
    running, median, p95 = 0, None, None
    for k in sorted(hist):
        running += hist[k]
        if median is None and running >= 0.5 * total:
            median = k
        if p95 is None and running >= 0.95 * total:
            p95 = k
    return f"median {median}, p95 {p95}, max {max(hist)}"


def revision(api: HfApi, repo_id: str, repo_type: str) -> str:
    """The commit sha, bare — the card quotes it both in a table and inside a code string."""
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
        return info.sha or "unresolved"
    except Exception as e:  # noqa: BLE001  # a private or renamed source should not kill the upload
        print(f"  ! could not resolve {repo_type} revision for {repo_id}: {e}")
        return "unresolved"


def build_card(
    api: HfApi,
    repo: str,
    dataset: str,
    source: str,
    column: str,
    model_tag: str,
    splits: dict,
    config: str | None = None,
) -> str:
    """Generate the dataset card. Every number comes from metadata.json."""
    first = next(iter(splits.values()))
    spec = first[column]
    rows = " / ".join(
        f"{v['num_observations']:,}".replace(",", ".") + f" {k}"
        for k, v in splits.items()
    )
    source_rev = revision(api, source, "dataset")

    # A cache that draws on several source splits carries the split per row; one that does
    # not can name its split in the snippet directly.
    args = f'"{source}", ' + (f'"{config}", ' if config else "")
    if "source_split" in first["features"]:
        lookup = f"""splits = {{
    name: load_dataset({args}split=name, revision="{source_rev}")
    for name in ("train", "validation", "test")
}}

row = train[5]
splits[row["source_split"]][row["source_idx"]]      # the original behind cache row 5"""
        idx_note = (
            "Rows carry `source_idx` **and** `source_split`, because this cache pools more than one "
            "split of the source dataset — you need both to land on the right row."
        )
    else:
        lookup = f"""source = load_dataset({args}split="train", revision="{source_rev}")
source[train[5]["source_idx"]]      # the original row behind cache row 5"""
        idx_note = (
            "Rows carry a `source_idx` column: the row's position in the **source** split, so you "
            "can go from a vector back to the row it came from without this repo having to carry it."
        )

    return f"""---
license: other
tags:
  - vectormesh
  - vector-cache
  - embeddings
---

# {dataset} — `{model_tag.split("/")[-1]}` cache

Frozen embeddings of `{source}`, so you can train a head on a CPU laptop in seconds instead
of running an encoder over the source data yourself.

Built for the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

| | |
|---|---|
| source dataset | `{source}`{f", config `{config}`" if config else ""}, revision `{source_rev}` |
| encoder | `{model_tag}`, revision `{revision(api, model_tag, "model")}` |
| hidden_size | {spec["hidden_size"]} |
| tensordtype | {spec["tensordtype"]} |
| chunk_sizes | {chunk_summary(spec.get("chunk_sizes"))} |
| rows | {rows} |
| columns | {", ".join(f"`{c}`" for c in first["features"])} |
| built with | vectormesh {spec["vectormesh_version"]}, {first["created_at"][:10]} |

The raw input column was dropped before upload — these are vectors and labels only.

## Getting the original back

{idx_note}

```python
from datasets import load_dataset

{lookup}
```

`source_idx` indexes the source dataset, not this cache, and the same index resolves to a different
row in a different revision of `{source}` — without erroring. Pass the `revision`.

## Load it

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from vectormesh import VectorCache

path = Path(snapshot_download("{repo}", repo_type="dataset"))
train = VectorCache.load(path / "train")
test = VectorCache.load(path / "test")
train.metadata          # read this before writing any model code
```

This is a `save_to_disk` layout, not parquet: `load_dataset()` will **not** work on it,
`VectorCache.load()` will. For one split only, pass `allow_patterns="train/*"`.

## Licence

The embeddings inherit the terms of the source dataset, `{source}`. Cite it as its authors ask.
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="short name for the repo id, e.g. fashion-mnist",
    )
    p.add_argument(
        "--source",
        help="hub id of the source dataset, e.g. zalando-datasets/fashion_mnist; "
        "required unless --card is given",
    )
    p.add_argument(
        "--card",
        type=Path,
        help="hand-written dataset card to upload instead of the generated one",
    )
    p.add_argument("--train", type=Path, required=True, help="local train cache folder")
    p.add_argument("--test", type=Path, required=True, help="local test cache folder")
    p.add_argument(
        "--config",
        help="source dataset config, if it has one — e.g. scotus for coastalcph/lex_glue",
    )
    p.add_argument("--org", default=ORG)
    p.add_argument("--column", help="vector column, if the cache has more than one")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the repo id and card, upload nothing",
    )
    args = p.parse_args()
    if args.card is None and args.source is None:
        p.error("--source is required unless --card is given")
    if args.card is not None and not args.card.is_file():
        p.error(f"--card {args.card} is not a file")

    splits = {"train": args.train, "test": args.test}
    metas, column, model_tag = {}, args.column, None
    for split, folder in splits.items():
        meta, col, tag = read_metadata(folder)
        column = column or col
        if model_tag and tag != model_tag:
            sys.exit(
                f"encoder mismatch: {tag} in {split} but {model_tag} in the other split"
            )
        model_tag = tag
        metas[split] = meta

    assert model_tag is not None
    repo = f"{args.org}/{args.dataset}-{model_tag.split('/')[-1]}"
    api = HfApi()
    if args.card is not None:
        card = args.card.read_text()
    else:
        card = build_card(
            api, repo, args.dataset, args.source, column, model_tag, metas, args.config
        )

    print(f"\nrepo:    {repo}")
    for split, folder in splits.items():
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
        print(
            f"  {split:5} <- {folder}  ({size:.0f} MB, {metas[split]['num_observations']:,} rows)"
        )

    if args.dry_run:
        print("\n--- dataset card (not uploaded) ---\n")
        print(card)
        return

    api.create_repo(repo, repo_type="dataset", private=args.private, exist_ok=True)
    for split, folder in splits.items():
        print(f"uploading {split} ...")
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=folder,
            path_in_repo=split,
            allow_patterns=CACHE_FILES,
            commit_message=f"{model_tag} embeddings, {split} split",
        )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message="dataset card",
    )
    print(f"\ndone: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
