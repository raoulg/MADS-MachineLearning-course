"""Build the SCOTUS caches of lesson 4 for one chunk setting: embeddings and regex counts.

    uv run python scripts/build_scotus_caches.py --chunk 256 --overlap 25
    uv run python scripts/build_scotus_caches.py --chunk 512 --overlap 128

A setting is a chunk size and an overlap, both in tokens of the `bge-large-en-v1.5`
tokenizer. For each split the script writes two caches into `artefacts/`:

- the embeddings, one `(chunks, 1024)` matrix per opinion, built on the GPU;
- the regex counts of `notebooks/4_language/legal_patterns.py`, one row per chunk with
  the same chunk boundaries, built from the tokenizer only.

The rows are the rows of the published cache `pttrn-io/scotus-bge-large-en-v1.5`, in
its order: its `source_idx` and `source_split` point into `coastalcph/lex_glue` at a
fixed revision. So row `i` is the same opinion in every setting, and a student can
swap one setting's cache for another's.

`--max-chunks` keeps the first chunks of an opinion and drops the rest; 0 keeps every
chunk. The regex vocabulary is fitted on the training texts only.

Building needs the GPU and the patience; publishing needs the credentials. The script
prints the publish commands at the end.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import ClassVar

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vectormesh import VectorCache, Vectorizer
from vectormesh.data.vectorizers import ChunkedRegexVectorizer

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "artefacts"
sys.path.insert(0, str(ROOT / "notebooks/4_language"))
from legal_patterns import build_scotus_pattern, harmonize_scotus

ENCODER = "BAAI/bge-large-en-v1.5"
SOURCE = "coastalcph/lex_glue"
SOURCE_REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
PUBLISHED = (
    "pttrn-io/scotus-bge-large-en-v1.5",
    "b3cb6e5c391c687a615d30fd3a99bc7bb5ad2998",
)


def vectorizer_for(chunk: int, overlap: int) -> Vectorizer:
    """A Vectorizer whose stride is `overlap` tokens: the stride is chunk // divisor."""
    divisor = chunk // overlap
    if chunk // divisor != overlap:
        sys.exit(
            f"overlap {overlap} is not {chunk} // k for a whole k; {chunk // divisor} is"
        )

    class Overlap(Vectorizer):
        STRIDE_DIVISOR: ClassVar[int] = divisor

    return Overlap(model_name=ENCODER, col_name="embed", max_length=chunk)


def opinions(
    split: str, max_chunks: int, chunk: int, overlap: int, limit: int
) -> Dataset:
    """The opinions behind one split of the published cache, in its row order."""
    path = Path(
        snapshot_download(PUBLISHED[0], repo_type="dataset", revision=PUBLISHED[1])
    )
    published = load_from_disk(str(path / split))
    assert isinstance(published, Dataset)
    rows = published.select_columns(["label", "source_idx", "source_split"])
    if limit:
        rows = rows.select(range(limit))
    src = {
        name: load_dataset(SOURCE, "scotus", split=name, revision=SOURCE_REVISION)
        for name in set(rows["source_split"])
    }
    texts = [
        src[s][i]["text"]
        for s, i in zip(rows["source_split"], rows["source_idx"], strict=True)
    ]
    if max_chunks:
        # keep the text of the first `max_chunks` chunks: one chunk adds chunk - overlap
        # tokens, and the tokenizer of the encoder decides where that ends in the text
        tok = AutoTokenizer.from_pretrained(ENCODER)
        assert tok is not None
        budget = max_chunks * (chunk - 2 - overlap) + overlap
        cut = []
        for t in texts:
            enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            cut.append(t if len(offsets) <= budget else t[: offsets[budget - 1][1]])
        texts = cut
    return rows.add_column("text", texts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--chunk", type=int, required=True, help="tokens per chunk, at most 512"
    )
    p.add_argument(
        "--overlap",
        type=int,
        required=True,
        help="tokens two neighbouring chunks share",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="keep the first chunks only; 0 keeps all",
    )
    p.add_argument("--batchsize", type=int, default=32)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="the first rows only, for a test on a laptop",
    )
    args = p.parse_args()

    setting = f"{args.chunk}-{args.overlap}" + (
        f"-max{args.max_chunks}" if args.max_chunks else ""
    )
    tag = f"scotus-{setting}" + ("-test" if args.limit else "")
    embedder = vectorizer_for(args.chunk, args.overlap)
    print(
        f"{ENCODER}: chunk {args.chunk}, stride {embedder.get_stride}, device {embedder.device}"
    )

    splits = {
        s: opinions(s, args.max_chunks, args.chunk, args.overlap, args.limit)
        for s in ("train", "test")
    }
    regex = ChunkedRegexVectorizer(
        tokenizer_name=ENCODER,
        context_size=args.chunk,
        stride=embedder.get_stride,
        pattern_builder=build_scotus_pattern,
        harmonizer=harmonize_scotus,
        training_texts=splits["train"]["text"],
        # a pattern counts once it occurs in 50 opinions; a test on a few rows keeps all
        min_doc_frequency=1 if args.limit else 50,
    )

    for split, dataset in splits.items():
        for kind, vectorizer in (("embed", embedder), ("regex", regex)):
            name = f"{tag}_{split}_{kind}"
            start = time.time()
            cache = VectorCache.create(
                cache_dir=ARTEFACTS,
                vectorizer=vectorizer,
                dataset=dataset,
                dataset_tag=name,
                remove_columns=["text"],
                vector_batch=args.batchsize,
            )
            print(
                f"  {cache.name}  ({time.time() - start:.0f}s, {len(dataset)} opinions)"
            )

    print(
        "\nnow publish them, embeddings and regex counts each as one repo:\n"
        f"  uv run python scripts/publish_cache.py --dataset {tag} --source {SOURCE} \\\n"
        f"      --train artefacts/<newest *{tag}_train_embed*> --test artefacts/<newest *{tag}_test_embed*>\n"
        f"  uv run python scripts/publish_cache.py --dataset {tag}-regex --source {SOURCE} \\\n"
        f"      --train artefacts/<newest *{tag}_train_regex*> --test artefacts/<newest *{tag}_test_regex*>"
    )


if __name__ == "__main__":
    main()
