"""Build a VectorCache for one (dataset x encoder) pair, ready to publish.

    uv run python scripts/build_cache.py \
        --source nkirschi/oxford-flowers \
        --encoder facebook/dinov2-small \
        --tag flowers

Writes two timestamped folders into artefacts/ -- one per split -- which
scripts/publish_cache.py then uploads. The two halves are separate on purpose: building
needs the GPU and the patience, publishing needs the credentials.

The notebooks do NOT build caches. They download the published ones and stop with an
error if a cache is missing, because a notebook that quietly starts a twenty-minute
encoding pass is a notebook that looks broken. This script is where building lives, and
it is what you run when you want a combination nobody has published yet -- a different
encoder, a bigger subsample, another dataset.

Subsampling matches what the lesson-2 notebooks expect: shuffle first (several hub
datasets are sorted by label), then select, and attach a `source_idx` column pointing
back at the row's position in the *source* split so error analysis survives dropping the
raw column. Pass --full to cache everything instead.
"""

import argparse
import sys
import time
from pathlib import Path

from datasets import load_dataset

from vectormesh import ImageVectorizer, VectorCache
from vectormesh.data import DatasetSchema

ARTEFACTS = Path(__file__).resolve().parents[1] / "artefacts"


def prepare(source: str, split: str, n: int | None, seed: int):
    """Load one split, attach source_idx, then shuffle+select."""
    ds = load_dataset(source)
    if split not in ds:
        sys.exit(f"{source} has no split '{split}' (has: {list(ds)})")
    full = ds[split]

    # add_column, not .map(): a map over a table holding images re-encodes every one of
    # them to disk. add_column touches the Arrow schema only, so it is free.
    full = full.add_column("source_idx", list(range(len(full))))

    if n is None or n >= len(full):
        return full
    # shuffle BEFORE select: oxford-flowers is sorted by label, so a plain select would
    # silently hand back a fraction of the classes.
    return full.shuffle(seed=seed).select(range(n))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source", required=True, help="hub id, e.g. nkirschi/oxford-flowers"
    )
    p.add_argument("--encoder", required=True, help="hub id of the vision model")
    p.add_argument(
        "--tag", required=True, help="short name for the artefact folder, e.g. flowers"
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument(
        "--n-train", type=int, default=4000, help="0 or --full for everything"
    )
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--full", action="store_true", help="cache the whole splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batchsize", type=int, default=32)
    args = p.parse_args()

    n_train = None if args.full or args.n_train == 0 else args.n_train
    n_test = None if args.full or args.n_test == 0 else args.n_test

    vectorizer = ImageVectorizer(model_name=args.encoder)
    slug = args.encoder.split("/")[-1]
    print(
        f"{args.encoder} -> dim {vectorizer.get_hidden_size}, device {vectorizer.device}"
    )

    for split, n in ((args.train_split, n_train), (args.test_split, n_test)):
        dataset = prepare(args.source, split, n, args.seed)
        schema = DatasetSchema.infer(dataset)
        name = f"{args.tag}_{split}_{slug}"

        print(f"building {name}: {len(dataset)} rows, dropping '{schema.input_col}'")
        start = time.time()
        cache = VectorCache.create(
            cache_dir=ARTEFACTS,
            vectorizer=vectorizer,
            dataset=dataset,
            dataset_tag=name,
            remove_columns=[schema.input_col],
            vector_batch=args.batchsize,
        )
        elapsed = time.time() - start
        print(
            f"  {cache.name}  ({elapsed:.0f}s, {len(dataset) / elapsed:.0f} img/s)\n"
            f"  columns: {cache.column_names}"
        )

    print(
        "\nnow publish them:\n"
        f"  uv run python scripts/publish_cache.py --dataset {args.tag} "
        f"--source {args.source} \\\n"
        f"      --train artefacts/<newest {args.tag}_{args.train_split}_{slug}_*> \\\n"
        f"      --test  artefacts/<newest {args.tag}_{args.test_split}_{slug}_*>\n"
        "  (or: make publish-<name> ENCODER=" + slug + ")"
    )


if __name__ == "__main__":
    main()
