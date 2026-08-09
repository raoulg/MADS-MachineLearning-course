"""Build the SCOTUS embedding cache this lesson routes on.

Students never run this — they call `VectorCache.from_hub(...)` and get the vectors. It is
here so the artefact is reproducible and so you can point it at a different encoder.

    uv run python build_scotus_cache.py

It only builds. Publishing goes through the repo's existing tool, which derives the repo
id and writes a dataset card from the cache metadata:

    uv run python scripts/publish_cache.py --dataset scotus \
        --source coastalcph/lex_glue --train <train_folder> --test <test_folder> --dry-run

Four choices in here were measured on a 2600-document subset rather than assumed:

* **512-token chunks** rather than a long context window. The encoder accepts far more,
  but a 2048 window cost 3x the encoding time for *no* accuracy gain — attention is
  quadratic in sequence length, while more chunks is linear.
* **24,000 characters per opinion** rather than 6,000. Feeding four times as much of the
  document is worth ~2 points of accuracy and ~7 points of recall on the rare classes,
  which is the part this lesson cares about.
* **bge-large over granite-small**: accuracy 0.677 vs 0.657, rare-class recall 0.447 vs
  0.420, head-vs-tail separability 0.834 vs 0.809. It costs ~5.5x the encoding time, which
  is paid once, here, and never by a student.
* **The chunk axis is kept.** At this document budget an opinion is ~12 chunks, so the
  published cache runs to a few hundred MB rather than a few tens. That buys the option of
  aggregating with something other than a mean — lesson 4's subject, on this data.
"""

from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from vectormesh import VectorCache, Vectorizer
from vectormesh.data.vectorizers import detect_device

ENCODER = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 512
DOC_CHARS = 24_000
CACHE_DIR = Path(__file__).resolve().parents[2] / "artefacts"


def with_provenance(ds, split_name: str):
    """Stamp each row with where it came from, before anything reorders it.

    A cached vector is only useful to this lesson if you can get the opinion *text* back:
    the LLM half of the router is fed text, not embeddings. `source_idx` alone cannot do
    that here, because the cache draws on two LexGLUE splits, so the split travels with it.
    """
    return ds.add_column("source_idx", list(range(len(ds)))).add_column(
        "source_split", [split_name] * len(ds)
    )


def build(split_name: str, ds) -> VectorCache:
    vec = Vectorizer(
        model_name=ENCODER,
        col_name="embed",
        input_col="text",
        max_length=MAX_TOKENS,
        device=detect_device(),
    )
    payload = Dataset.from_dict(
        {
            "text": [r["text"][:DOC_CHARS] for r in ds],
            "label": [int(r["label"]) for r in ds],
            "source_idx": [int(r["source_idx"]) for r in ds],
            "source_split": [r["source_split"] for r in ds],
        }
    )
    cache = VectorCache.get_or_create(
        CACHE_DIR, f"scotus_{split_name}", vec, payload, remove_columns=["text"]
    )
    y = np.asarray(cache.dataset.with_format("numpy")["label"])
    logger.info(
        f"{split_name}: {len(y)} opinions, "
        f"per-class {np.bincount(y, minlength=13).tolist()}"
    )
    return cache


def main() -> None:
    tr = load_dataset("coastalcph/lex_glue", "scotus", split="train")
    va = load_dataset("coastalcph/lex_glue", "scotus", split="validation")
    te = load_dataset("coastalcph/lex_glue", "scotus", split="test")

    # train + validation is every labelled document LexGLUE gives you outside the test
    # split: 6400 of them. The rare classes stay rare -- Attorneys has 35 in the entire
    # dataset -- and that scarcity is what this lesson is about, not an artefact of
    # subsampling.
    full_train = concatenate_datasets(
        [with_provenance(tr, "train"), with_provenance(va, "validation")]
    ).shuffle(seed=0)
    logger.info(f"train+validation {len(full_train)}, test {len(te)}")

    build("train", full_train)
    build("test", with_provenance(te, "test"))
    logger.info(
        "built. The two folders above are what scripts/publish_cache.py wants; "
        "run it with --dry-run first to inspect the repo id and the card."
    )


if __name__ == "__main__":
    main()
