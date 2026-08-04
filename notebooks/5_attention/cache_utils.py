"""Load-or-build for VectorCache, shared by every notebook in this lesson.

`05_long_documents.ipynb` and `06_composition.ipynb` each grew their own glob-then-load-else-
create closure — three copies of the same fifteen lines by the time 06 was done, the exact
"every user invents their own tagging helper" drift raoulg/vectormesh#1 warns about. One
definition, imported by both, the way `2_convolutions/cnn.py` shares the CNN.

It also does the thing none of those three copies did safely: `VectorCache.create`'s folder
name is `{timestamp}_{dataset_tag}_{column_name}` (vectormesh 0.4.1) with no mention of which
*encoder* built it, so a plain `glob(f"*_{dataset_tag}_{column}")` returns whichever model ran
first, silently, if you swap `model_name` and rerun. vectormesh#1 proposes fixing this in the
library (fold the encoder into the folder name, raise on a metadata mismatch); until it lands,
this module does both halves here, course-side.

One more thing, found while building this: `VectorCache._vectorize` passes `dataset.map()` a
deterministic `new_fingerprint` (a hash of dataset content + vectorizer config + batch sizes —
notably *not* `cache_dir` or `dataset_tag`, vectormesh 0.4.1's fix for the flat ~21s hashing
overhead, lesson2.yml C2). That is correct for skipping recomputation, but `datasets.map()`
implements it by reusing an on-disk cache **file** keyed by that same fingerprint — and a
cache hit there skips re-running the mapped function entirely, including its Python-side
effects. `Vectorizer.chunk_sizes` is exactly such a side effect (a `Counter` the vectorizer
fills in as it runs, deliberately excluded from the fingerprint itself via
`FINGERPRINT_EXCLUDE` so re-running the same vectorizer doesn't change the hash) — so any
*second* `VectorCache.create` call over a dataset+vectorizer pair `datasets` has already seen
(even under a different `dataset_tag` or `cache_dir` — the map-cache doesn't know about
either) writes `metadata.json` with `chunk_sizes: {}`, silently, while `num_observations`
stays correct. `datasets.disable_caching()` below is the fix: it turns off `.map()`'s own
file-cache layer, which is redundant with `VectorCache`'s folder-based artefact caching
anyway — the whole reason this course caches on disk is that `VectorCache` is the artefact of
record, not `datasets`' opaque hash-named files underneath it.
"""

from pathlib import Path
from typing import Callable, Optional

import datasets
from datasets import Dataset
from loguru import logger
from vectormesh import VectorCache
from vectormesh.types import VectorMeshError

# See the module docstring: VectorCache's own fingerprint-based speedup interacts badly with
# datasets.map()'s file cache, which can silently skip a vectorizer's side effects (chunk_sizes)
# on what looks like a fresh build. VectorCache's own on-disk cache folders are this course's
# artefact of record, so disabling datasets' redundant layer costs nothing here.
datasets.disable_caching()


def _model_slug(vectorizer) -> str:
    """Filesystem-safe encoder identity, e.g. "granite-embedding-small-english-r2".

    `Vectorizer.model_name` is the swappable HF id and is what must disambiguate a cache.
    `ChunkedRegexVectorizer.model_name` defaults to a fixed constant ("chunked_regex_vectorizer")
    rather than a swappable checkpoint — correct here, since this course does not treat a regex
    feature *design* as an interchangeable "model" the way it does a pretrained encoder.
    """
    name = getattr(vectorizer, "model_name", None) or vectorizer.__class__.__name__
    return name.split("/")[-1]


def get_or_create_cache(
    cache_dir: Path,
    dataset_tag: str,
    vectorizer,
    dataset: "Dataset | Callable[[], Dataset]",
    remove_columns: Optional[list[str]] = None,
) -> VectorCache:
    """Load the cache for (dataset_tag, this vectorizer's encoder) if one is on disk; build it
    otherwise.

    `dataset` may be a ready `Dataset` or a zero-argument callable that builds one lazily — a
    cache hit then costs nothing beyond a glob, which matters when building the dataset itself
    is not free (e.g. a fresh HF `load_dataset` + `.select()`).

    Raises `VectorMeshError` on a cache hit whose stored `model_tag` disagrees with the
    vectorizer handed in — the "raise on mismatch" half of vectormesh#1. This is the case the
    encoder-in-the-tag naming below cannot itself catch: a folder from before this course
    adopted the convention, or one renamed by hand.
    """
    column = vectorizer.col_name
    tag = f"{dataset_tag}_{_model_slug(vectorizer)}"
    hits = sorted(cache_dir.glob(f"*_{tag}_{column}"))
    if hits:
        cache = VectorCache.load(hits[-1])
        stored = (cache.metadata or {}).get(column, {}).get("model_tag")
        wanted = getattr(vectorizer, "model_name", None)
        if stored and wanted and stored != wanted:
            raise VectorMeshError(
                f"{hits[-1]} was built with model_tag={stored!r}, but the vectorizer handed to "
                f"get_or_create_cache has model_name={wanted!r}. Delete the stale cache or pick "
                f"a new dataset_tag rather than silently reusing someone else's encoding."
            )
        logger.info(f"found cache: {hits[-1].name}")
        return cache

    logger.info(f"no cache for {tag}_{column}, building")
    built = dataset if isinstance(dataset, Dataset) else dataset()
    return VectorCache.create(
        cache_dir=cache_dir,
        vectorizer=vectorizer,
        dataset=built,
        dataset_tag=tag,
        remove_columns=remove_columns,
    )
