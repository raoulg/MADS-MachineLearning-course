"""Generate the LLM's answers for the SCOTUS test split, once, so students never wait.

    uv run python build_llm_predictions.py

Needs ollama serving `qwen3.6:35b`, and takes about half an hour. Everything else in this
lesson reads the resulting ~100 KB JSON straight from `data/`, which is why it is committed
rather than published: the vectors are 318 MB and belong on the hub, this does not.

All 1400 test rows are generated, not just the ones a particular router escalates. The
threshold sweep moves the head/LLM boundary around, and every row lands on the expensive
side of it at some threshold.

Answers are written after each batch, so an interrupted run resumes where it stopped —
`CachedLLM` skips documents it already has.
"""

from pathlib import Path

from llm_client import CachedLLM
from loguru import logger
from scotus import AREAS, CACHE_REPO, LLM_CHARS
from scotus import opinions as scotus_opinions
from vectormesh import VectorCache

BATCH = 25
CACHE_PATH = Path(__file__).parent / "data" / "scotus_llm_predictions.json"


def main() -> None:
    test = VectorCache.from_hub(CACHE_REPO, split="test")
    texts = scotus_opinions(test.dataset, doc_chars=LLM_CHARS)
    logger.info(f"{len(texts)} opinions, {LLM_CHARS:,} chars each — roughly 30 minutes")

    llm = CachedLLM(CACHE_PATH, live=True)
    logger.info(f"{len(llm)} answers already cached")

    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        llm.classify(batch, AREAS)
        logger.info(
            f"{min(start + BATCH, len(texts))}/{len(texts)} -> {llm.cache_path.name}"
        )

    preds = llm.classify(texts, AREAS)
    unparsed = sum(p is None for p in preds)
    logger.info(f"done: {len(preds) - unparsed} parsed, {unparsed} unparsable")


if __name__ == "__main__":
    main()
