"""Generate the judgement cache that `02_evaluating_generative_output` reads.

Students do not run this — they load the published `judgements.json` and do the analysis on
a CPU in seconds. It is here so the generation is inspectable and reproducible, and so you
can point it at a different model or a different corpus.

    ollama serve                      # in another terminal
    uv run python build_judgements.py

Two things it produces, both on IMDB reviews with known labels:

- **direct**: the model's own sentiment call per review, for agreement against ground truth
- **pairwise**: "which of these two is more positive", asked **twice per pair** with the
  order swapped, which is what makes position bias measurable
"""

import json
import time
from pathlib import Path

import requests
from datasets import load_dataset
from loguru import logger

MODEL = "qwen3.6:35b"
URL = "http://localhost:11434/api/generate"
N_DIRECT = 200
N_PAIRS = 120
MAX_CHARS = 1200
OUT = Path(__file__).parent / "data" / "judgements.json"


def ask(prompt: str, n_predict: int = 8) -> str:
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": n_predict, "temperature": 0.0},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def parse_label(text: str) -> int | None:
    t = text.lower()
    if "positive" in t and "negative" not in t:
        return 1
    if "negative" in t and "positive" not in t:
        return 0
    return None


def parse_choice(text: str) -> str | None:
    t = text.strip().upper()
    for ch in ("A", "B"):
        if t.startswith(ch) or f" {ch}" in t[:6] or f"({ch})" in t[:6]:
            return ch
    return None


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=0)

    reviews, labels = [], []
    for row in ds:
        if len(reviews) >= max(N_DIRECT, N_PAIRS * 2):
            break
        reviews.append(row["text"][:MAX_CHARS])
        labels.append(int(row["label"]))

    t0 = time.time()
    direct = []
    for i in range(N_DIRECT):
        out = ask(
            f"Is the sentiment of this movie review positive or negative? "
            f"Answer with one word.\n\nReview: {reviews[i]}\n\nAnswer:"
        )
        direct.append(
            {
                "i": i,
                "truth": labels[i],
                "raw": out,
                "pred": parse_label(out),
                "n_chars": len(reviews[i]),
            }
        )
        if (i + 1) % 25 == 0:
            logger.info(f"direct {i + 1}/{N_DIRECT}  ({time.time() - t0:.0f}s)")

    # pairs with one positive and one negative, so there is a ground-truth answer
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    pairs = list(zip(pos[:N_PAIRS], neg[:N_PAIRS]))

    def compare(first: int, second: int) -> str | None:
        return parse_choice(
            ask(
                "Two movie reviews are given. Which one is more positive about the film? "
                "Answer with a single letter, A or B.\n\n"
                f"A: {reviews[first]}\n\nB: {reviews[second]}\n\nAnswer:"
            )
        )

    pairwise = []
    for k, (p, n) in enumerate(pairs):
        # order 1: positive shown as A. order 2: the same pair, swapped.
        pairwise.append(
            {
                "pos": p,
                "neg": n,
                "pos_first": compare(p, n),  # correct answer is "A"
                "neg_first": compare(n, p),  # correct answer is "B"
                "n_chars_pos": len(reviews[p]),
                "n_chars_neg": len(reviews[n]),
            }
        )
        if (k + 1) % 20 == 0:
            logger.info(f"pairs {k + 1}/{len(pairs)}  ({time.time() - t0:.0f}s)")

    OUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_direct": N_DIRECT,
                "n_pairs": len(pairs),
                "max_chars": MAX_CHARS,
                "direct": direct,
                "pairwise": pairwise,
            },
            indent=1,
        )
    )
    logger.info(
        f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
