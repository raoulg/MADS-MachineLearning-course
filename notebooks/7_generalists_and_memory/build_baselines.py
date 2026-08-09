"""Generate `data/baselines.json`: the generalist against a trained head, on two datasets.

A zero-shot accuracy on its own is not a result. This produces the baseline it has to beat
— a linear probe on cached embeddings — on the *same items*, plus the wall-clock cost of
each, so `02_evaluating_generative_output` can report a comparison instead of a number.

Two datasets on purpose:

- **IMDB** — short reviews, binary. The easy case, where a cheap head should be competitive.
- **SCOTUS** (LexGLUE) — Supreme Court opinions, 13 issue areas, median opinion far longer
  than any context window here. The hard case, and the one lesson 4 built a head for.

    ollama serve
    uv run python build_baselines.py
"""

import json
import time
from pathlib import Path

import numpy as np
import requests
from datasets import Dataset, load_dataset
from loguru import logger
from vectormesh import VectorCache, Vectorizer
from vectormesh.data.vectorizers import detect_device

MODEL = "qwen3.6:35b"
URL = "http://localhost:11434/api/generate"
EMBEDDER = "ibm-granite/granite-embedding-small-english-r2"  # the encoder lesson 4 uses
MAX_TOKENS = 512
CACHE_DIR = Path(__file__).resolve().parents[2] / "artefacts"
OUT = Path(__file__).parent / "data" / "baselines.json"

N_EVAL = 200
N_TRAIN = 800
IMDB_CHARS = 1200
SCOTUS_CHARS = 6000

# The loaded dataset exposes 13 classes named '1'..'13'; the LexGLUE card describes 14
# issue areas (the 14th, "Private Action", is absent here). Index i -> issue area i+1.
SCOTUS_AREAS = [
    "Criminal Procedure",
    "Civil Rights",
    "First Amendment",
    "Due Process",
    "Privacy",
    "Attorneys",
    "Unions",
    "Economic Activity",
    "Judicial Power",
    "Federalism",
    "Interstate Relations",
    "Federal Taxation",
    "Miscellaneous",
]


def ask(prompt: str, n_predict: int) -> str:
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": n_predict, "temperature": 0.0},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def probe(Xtr, ytr, Xte, yte, lam=1e-2):
    """Closed-form ridge on one-hot targets. Deterministic, and a second to fit."""
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]
    Xte = np.c_[Xte, np.ones(len(Xte))]
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None, :]).astype(float)
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return float((classes[np.argmax(Xte @ W, 1)] == yte).mean())


def embed(tag: str, texts: list[str], labels: list[int]) -> np.ndarray:
    """Encode with the course's own Vectorizer, cached on disk like every other lesson.

    A document longer than the 512-token window becomes several chunks, so the cache
    column is (chunks, dim) per row. Mean-pool over the chunk axis -- lesson 4's
    `MeanAggregator`, the simplest thing that turns a document into one vector. Lesson 4
    shows that masked and attention aggregation do better; this is a baseline, not the
    best head anyone could build.
    """
    vec = Vectorizer(
        model_name=EMBEDDER,
        col_name="embed",
        input_col="text",
        max_length=MAX_TOKENS,
        device=detect_device(),
    )
    ds = Dataset.from_dict({"text": texts, "label": labels})
    cache = VectorCache.get_or_create(CACHE_DIR, tag, vec, ds, remove_columns=["text"])
    rows = cache.dataset.with_format("numpy")["embed"]
    return np.stack(
        [np.asarray(r, dtype="float64").reshape(-1, 384).mean(0) for r in rows]
    )


def run(name, texts_tr, y_tr, texts_te, y_te, prompt_fn, parse_fn, n_predict):
    logger.info(f"[{name}] embedding {len(texts_tr) + len(texts_te)} documents")
    t0 = time.time()
    Etr = embed(f"{name}_train", texts_tr, y_tr)
    Ete = embed(f"{name}_eval", texts_te, y_te)
    embed_s = time.time() - t0

    t0 = time.time()
    acc_probe = probe(Etr, np.asarray(y_tr), Ete, np.asarray(y_te))
    probe_s = time.time() - t0

    logger.info(
        f"[{name}] probe {acc_probe:.4f}  (embed {embed_s:.0f}s, fit {probe_s:.2f}s)"
    )

    t0, preds, raws = time.time(), [], []
    for i, t in enumerate(texts_te):
        out = ask(prompt_fn(t), n_predict)
        raws.append(out)
        preds.append(parse_fn(out))
        if (i + 1) % 50 == 0:
            logger.info(
                f"[{name}] llm {i + 1}/{len(texts_te)}  ({time.time() - t0:.0f}s)"
            )
    llm_s = time.time() - t0

    ok = [(p, t) for p, t in zip(preds, y_te) if p is not None]
    acc_llm = float(np.mean([p == t for p, t in ok])) if ok else 0.0
    logger.info(
        f"[{name}] llm {acc_llm:.4f} on {len(ok)}/{len(y_te)} parsed  ({llm_s:.0f}s)"
    )

    return {
        "n_eval": len(y_te),
        "n_train": len(y_tr),
        "n_classes": int(len(set(y_te))),
        "majority": float(max(np.bincount(np.asarray(y_te)) / len(y_te))),
        "probe_acc": acc_probe,
        "embed_seconds": embed_s,
        "probe_fit_seconds": probe_s,
        "llm_acc": acc_llm,
        "llm_parsed": len(ok),
        "llm_seconds": llm_s,
        "llm_raw_sample": raws[:5],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {"model": MODEL, "embedder": EMBEDDER, "max_tokens": MAX_TOKENS}

    # ---------- IMDB: the same 200 items 02 already has LLM judgements for ----------
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=0)
    texts = [r["text"][:IMDB_CHARS] for r in ds.select(range(N_EVAL + N_TRAIN))]
    labels = [int(r["label"]) for r in ds.select(range(N_EVAL + N_TRAIN))]
    results["imdb"] = run(
        "imdb",
        texts[N_EVAL:],
        labels[N_EVAL:],
        texts[:N_EVAL],
        labels[:N_EVAL],
        lambda t: (
            f"Is the sentiment of this movie review positive or negative? "
            f"Answer with one word.\n\nReview: {t}\n\nAnswer:"
        ),
        lambda s: (
            1
            if ("positive" in s.lower() and "negative" not in s.lower())
            else (0 if "negative" in s.lower() else None)
        ),
        8,
    )

    # ---------- SCOTUS: long documents, 13 classes ----------
    tr = load_dataset("coastalcph/lex_glue", "scotus", split="train").shuffle(seed=0)
    te = load_dataset("coastalcph/lex_glue", "scotus", split="test").shuffle(seed=0)
    s_tr = [r["text"][:SCOTUS_CHARS] for r in tr.select(range(N_TRAIN))]
    y_tr = [int(r["label"]) for r in tr.select(range(N_TRAIN))]
    s_te = [r["text"][:SCOTUS_CHARS] for r in te.select(range(N_EVAL))]
    y_te = [int(r["label"]) for r in te.select(range(N_EVAL))]

    menu = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(SCOTUS_AREAS))

    def parse_area(s: str):
        digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
        for d in digits:
            v = int(d)
            if 1 <= v <= len(SCOTUS_AREAS):
                return v - 1
        low = s.lower()
        for i, a in enumerate(SCOTUS_AREAS):
            if a.lower() in low:
                return i
        return None

    results["scotus"] = run(
        "scotus",
        s_tr,
        y_tr,
        s_te,
        y_te,
        lambda t: (
            "Classify this US Supreme Court opinion into one of these issue areas. "
            f"Answer with the number only.\n\n{menu}\n\nOpinion: {t}\n\nAnswer:"
        ),
        parse_area,
        8,
    )
    results["scotus"]["areas"] = SCOTUS_AREAS
    results["scotus_chars"] = SCOTUS_CHARS
    results["imdb_chars"] = IMDB_CHARS

    OUT.write_text(json.dumps(results, indent=1))
    logger.info(f"wrote {OUT}")


if __name__ == "__main__":
    main()
