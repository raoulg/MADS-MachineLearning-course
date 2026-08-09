"""How much labelled data does a cheap head need to catch the generalist?

`build_baselines.py` scores a zero-shot LLM against one probe trained on 800 examples,
which is a single point on a curve and therefore easy to argue with. This produces the
curve: probe accuracy against training-set size, on the same evaluation items, with the
LLM's zero-shot score as a horizontal line.

That turns "is the LLM better?" into the question a practitioner actually has: **how many
labels would I have to collect before the thing that costs nothing to run wins?**

    ollama serve            # only needed by build_baselines.py, not by this script
    uv run python build_learning_curve.py
"""

import json
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from vectormesh import VectorCache, Vectorizer
from vectormesh.data.vectorizers import detect_device

EMBEDDER = "ibm-granite/granite-embedding-small-english-r2"
MAX_TOKENS = 512
DIM = 384
CACHE_DIR = Path(__file__).parent / "artefacts"
OUT = Path(__file__).parent / "data" / "learning_curve.json"

N_EVAL = 200
IMDB_CHARS, SCOTUS_CHARS = 1200, 6000
IMDB_SIZES = [100, 400, 1600, 6400]
SCOTUS_SIZES = [100, 400, 1600, 4800]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def embed(tag: str, texts: list[str], labels: list[int]) -> np.ndarray:
    """Encode with the course's own Vectorizer, then mean-pool chunks and L2-normalise.

    Normalising matters more than it looks: these embeddings have norm ~15, and an
    unnormalised ridge fit on them scores several points worse.
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
    X = np.stack(
        [np.asarray(r, dtype="float64").reshape(-1, DIM).mean(0) for r in rows]
    )
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ridge_fit(X, y, lam):
    A = np.c_[X, np.ones(len(X))]
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(float)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return W, classes


def ridge_score(W, classes, X, y):
    return float((classes[np.argmax(np.c_[X, np.ones(len(X))] @ W, 1)] == y).mean())


def probe_at(Xtr, ytr, Xte, yte, n, seed=0):
    """Fit on `n` examples, choosing the ridge penalty on a held-out slice of the train set."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))[:n]
    cut = max(20, int(0.8 * n))
    fit_i, val_i = idx[:cut], idx[cut:]
    best, best_lam = -1.0, LAMBDAS[0]
    for lam in LAMBDAS:
        W, cls = ridge_fit(Xtr[fit_i], ytr[fit_i], lam)
        s = ridge_score(W, cls, Xtr[val_i], ytr[val_i])
        if s > best:
            best, best_lam = s, lam
    W, cls = ridge_fit(Xtr[idx], ytr[idx], best_lam)  # refit on all n with the winner
    return ridge_score(W, cls, Xte, yte), best_lam


def curve(name, texts_tr, y_tr, texts_te, y_te, sizes):
    t0 = time.time()
    Xtr = embed(f"{name}_curve_train", texts_tr, y_tr)
    Xte = embed(f"{name}_eval", texts_te, y_te)
    ytr, yte = np.asarray(y_tr), np.asarray(y_te)
    logger.info(f"[{name}] embedded {len(Xtr)} + {len(Xte)} in {time.time() - t0:.0f}s")

    points = []
    for n in sizes:
        if n > len(Xtr):
            continue
        accs = [probe_at(Xtr, ytr, Xte, yte, n, seed=s)[0] for s in (0, 1, 2)]
        points.append({"n": n, "acc": float(np.mean(accs)), "std": float(np.std(accs))})
        logger.info(
            f"[{name}] n={n:5d}  probe {np.mean(accs):.4f} +- {np.std(accs):.4f}"
        )
    return {
        "points": points,
        "n_eval": len(yte),
        "majority": float(max(np.bincount(yte) / len(yte))),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {"embedder": EMBEDDER, "max_tokens": MAX_TOKENS}

    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=0)
    n_need = N_EVAL + max(IMDB_SIZES)
    rows = ds.select(range(n_need))
    texts = [r["text"][:IMDB_CHARS] for r in rows]
    labels = [int(r["label"]) for r in rows]
    results["imdb"] = curve(
        "imdb",
        texts[N_EVAL:],
        labels[N_EVAL:],
        texts[:N_EVAL],
        labels[:N_EVAL],
        IMDB_SIZES,
    )

    tr = load_dataset("coastalcph/lex_glue", "scotus", split="train").shuffle(seed=0)
    te = load_dataset("coastalcph/lex_glue", "scotus", split="test").shuffle(seed=0)
    n_tr = min(max(SCOTUS_SIZES), len(tr))
    s_tr = [r["text"][:SCOTUS_CHARS] for r in tr.select(range(n_tr))]
    y_tr = [int(r["label"]) for r in tr.select(range(n_tr))]
    s_te = [r["text"][:SCOTUS_CHARS] for r in te.select(range(N_EVAL))]
    y_te = [int(r["label"]) for r in te.select(range(N_EVAL))]
    results["scotus"] = curve("scotus", s_tr, y_tr, s_te, y_te, SCOTUS_SIZES)

    OUT.write_text(json.dumps(results, indent=1))
    logger.info(f"wrote {OUT}")


if __name__ == "__main__":
    main()
