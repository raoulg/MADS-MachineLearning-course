"""Two things:

(a) confound check for the 'distance to centroid = generality' hypernym heuristic:
    does a trivial surface feature (character length / token count of the phrase)
    do just as well?  If yes, the geometry claim is not earning its keep.

(b) negation / boolean constraint: does cosine retrieval score *below* the base rate
    when the query says 'not X'?  Symbolic filter is 1.0 by construction.
"""

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------------------------- (a) confound
def subtree_edges(root):
    s = wn.synset(root)
    edges, seen, stack = set(), {s}, [s]
    while stack:
        x = stack.pop()
        for h in x.hyponyms():
            edges.add((h, x))
            if h not in seen:
                seen.add(h)
                stack.append(h)
    return sorted(edges, key=lambda e: (e[0].name(), e[1].name()))


def phrase(s):
    return s.lemmas()[0].name().replace("_", " ")


def confound(model):
    print("=== (a) is 'closer to centroid' just 'shorter string'? ===")
    print(
        f"{'taxonomy':26s} {'centroid':>9s} {'shorter':>9s} {'fewer tok':>10s} "
        f"{'centroid|len-tied':>18s}"
    )
    for root in (
        "mammal.n.01",
        "plant.n.02",
        "vehicle.n.01",
        "musical_instrument.n.01",
    ):
        e = subtree_edges(root)
        words = sorted({phrase(a) for a, b in e} | {phrase(b) for a, b in e})
        V = dict(
            zip(
                words,
                model.encode(
                    words,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=256,
                ),
            )
        )
        child = np.stack([V[phrase(a)] for a, b in e])
        parent = np.stack([V[phrase(b)] for a, b in e])
        rng = np.random.default_rng(0)
        flip = rng.random(len(e)) < 0.5
        u = np.where(flip[:, None], parent, child)
        v = np.where(flip[:, None], child, parent)
        y = (~flip).astype(int)
        cu = [phrase(b) if f else phrase(a) for (a, b), f in zip(e, flip)]
        cv = [phrase(a) if f else phrase(b) for (a, b), f in zip(e, flip)]

        allv = np.concatenate([child, parent])
        mu = allv.mean(0)
        mu /= np.linalg.norm(mu)
        cen = ((v @ mu > u @ mu).astype(int) == y).mean()
        shorter = np.mean([(len(b) < len(a)) == bool(t) for a, b, t in zip(cu, cv, y)])
        fewtok = np.mean(
            [(b.count(" ") < a.count(" ")) == bool(t) for a, b, t in zip(cu, cv, y)]
        )
        tie = [i for i in range(len(e)) if cu[i].count(" ") == cv[i].count(" ")]
        cen_tied = ((v[tie] @ mu > u[tie] @ mu).astype(int) == y[tie]).mean()
        print(
            f"{root:26s} {cen:9.3f} {shorter:9.3f} {fewtok:10.3f} "
            f"{cen_tied:12.3f} (n={len(tie)})"
        )


# ----------------------------------------------------------------- (b) negation
ITEMS = [
    "laptop",
    "jacket",
    "backpack",
    "camera",
    "bicycle",
    "headphones",
    "tent",
    "watch",
    "keyboard",
    "mattress",
]
ATTRS = [
    "waterproof",
    "lightweight",
    "wireless",
    "rechargeable",
    "foldable",
    "insulated",
    "adjustable",
    "shockproof",
]


def negation(model, n_docs=800, k=10, seed=0):
    print("\n=== (b) 'A but not B' retrieval ===")
    rng = np.random.default_rng(seed)
    docs, tags = [], []
    for i in range(n_docs):
        item = ITEMS[rng.integers(len(ITEMS))]
        m = rng.integers(2, 5)
        att = sorted(rng.choice(len(ATTRS), size=m, replace=False))
        tags.append(set(att))
        docs.append(f"The {item} is {', '.join(ATTRS[a] for a in att)}.")
    D = model.encode(
        docs, normalize_embeddings=True, show_progress_bar=False, batch_size=256
    )

    rows = []
    for a in range(len(ATTRS)):
        for b in range(len(ATTRS)):
            if a == b:
                continue
            q = f"a product that is {ATTRS[a]} but not {ATTRS[b]}"
            ok = np.array([(a in t) and (b not in t) for t in tags])
            if ok.sum() < k:
                continue
            qv = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[
                0
            ]
            top = np.argsort(-(D @ qv))[:k]
            rows.append(
                (
                    ok[top].mean(),  # full constraint
                    np.mean([a in tags[i] for i in top]),  # positive half
                    np.mean([b not in tags[i] for i in top]),  # negative half
                    ok.mean(),  # base rate
                    np.mean([b not in t for t in tags]),
                )
            )  # base rate of ¬B
    R = np.array(rows)
    print(f"  {len(R)} queries, {n_docs} docs, k={k}")
    print(
        f"  precision@{k} full constraint (A and not B) : {R[:, 0].mean():.3f}"
        f"   base rate {R[:, 3].mean():.3f}"
    )
    print(f"  precision@{k} positive half  (A)           : {R[:, 1].mean():.3f}")
    print(
        f"  precision@{k} negative half  (not B)       : {R[:, 2].mean():.3f}"
        f"   base rate {R[:, 4].mean():.3f}"
    )
    print("  symbolic filter over the same metadata      : 1.000")


if __name__ == "__main__":
    import time

    t0 = time.time()
    m = SentenceTransformer(MODEL, device="cpu")
    confound(m)
    negation(m)
    print(f"\ntotal {time.time() - t0:.1f}s")
