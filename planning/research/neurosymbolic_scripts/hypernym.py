"""Is-a direction from a symmetric metric, and what a trained head recovers.

Task: given a WordNet hyponym/hypernym pair {x, y}, say which one is the parent.
Chance = 50%. Cosine similarity is symmetric so it is *provably* at 50%.
Then: can a small head on the same frozen vectors do it, and does it generalise
to a taxonomy it never saw (Levy et al. 2015 lexical-memorisation test)?
"""

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


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


def featurize(edges, model, cache):
    words = sorted({phrase(a) for a, b in edges} | {phrase(b) for a, b in edges})
    new = [w for w in words if w not in cache]
    if new:
        V = model.encode(
            new, normalize_embeddings=True, batch_size=256, show_progress_bar=False
        )
        cache.update(dict(zip(new, V)))
    child = np.stack([cache[phrase(a)] for a, b in edges])
    parent = np.stack([cache[phrase(b)] for a, b in edges])
    return child, parent


def direction_dataset(child, parent, seed=0):
    """Half the rows keep (u=child, v=parent) -> label 1 ('v is the parent'),
    half are flipped -> label 0. Feature = u - v (antisymmetric by construction)."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(child)) < 0.5
    u = np.where(flip[:, None], parent, child)
    v = np.where(flip[:, None], child, parent)
    y = (~flip).astype(int)
    return np.concatenate([u - v, u * v], axis=1), y, u, v


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    cache = {}

    roots = {
        "mammal.n.01": None,
        "plant.n.02": None,
        "vehicle.n.01": None,
        "musical_instrument.n.01": None,
    }
    data = {}
    for r in roots:
        e = subtree_edges(r)
        c, p = featurize(e, model, cache)
        X, y, u, v = direction_dataset(c, p)
        data[r] = dict(edges=e, X=X, y=y, u=u, v=v, child=c, parent=p)
        print(f"{r:28s} edges={len(e):5d}")

    # --- 1. symmetric baselines -------------------------------------------------
    print("\n--- symmetric score functions on the direction task (chance = 0.500) ---")
    for r, d in data.items():
        cos = (d["u"] * d["v"]).sum(1)  # identical for both orderings
        acc_cos = max(
            (cos > np.median(cos)).astype(int).__eq__(d["y"]).mean(),
            (cos <= np.median(cos)).astype(int).__eq__(d["y"]).mean(),
        )
        print(f"  {r:26s} best-threshold-on-cosine acc = {acc_cos:.3f}")

    # --- 2. asymmetric heuristic: vector norm before normalisation is gone, so use
    #        'distance to the centroid of the whole taxonomy' as a generality proxy
    print(
        "\n--- untrained asymmetric heuristic: closer to corpus centroid = parent ---"
    )
    for r, d in data.items():
        allv = np.concatenate([d["child"], d["parent"]])
        mu = allv.mean(0)
        mu /= np.linalg.norm(mu)
        gu, gv = d["u"] @ mu, d["v"] @ mu
        acc = ((gv > gu).astype(int) == d["y"]).mean()
        print(f"  {r:26s} acc = {acc:.3f}")

    # --- 3. trained head, in-domain random split --------------------------------
    print("\n--- logistic head on [u-v, u*v], in-domain random 50/50 split ---")
    for r, d in data.items():
        n = len(d["y"])
        idx = np.random.default_rng(1).permutation(n)
        tr, te = idx[: n // 2], idx[n // 2 :]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(d["X"][tr], d["y"][tr])
        print(f"  {r:26s} test acc = {clf.score(d['X'][te], d['y'][te]):.3f}")

    # --- 4. trained head, cross-taxonomy (lexical) split ------------------------
    print("\n--- SAME head trained on mammals, tested on other taxonomies ---")
    dm = data["mammal.n.01"]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(dm["X"], dm["y"])
    for r, d in data.items():
        if r == "mammal.n.01":
            continue
        print(f"  train=mammal -> test={r:22s} acc = {clf.score(d['X'], d['y']):.3f}")

    # --- 5. node-disjoint split within mammals ----------------------------------
    print("\n--- node-disjoint split inside mammals (no word seen in both halves) ---")
    e = dm["edges"]
    words = sorted({phrase(a) for a, b in e} | {phrase(b) for a, b in e})
    rng = np.random.default_rng(2)
    rng.shuffle(words)
    left = set(words[: len(words) // 2])
    tr = [i for i, (a, b) in enumerate(e) if phrase(a) in left and phrase(b) in left]
    te = [
        i
        for i, (a, b) in enumerate(e)
        if phrase(a) not in left and phrase(b) not in left
    ]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(dm["X"][tr], dm["y"][tr])
    print(
        f"  train n={len(tr)}  test n={len(te)}  acc = "
        f"{clf.score(dm['X'][te], dm['y'][te]):.3f}"
    )


if __name__ == "__main__":
    import time

    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
