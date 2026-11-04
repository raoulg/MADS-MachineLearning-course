"""Synthetic 2-hop retrieval: what cosine can and cannot reach.

Corpus of atomic facts:
    A: "<person> is employed by <company>."
    B: "<company> has its headquarters in <city>."
Query: "In which city does <person> work?"

Doc A is lexically and semantically anchored to the query (the person's name is in both).
Doc B shares no content word with the query at all. Answering needs both.
"""

import random
import numpy as np
from sentence_transformers import SentenceTransformer

rng = random.Random(0)

FIRST = [
    "Alden",
    "Brisa",
    "Corin",
    "Dahlia",
    "Emeric",
    "Fenna",
    "Gethin",
    "Halcyon",
    "Ivo",
    "Juniper",
    "Kestrel",
    "Lior",
    "Maren",
    "Nerio",
    "Orla",
    "Perrin",
    "Quilla",
    "Rowan",
    "Sable",
    "Tamsin",
    "Ulric",
    "Verity",
    "Wrenna",
    "Xanthe",
    "Yorick",
    "Zelia",
    "Anwen",
    "Bramwell",
    "Caius",
    "Delphine",
]
LAST = [
    "Ashgrove",
    "Bellweather",
    "Cinderhall",
    "Drakemoor",
    "Everline",
    "Fallowmere",
    "Grimsby",
    "Hartwood",
    "Ironvale",
    "Jessamine",
    "Kelbrook",
    "Lindenhall",
    "Marchetti",
    "Northgate",
    "Oakhurst",
    "Pemberton",
    "Quillon",
    "Ravenscroft",
    "Stonebury",
    "Thornfield",
    "Umberlee",
    "Vasseur",
    "Whitlock",
    "Yarrow",
]
COMPS = [
    "Kaledon",
    "Torvex",
    "Mirabec",
    "Halcyne",
    "Verrado",
    "Quintaro",
    "Nyxis",
    "Oberith",
    "Palvantis",
    "Rhodex",
    "Sylvara",
    "Trellon",
    "Umbriel",
    "Valken",
    "Wyndar",
    "Xelora",
    "Yarrowen",
    "Zephrix",
    "Astrolin",
    "Brenmark",
]
CITIES = [
    "Lisbon",
    "Osaka",
    "Montevideo",
    "Reykjavik",
    "Nairobi",
    "Bergen",
    "Chengdu",
    "Porto",
    "Tallinn",
    "Adelaide",
    "Casablanca",
    "Bratislava",
]


def build(n_people, seed=0):
    r = random.Random(seed)
    people, seen = [], set()
    while len(people) < n_people:
        p = f"{r.choice(FIRST)} {r.choice(LAST)}"
        if p not in seen:
            seen.add(p)
            people.append(p)
    comps = [
        f"{c} {s}"
        for c in COMPS
        for s in ("Systems", "Dynamics", "Holdings", "Labs", "Partners", "Analytics")
    ]
    r.shuffle(comps)
    comps = comps[: max(20, n_people // 3)]
    comp_city = {c: r.choice(CITIES) for c in comps}
    p_comp = {p: r.choice(comps) for p in people}

    docs, meta = [], []
    for p in people:
        docs.append(f"{p} is employed by {p_comp[p]}.")
        meta.append(("A", p))
    for c in comps:
        docs.append(f"{c} has its headquarters in {comp_city[c]}.")
        meta.append(("B", c))
    queries = [f"In which city does {p} work?" for p in people]
    return docs, meta, queries, people, p_comp, comp_city


def run(n_people, model):
    docs, meta, queries, people, p_comp, comp_city = build(n_people)
    D = model.encode(
        docs, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    )
    Q = model.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    )
    S = Q @ D.T  # (n_q, n_docs)
    order = np.argsort(-S, axis=1)
    idxA = {p: i for i, (t, k) in enumerate(meta) if t == "A" for p in [k]}
    idxB = {c: i for i, (t, k) in enumerate(meta) if t == "B" for c in [k]}

    out = {}
    for k in (1, 5, 10, 20):
        top = order[:, :k]
        hitA = np.mean([idxA[p] in top[i] for i, p in enumerate(people)])
        hitB = np.mean([idxB[p_comp[p]] in top[i] for i, p in enumerate(people)])
        both = np.mean(
            [
                (idxA[p] in top[i]) and (idxB[p_comp[p]] in top[i])
                for i, p in enumerate(people)
            ]
        )
        out[k] = (hitA, hitB, both)
    # graph traversal: doc A -> company -> doc B, always both docs, k=2
    return out, len(docs)


if __name__ == "__main__":
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    import time

    for n in (100, 300, 600):
        t0 = time.time()
        out, ndocs = run(n, m)
        print(f"\nn_people={n}  n_docs={ndocs}  ({time.time() - t0:.1f}s)")
        print("  k | recall hop-1 doc | recall hop-2 doc | both (answerable)")
        for k, (a, b, both) in out.items():
            print(f" {k:>2} |      {a:.3f}       |      {b:.3f}       |   {both:.3f}")
