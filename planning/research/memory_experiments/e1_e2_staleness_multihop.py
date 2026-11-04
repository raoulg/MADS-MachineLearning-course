"""Two laptop-scale synthetic memory experiments, CPU only.

E1  knowledge-update / temporal: flat cosine vs recency-weighted vs last-write-wins index
E2  multi-hop bridge: flat cosine vs one-hop entity-graph expansion
"""

import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer

rng = random.Random(0)
t0 = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print(f"model load {time.time() - t0:.1f}s")


def embed(texts):
    return model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")


FIRST = [
    "Ada",
    "Ben",
    "Cleo",
    "Dara",
    "Eli",
    "Fay",
    "Gus",
    "Hana",
    "Ivo",
    "Jo",
    "Kai",
    "Lena",
    "Mira",
    "Noor",
    "Omar",
    "Pia",
    "Quinn",
    "Rex",
    "Sana",
    "Tomas",
]
LAST = [
    "Ostrov",
    "Bakker",
    "Ferrari",
    "Nakamura",
    "Silva",
    "Okoye",
    "Haddad",
    "Lindqvist",
]
ORGS = [
    "Acme Logistics",
    "Globex Health",
    "Initech Data",
    "Umbra Energy",
    "Soylent Foods",
    "Vandelay Imports",
    "Wonka Labs",
    "Cyberdyne Robotics",
    "Tyrell Optics",
    "Stark Metals",
]
CITIES = ["Utrecht", "Lisbon", "Tallinn", "Bergen", "Kyoto", "Cork", "Malmo", "Porto"]

# ---------------------------------------------------------------- E1
N_PEOPLE, N_UPDATES = 60, 4
docs, meta = [], []  # meta: (person, predicate, timestep)
for i in range(N_PEOPLE):
    p = f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST) % len(LAST)]}{i}"
    orgs = rng.sample(ORGS, N_UPDATES)
    for t, org in enumerate(orgs):  # the updated fact
        year = 2019 + t
        docs.append(f"In {year}, {p} started working at {org} as a project lead.")
        meta.append((p, "employer", t))
    for t in range(N_UPDATES):  # topical distractors, same person
        docs.append(
            f"{p} attended the {rng.choice(CITIES)} conference and "
            f"presented on supply chain analytics."
        )
        meta.append((p, "other", t))

queries, gold = [], []
for i in range(N_PEOPLE):
    p = f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST) % len(LAST)]}{i}"
    queries.append(f"Where does {p} work now?")
    gold.append(max(j for j, m in enumerate(meta) if m[0] == p and m[1] == "employer"))

t0 = time.time()
D = embed(docs)
Q = embed(queries)
t_embed = time.time() - t0
print(f"E1 corpus {len(docs)} docs, {len(queries)} queries, embed {t_embed:.1f}s")

S = Q @ D.T  # (nq, nd) cosine, all of it numpy
age = np.array([N_UPDATES - 1 - m[2] for m in meta], dtype="float32")
recency = np.exp(-0.9 * age)  # Generative-Agents style exponential decay


def metrics(scores, gold, mask=None):
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    order = np.argsort(-scores, axis=1)
    ranks = np.array([np.where(order[i] == g)[0][0] + 1 for i, g in enumerate(gold)])
    return dict(r1=(ranks <= 1).mean(), r5=(ranks <= 5).mean(), mrr=(1 / ranks).mean())


runs = {
    "flat cosine": S,
    "cosine + recency (w=0.3)": S + 0.3 * recency,
    "cosine + recency (w=1.0)": S + 1.0 * recency,
}
for name, sc in runs.items():
    print(f"  E1 {name:28s} {metrics(sc, gold)}")

# structured: last-write-wins over (entity, predicate) -> only newest survives the index
keep = np.zeros(len(docs), dtype=bool)
latest = {}
for j, (p, pred, t) in enumerate(meta):
    if pred == "employer":
        if (p, pred) not in latest or t > latest[(p, pred)][0]:
            latest[(p, pred)] = (t, j)
    else:
        keep[j] = True
for _, j in latest.values():
    keep[j] = True
print(f"  E1 last-write-wins index keeps {keep.sum()}/{len(docs)} docs")
print(f"  E1 {'last-write-wins index':28s} {metrics(S, gold, mask=keep[None, :])}")

# how much of top-5 is stale duplicates of the right fact?
order = np.argsort(-S, axis=1)[:, :5]
stale = np.mean(
    [
        [meta[j][1] == "employer" and j != gold[i] for j in order[i]]
        for i in range(len(gold))
    ]
)
print(f"  E1 fraction of flat top-5 that are STALE versions of the answer: {stale:.2f}")

# ---------------------------------------------------------------- E2
docs2, meta2, queries2, gold2 = [], [], [], []
for i in range(N_PEOPLE):
    p = f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST) % len(LAST)]}{i}"
    org = f"{rng.choice(ORGS)} #{i}"
    city = rng.choice(CITIES)
    bridge = len(docs2)
    docs2.append(f"{p} is employed by {org}.")
    meta2.append({p, org})
    ans = len(docs2)
    docs2.append(f"{org} has its headquarters in {city}.")
    meta2.append({org, city})
    for _ in range(3):  # noise mentioning the org
        docs2.append(f"{org} published a quarterly report on operations.")
        meta2.append({org})
    queries2.append(
        f"In which city is the headquarters of the company that employs {p}?"
    )
    gold2.append((bridge, ans))

D2 = embed(docs2)
Q2 = embed(queries2)
S2 = Q2 @ D2.T
order2 = np.argsort(-S2, axis=1)
K = 5
flat_bridge = np.mean([gold2[i][0] in order2[i][:K] for i in range(len(gold2))])
flat_ans = np.mean([gold2[i][1] in order2[i][:K] for i in range(len(gold2))])
flat_both = np.mean(
    [
        gold2[i][0] in order2[i][:K] and gold2[i][1] in order2[i][:K]
        for i in range(len(gold2))
    ]
)
print(
    f"\nE2 corpus {len(docs2)} docs. flat@{K}: bridge {flat_bridge:.2f}  "
    f"answer {flat_ans:.2f}  BOTH {flat_both:.2f}"
)

# one-hop expansion: take top-2 flat, pull in docs sharing an entity, re-rank by cosine
hop_both = []
for i in range(len(gold2)):
    seed = list(order2[i][:2])
    ents = set().union(*[meta2[j] for j in seed])
    cand = sorted(
        {j for j in range(len(docs2)) if meta2[j] & ents} | set(seed),
        key=lambda j: -S2[i, j],
    )[:K]
    hop_both.append(gold2[i][0] in cand and gold2[i][1] in cand)
print(f"E2 one-hop entity expansion@{K}: BOTH {np.mean(hop_both):.2f}")
print(f"\ntotal wall clock {time.time() - t0:.1f}s (after model load)")
