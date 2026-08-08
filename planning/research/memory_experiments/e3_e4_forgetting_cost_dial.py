"""E3: the cost of forgetting.  E4: the difficulty dial on multi-hop."""

import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer

rng = random.Random(0)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb = lambda t: model.encode(
    t, batch_size=256, normalize_embeddings=True, show_progress_bar=False
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
N, U = 60, 4
name = lambda i: f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST) % len(LAST)]}{i}"

# ------------------------------------------------- E3: recency helps NOW, hurts THEN
docs, meta = [], []
for i in range(N):
    p = name(i)
    for t, org in enumerate(rng.sample(ORGS, U)):
        docs.append(f"In {2019 + t}, {p} started working at {org} as a project lead.")
        meta.append((p, t))
    for t in range(U):
        docs.append(f"{p} attended the {rng.choice(CITIES)} conference on analytics.")
        meta.append((p, -1))

qs_now = [f"Where does {name(i)} work now?" for i in range(N)]
gold_now = [
    max(j for j, m in enumerate(meta) if m[0] == name(i) and m[1] >= 0)
    for i in range(N)
]
qs_past = [f"Where did {name(i)} work in 2019?" for i in range(N)]
gold_past = [
    min(j for j, m in enumerate(meta) if m[0] == name(i) and m[1] >= 0)
    for i in range(N)
]

t0 = time.time()
D = emb(docs)
age = np.array([U - 1 - m[1] if m[1] >= 0 else 0 for m in meta], dtype="float32")
rec = np.exp(-0.9 * age)


def mrr_r1(S, gold):
    o = np.argsort(-S, axis=1)
    r = np.array([np.where(o[i] == g)[0][0] + 1 for i, g in enumerate(gold)])
    return (r <= 1).mean(), (1 / r).mean()


print("E3  weight |  'work now' r1/mrr  |  'work in 2019' r1/mrr")
for w in [0.0, 0.1, 0.3, 1.0]:
    a = mrr_r1(emb(qs_now) @ D.T + w * rec, gold_now)
    b = mrr_r1(emb(qs_past) @ D.T + w * rec, gold_past)
    print(
        f"E3   w={w:<4}  |   {a[0]:.3f} / {a[1]:.3f}       |   {b[0]:.3f} / {b[1]:.3f}"
    )

# hard forgetting: delete stale rows entirely
keep = np.zeros(len(docs), bool)
latest = {}
for j, (p, t) in enumerate(meta):
    if t < 0:
        keep[j] = True
    elif p not in latest or t > latest[p][0]:
        latest[p] = (t, j)
for _, j in latest.values():
    keep[j] = True
mask = keep[None, :]
Sn = np.where(mask, emb(qs_now) @ D.T, -1e9)
Sp = np.where(mask, emb(qs_past) @ D.T, -1e9)
gp_ok = np.mean([keep[g] for g in gold_past])
print(
    f"E3   hard-delete stale (keeps {keep.sum()}/{len(docs)}): "
    f"now r1={mrr_r1(Sn, gold_now)[0]:.3f}   2019 r1={mrr_r1(Sp, gold_past)[0]:.3f} "
    f"(gold still in index for {gp_ok:.0%} of past questions)"
)

# ------------------------------------------------- E4: multi-hop difficulty dial
print("\nE4  leak = fraction of the bridge entity's tokens visible in the query")
for leak in [0.0, 0.5, 1.0]:
    d2, m2, q2, g2 = [], [], [], []
    for i in range(N):
        p, city = name(i), rng.choice(CITIES)
        org = f"{rng.choice(ORGS)} #{i}"
        head = org.split()[0]
        d2.append(f"{p} is employed by {org}.")
        m2.append({p, org})
        b = len(d2) - 1
        d2.append(f"{org} has its headquarters in {city}.")
        m2.append({org, city})
        a = len(d2) - 1
        for _ in range(3):
            d2.append(f"{org} published a quarterly report.")
            m2.append({org})
        hint = (
            ""
            if leak == 0
            else (f" (a {head} company)" if leak == 0.5 else f" ({org})")
        )
        q2.append(f"In which city is the HQ of the company that employs {p}{hint}?")
        g2.append((b, a))
    D2, Q2 = emb(d2), emb(q2)
    S2 = Q2 @ D2.T
    o2 = np.argsort(-S2, axis=1)
    flat = np.mean([g2[i][0] in o2[i][:5] and g2[i][1] in o2[i][:5] for i in range(N)])
    hop = []
    for i in range(N):
        seed = list(o2[i][:2])
        ents = set().union(*[m2[j] for j in seed])
        cand = sorted(
            {j for j in range(len(d2)) if m2[j] & ents} | set(seed),
            key=lambda j: -S2[i, j],
        )[:5]
        hop.append(g2[i][0] in cand and g2[i][1] in cand)
    print(
        f"E4   leak={leak:<4} flat both@5={flat:.2f}   one-hop both@5={np.mean(hop):.2f}"
        f"   gap={np.mean(hop) - flat:+.2f}"
    )
print(f"\nE3+E4 wall clock {time.time() - t0:.1f}s")
