"""E5: catastrophic forgetting + replay on frozen embeddings ("embed once, train many heads").

Class-incremental: 6 newsgroup classes seen as 3 sequential tasks of 2 classes.
A small MLP head on FROZEN MiniLM vectors. Measure forgetting, then add replay.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

torch.manual_seed(0)
np.random.seed(0)
CATS = [
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.mideast",
    "comp.graphics",
    "sci.med",
    "rec.motorcycles",
]
tr = fetch_20newsgroups(
    subset="train", categories=CATS, remove=("headers", "footers", "quotes")
)
te = fetch_20newsgroups(
    subset="test", categories=CATS, remove=("headers", "footers", "quotes")
)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
t0 = time.time()
Xtr = m.encode(
    [t[:1000] for t in tr.data],
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=False,
)
Xte = m.encode(
    [t[:1000] for t in te.data],
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=False,
)
print(
    f"E5 embed {len(tr.data)}+{len(te.data)} docs on CPU: {time.time() - t0:.1f}s "
    f"(instructor does this ONCE and ships {Xtr.nbytes / 1e6:.1f}MB)"
)
ytr, yte = np.array(tr.target), np.array(te.target)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
ytr_t, yte_t = torch.tensor(ytr), torch.tensor(yte)
TASKS = [(0, 1), (2, 3), (4, 5)]


def run(replay_per_class):
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 6))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    buf_x, buf_y, acc_matrix = [], [], []
    for cls in TASKS:
        idx = np.where(np.isin(ytr, cls))[0]
        x, y = Xtr[idx], ytr_t[idx]
        if replay_per_class and buf_x:
            x = torch.cat([x] + buf_x)
            y = torch.cat([y] + buf_y)
        for _ in range(30):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 64):
                b = perm[i : i + 64]
                opt.zero_grad()
                nn.functional.cross_entropy(head(x[b]), y[b]).backward()
                opt.step()
        if replay_per_class:
            for c in cls:
                ci = np.where(ytr == c)[0][:replay_per_class]
                buf_x.append(Xtr[ci])
                buf_y.append(ytr_t[ci])
        with torch.no_grad():
            pred = head(Xte).argmax(1)
        acc_matrix.append(
            [
                (pred[np.isin(yte, t)] == yte_t[np.isin(yte, t)]).float().mean().item()
                for t in TASKS
            ]
        )
    return np.array(acc_matrix)


print("\nE5  per-task test accuracy after each task (rows = after task 1,2,3)")
for rp in [0, 5, 20, 100]:
    A = run(rp)
    final = A[-1]
    # average forgetting: best-ever minus final, over tasks seen before the last
    forget = np.mean([A[:, t].max() - A[-1, t] for t in range(len(TASKS) - 1)])
    tag = "no replay" if rp == 0 else f"replay {rp}/class"
    print(
        f"E5  {tag:16s} final per-task {np.round(final, 3)}  "
        f"avg-acc {final.mean():.3f}  avg-forgetting {forget:.3f}"
    )
print(f"E5  total wall clock {time.time() - t0:.1f}s")
