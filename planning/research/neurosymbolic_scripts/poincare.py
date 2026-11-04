"""Hyperbolic vs Euclidean embedding of the WordNet mammal hierarchy.

Nickel & Kiela (NeurIPS 2017) claim: a tree needs far fewer hyperbolic dimensions
than Euclidean ones to be reconstructed, because hyperbolic volume grows
exponentially with radius. Test it at laptop scale on the same subtree they used.

Loss (theirs): for each edge (u,v), softmax over d(u,v) against N sampled negatives.
Riemannian SGD on the Poincare ball = Euclidean grad scaled by (1-|x|^2)^2/4,
then projected back inside the ball.
"""

import time
import numpy as np
import torch
from nltk.corpus import wordnet as wn

EPS = 1e-5


def mammal_edges():
    s = wn.synset("mammal.n.01")
    edges, seen, stack = set(), {s}, [s]
    while stack:
        x = stack.pop()
        for h in x.hyponyms():
            edges.add((h.name(), x.name()))
            if h not in seen:
                seen.add(h)
                stack.append(h)
    # transitive closure: every (descendant, ancestor) pair, as in the paper
    import collections

    parents = collections.defaultdict(set)
    for c, p in edges:
        parents[c].add(p)
    closure = set()
    for c in list(parents):
        stack, seen2 = list(parents[c]), set()
        while stack:
            a = stack.pop()
            if a in seen2:
                continue
            seen2.add(a)
            closure.add((c, a))
            stack.extend(parents[a])
    return sorted(closure)


def poincare_dist(u, v):
    sq = ((u - v) ** 2).sum(-1)
    nu = (u**2).sum(-1).clamp(max=1 - EPS)
    nv = (v**2).sum(-1).clamp(max=1 - EPS)
    x = 1 + 2 * sq / ((1 - nu) * (1 - nv)).clamp(min=EPS)
    return torch.log(x + (x**2 - 1).clamp(min=EPS).sqrt())


def euclid_dist(u, v):
    return ((u - v) ** 2).sum(-1).clamp(min=EPS).sqrt()


def train(edges, n_nodes, dim, hyperbolic, epochs=1500, n_neg=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = 1e-3 if hyperbolic else 1e-1
    E = ((torch.rand(n_nodes, dim, generator=g) - 0.5) * 2 * scale).requires_grad_(True)
    src = torch.tensor([e[0] for e in edges])
    dst = torch.tensor([e[1] for e in edges])
    dist = poincare_dist if hyperbolic else euclid_dist
    lr0 = 0.3 if hyperbolic else 0.1
    for ep in range(epochs):
        lr = lr0 * (0.1 if ep < 20 else 1.0)  # burn-in
        neg = torch.randint(0, n_nodes, (len(edges), n_neg), generator=g)
        u, v, w = E[src], E[dst], E[neg]
        d_pos = dist(u, v)
        d_neg = dist(u.unsqueeze(1), w)
        logits = -torch.cat([d_pos.unsqueeze(1), d_neg], 1)
        loss = -(logits[:, 0] - torch.logsumexp(logits, 1)).mean()
        if E.grad is not None:
            E.grad = None
        loss.backward()
        with torch.no_grad():
            gr = E.grad
            if hyperbolic:
                gr = gr * ((1 - (E**2).sum(-1, keepdim=True)) ** 2) / 4
            E -= lr * gr
            if hyperbolic:
                n = E.norm(dim=-1, keepdim=True)
                E *= torch.where(n >= 1 - EPS, (1 - EPS) / n, torch.ones_like(n))
    return E.detach(), dist


def evaluate(E, dist, edges, n_nodes):
    """Mean rank and MAP of the true parent among all nodes, as in the paper."""
    import collections

    by_src = collections.defaultdict(set)
    for a, b in edges:
        by_src[a].add(b)
    ranks, aps = [], []
    for a, targets in by_src.items():
        d = dist(E[a].unsqueeze(0), E).clone()
        d[a] = float("inf")
        order = torch.argsort(d).tolist()
        pos = [i for i, n in enumerate(order) if n in targets]
        ranks.extend([p + 1 for p in pos])
        aps.append(np.mean([(j + 1) / (p + 1) for j, p in enumerate(pos)]))
    return float(np.mean(ranks)), float(np.mean(aps))


if __name__ == "__main__":
    t0 = time.time()
    e_names = mammal_edges()
    nodes = sorted({a for a, b in e_names} | {b for a, b in e_names})
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[a], idx[b]) for a, b in e_names]
    print(f"mammal transitive closure: {len(nodes)} nodes, {len(edges)} edges")
    print(f"{'dim':>4} {'geometry':>10} {'mean rank':>10} {'MAP':>7} {'sec':>6}")
    for dim in (2, 5, 10, 20):
        for hyp in (True, False):
            t1 = time.time()
            E, dist = train(edges, len(nodes), dim, hyp)
            mr, mapv = evaluate(E, dist, edges, len(nodes))
            print(
                f"{dim:>4} {'hyperbolic' if hyp else 'euclidean':>10} "
                f"{mr:>10.2f} {mapv:>7.4f} {time.time() - t1:>6.1f}"
            )
    print(f"total {time.time() - t0:.1f}s")
