"""How much extraction error can a knowledge graph survive?

Gold chains of length h (entity_0 -> entity_1 -> ... -> entity_h). An extractor
recovers each edge with error rate p; an erroneous edge is repointed to a random
wrong target (the realistic LLM failure: a confident but wrong tail entity).
Question answered by traversal: what is entity_h given entity_0?

Compare against the dense-retrieval baseline measured in multihop.py:
answer accuracy 0.000 for h >= 2 at k <= 20 on a 720-doc corpus.
"""

import numpy as np

N_ENT = 600
N_TRIALS = 400
RNG = np.random.default_rng(0)


def run(h, p, n_trials=N_TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n_trials):
        # one gold chain of h edges over distinct entities
        chain = rng.choice(N_ENT, size=h + 1, replace=False)
        pred = [chain[0]]
        good = True
        for i in range(h):
            if rng.random() < p:  # extractor got this edge wrong
                wrong = rng.integers(N_ENT)
                while wrong == chain[i + 1]:
                    wrong = rng.integers(N_ENT)
                pred.append(wrong)
                good = False
            else:
                pred.append(chain[i + 1])
        ok += int(good and pred[-1] == chain[-1])
    return ok / n_trials


if __name__ == "__main__":
    import time

    t0 = time.time()
    ps = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    print("graph traversal answer accuracy under per-edge extraction error p")
    print("hops |" + "".join(f"  p={p:<5.2f}" for p in ps) + "   (theory (1-p)^h)")
    for h in (1, 2, 3, 4):
        row = [run(h, p, seed=h) for p in ps]
        print(
            f"  {h}  |"
            + "".join(f"  {a:<7.3f}" for a in row)
            + "   "
            + " ".join(f"{(1 - p) ** h:.2f}" for p in ps)
        )
    print("\ndense-retrieval baseline (MEASURED in multihop.py, 720 docs, k=20):")
    print("  h=1: 1.000   h=2: 0.000")
    print("\nbreak-even per-edge accuracy needed for the graph to beat a vector")
    print("baseline of 0.30 (a generous stand-in for a real single-hop-ish corpus):")
    for h in (2, 3, 4):
        print(f"  h={h}: p_max = {1 - 0.30 ** (1 / h):.3f}")
    print(f"\ntotal {time.time() - t0:.2f}s")
