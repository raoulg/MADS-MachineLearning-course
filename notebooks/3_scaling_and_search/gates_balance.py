"""Train `sparse_moe (N=64)` with a load-balancing loss and store what the notebook draws.

The same model and training as `gates_family.py`, with `balance` times the load-balancing
loss added to the training loss. Per seed the results hold the val MAE and val loss of
every epoch, the kept epoch, the val and test MAE at that epoch, and the routing on the
val windows after training (windows per expert, mean top weight). The unbalanced runs it
is compared with are those of `gates_family.py`.

    uv run python gates_balance.py           # balance 1e-2, k = 1, 3 seeds x 250 epochs
    uv run python gates_balance.py --test    # 2 seeds x 2 epochs, results/test/balance.json

Other doses and k = 2 (the exercises of the notebook) are stored under their own name:

    uv run python gates_balance.py --balance 1e-3
    uv run python gates_balance.py --k 2 --normalize --balance 3e-4

Run it from this folder. The notebook runs the default for you when `RERUN = True`.
"""

import argparse
import json
import os
import time

import torch
from gates import HERE, load_windows, routing, run

N_EXPERTS = 64

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--balance", type=float, default=1e-2)
parser.add_argument("--k", type=int, default=1)
parser.add_argument(
    "--normalize", action="store_true", help="rescale the k chosen weights to sum to 1"
)
parser.add_argument(
    "--test", action="store_true", help="a run of seconds, stored apart"
)
args = parser.parse_args()
TEST = args.test or bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))

SEEDS = (0, 1) if TEST else (0, 1, 2)
EPOCHS = 2 if TEST else 250
DEFAULT = args.balance == 1e-2 and args.k == 1 and not args.normalize
NAME = (
    "balance"
    if DEFAULT
    else f"balance_k{args.k}{'_normalized' if args.normalize else ''}_{args.balance:g}"
)
OUT = HERE / "results" / ("test" if TEST else "") / f"{NAME}.json"


def main() -> None:
    data = load_windows()
    start = time.perf_counter()
    runs = []
    for seed in SEEDS:
        r = run(
            data,
            "sparse_moe",
            seed,
            EPOCHS,
            balance=args.balance,
            n_experts=N_EXPERTS,
            k=args.k,
            normalize=args.normalize,
        )
        counts, top_weight = routing(r.model, data.xva)
        runs.append(
            {
                "seed": seed,
                "best_epoch": r.best_epoch,
                "val_mae": r.val_mae,
                "test_mae": r.test_mae,
                "val_mae_curve": r.val_mae_curve,
                "val_loss_curve": r.val_loss_curve,
                "counts": counts,
                "top_weight": top_weight,
            }
        )
        print(
            f"seed {seed}  val MAE {r.val_mae:.4f}  experts used {sum(c > 0 for c in counts)}"
            f"  top weight {top_weight:.3f}  ({time.perf_counter() - start:5.0f} s)",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "n_experts": N_EXPERTS,
                "balance": args.balance,
                "k": args.k,
                "normalize": args.normalize,
                "seeds": list(SEEDS),
                "epochs": EPOCHS,
                "minutes": round((time.perf_counter() - start) / 60, 1),
                "threads": torch.get_num_threads(),
                "runs": runs,
            }
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
