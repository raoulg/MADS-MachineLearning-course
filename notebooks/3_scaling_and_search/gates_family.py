"""Train the nine models of `01_gates_highways_moe` and store what the notebook draws.

Every model of `gates.MODEL_CONFIGS` trains once per seed for the whole budget and keeps
the epoch with the lowest val loss. Per model and seed the results hold the val MAE and
val loss of every epoch, the kept epoch, and the val and test MAE at that epoch; for the
two sparse models also the routing on the val windows, untrained and trained.

    uv run python gates_family.py           # 9 models x 3 seeds x 250 epochs, results/family.json
    uv run python gates_family.py --test    # 2 seeds x 2 epochs, results/test/family.json

Run it from this folder. The notebook runs it for you when `RERUN = True`.
"""

import argparse
import json
import os
import time

import torch
from gates import (
    CONTROL,
    HERE,
    MODEL_CONFIGS,
    load_windows,
    make_model,
    n_params,
    routing,
    run,
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--test", action="store_true", help="a run of seconds, stored apart"
)
TEST = parser.parse_args().test or bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))

SEEDS = (0, 1) if TEST else (0, 1, 2)
EPOCHS = 2 if TEST else 250
OUT = HERE / "results" / ("test" if TEST else "") / "family.json"


def main() -> None:
    data = load_windows()
    start = time.perf_counter()
    models = {}
    for label, mechanism, kwargs, control in MODEL_CONFIGS:
        runs, routes = [], []
        for seed in SEEDS:
            r = run(data, mechanism, seed, EPOCHS, **kwargs)
            runs.append(
                {
                    "seed": seed,
                    "best_epoch": r.best_epoch,
                    "val_mae": r.val_mae,
                    "test_mae": r.test_mae,
                    "val_mae_curve": r.val_mae_curve,
                    "val_loss_curve": r.val_loss_curve,
                }
            )
            if mechanism == "sparse_moe":
                torch.manual_seed(seed)  # the untrained model run() started from
                untrained = make_model(mechanism, **kwargs)
                (c0, w0), (c1, w1) = (
                    routing(untrained, data.xva),
                    routing(r.model, data.xva),
                )
                routes.append(
                    {
                        "seed": seed,
                        "untrained": {"counts": c0, "top_weight": w0},
                        "trained": {"counts": c1, "top_weight": w1},
                    }
                )
            print(
                f"{label:20s} seed {seed}  val MAE {r.val_mae:.4f}  epoch {r.best_epoch:3d}"
                f"  ({time.perf_counter() - start:5.0f} s)",
                flush=True,
            )
        models[label] = {
            "mechanism": mechanism,
            "kwargs": kwargs,
            "control": CONTROL[label],
            "params": n_params(make_model(mechanism, **kwargs)),
            "runs": runs,
        }
        if routes:
            models[label]["routing"] = routes
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "epochs": EPOCHS,
                "minutes": round((time.perf_counter() - start) / 60, 1),
                "threads": torch.get_num_threads(),
                "models": models,
            }
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
