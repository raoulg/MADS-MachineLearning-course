"""Measure EuroSAT's data axis on `dinov2-small` and store what `02_scaling_curves` draws.

The protocol of the notebook's DTD data axis: the baseline head, `NeuralNet(384, 10)`, is
trained for 80 epochs per seed and its epoch count is chosen on valid by the notebook's
rule (the epoch with the highest mean valid accuracy). Then the head is trained on 5%, 10%,
25%, 50% and 100% of the train rows (balanced prefixes), at equal epochs and at equal
steps. The results hold, per seed, the valid accuracy of every epoch of the baseline runs,
and per fraction the steps and the test accuracy of each arm.

    uv run python eurosat_data.py           # 3 seeds, results/eurosat_data.json
    uv run python eurosat_data.py --test    # 2 seeds x 2 epochs, results/test/eurosat_data.json

Run it from this folder. The notebook runs it for you when `RERUN = True`.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from scaling import choose_epochs, data_axis, load, repeat
from vectormesh.components import NeuralNet

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--test", action="store_true", help="a run of seconds, stored apart"
)
args = parser.parse_args()
TEST = args.test or bool(os.environ.get("NOTEBOOKTESTER_RUNNING"))

SEEDS = (0, 1) if TEST else (0, 1, 2)
MAX_EPOCHS = 2 if TEST else 80
FRACTIONS = [0.25, 1.0] if TEST else [0.05, 0.1, 0.25, 0.5, 1.0]
OUT = Path(__file__).parent / "results" / ("test" if TEST else "") / "eurosat_data.json"


def main() -> None:
    start = time.perf_counter()
    d = load("eurosat", "dinov2-small")
    dim, k = d["train"][0].shape[1], int(d["train"][1].max()) + 1
    base = repeat(lambda: NeuralNet(dim, k), d, SEEDS, epochs=MAX_EPOCHS)
    epochs = choose_epochs(base["valid"])
    print(f"EuroSAT: {len(d['train'][0])} train rows, epochs chosen on valid: {epochs}")
    rows = data_axis(d, base, epochs, FRACTIONS, SEEDS)
    for r in rows:
        mean = {
            arm: sum(r[f"test_{arm}"]) / len(SEEDS)
            for arm in ("equal_epochs", "equal_steps")
        }
        print(
            f"  {r['rows']:5d} rows  test accuracy {mean['equal_epochs']:.4f} at equal epochs, "
            f"{mean['equal_steps']:.4f} at equal steps"
        )
    print(f"{time.perf_counter() - start:.0f} s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "dataset": "eurosat",
                "encoder": "dinov2-small",
                "seeds": list(SEEDS),
                "max_epochs": MAX_EPOCHS,
                "epochs": epochs,
                "classes": k,
                "train_rows": len(d["train"][0]),
                "valid_curve": base["valid"].tolist(),
                "data_axis": rows,
                "minutes": round((time.perf_counter() - start) / 60, 1),
                "threads": torch.get_num_threads(),
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
