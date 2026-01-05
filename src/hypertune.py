from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import itertools

import pandas as pd

from src.train import TrainConfig, run_training


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Baseline config (alles wat niet in de grid zit)
    base_cfg = TrainConfig(
        epochs=5,
        batch_size=64,
        weight_decay=1e-4,   # constant houden voor deze exercise
        base_channels=16,
        dropout=0.0,
        log_root=Path("runs/grid"),
        seed=42,
    )

    # Grid over twee hyperparameters (interactie!)
    lr_grid = [1e-2, 3e-3, 1e-3]
    fc_units_grid = [32, 64, 128]

    all_results: list[dict] = []

    for lr, fc_units in itertools.product(lr_grid, fc_units_grid):
        cfg = replace(base_cfg, lr=lr, fc_units=fc_units)
        run_name = f"lr={lr}_fc={fc_units}"
        print(f"\n=== RUN {run_name} ===")

        res = run_training(cfg, run_name=run_name)
        all_results.append(res)

        print(f"best_val_acc={res['best_val_acc']:.3f} | best_val_loss={res['best_val_loss']:.3f}")

    df = pd.DataFrame(all_results).sort_values("best_val_acc", ascending=False)
    out_csv = results_dir / "metrics.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nWrote results to: {out_csv.resolve()}")
    print("Top 5 configs:")
    print(df[["run_name", "best_val_acc", "lr", "fc_units"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
