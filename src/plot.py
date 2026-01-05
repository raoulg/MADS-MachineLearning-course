from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    metrics_path = Path("results/metrics.csv")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)

    # Zorg dat lr netjes sorteert (als float) en fc_units als int
    df["lr"] = df["lr"].astype(float)
    df["fc_units"] = df["fc_units"].astype(int)

    # Maak pivot table voor heatmap: rijen=fc_units, kolommen=lr
    pivot = df.pivot_table(
        index="fc_units",
        columns="lr",
        values="best_val_acc",
        aggfunc="max",
    ).sort_index()

    # Plot
    plt.figure()
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="best_val_acc")

    # As labels
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), [f"{x:.4g}" for x in pivot.columns], rotation=45)

    plt.xlabel("learning rate (lr)")
    plt.ylabel("fc_units")
    plt.title("Hyperparameter interaction: lr × fc_units (best_val_acc)")

    out_path = out_dir / "heatmap_best_val_acc.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved heatmap to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
