"""Random against TPE on the Food-101 heads, with every trial's validation curve stored.

The heads sit on frozen dinov2-large vectors: 150 training images per class, 101 classes,
1,515 of them held out for validation. Every trial trains for the full number of epochs and
records its validation accuracy after each epoch, so successive halving can be replayed on
the stored curves afterwards without a second search.

    uv run python notebooks/3_scaling_and_search/search_budget.py
    uv run python notebooks/3_scaling_and_search/search_budget.py --trials 4 --epochs 2

The result is `results/search_budget.json`: per sampler, per trial, its parameters, its
curve, its run time and its parameter count.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch import nn
from vectormesh import VectorCache

HERE = Path(__file__).parent
PER_CLASS = 150
VALID = 1515
CLASSES = 101
WIDTHS = [0, 64, 128, 256, 512, 1024]
BATCHES = [64, 128, 256, 512]


def load() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardised training and validation vectors with their labels."""
    data = VectorCache.from_hub("pttrn-io/food101-dinov2-large", split="train")
    data = data.dataset.with_format("numpy")
    X, y = np.asarray(data["embed"], dtype=np.float32), np.asarray(data["label"])
    if X.ndim == 3:
        X = X.mean(1)
    pick = np.random.default_rng(0)
    rows = np.concatenate(
        [pick.permutation(np.flatnonzero(y == c))[:PER_CLASS] for c in range(CLASSES)]
    )
    X, y = torch.tensor(X[rows]), torch.tensor(y[rows]).long()
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    valid, train = order[:VALID], order[VALID:]
    mean, std = X[train].mean(0), X[train].std(0) + 1e-6
    X = (X - mean) / std
    return X[train], y[train], X[valid], y[valid]


def head(dim: int, width: int, depth: int, dropout: float) -> nn.Module:
    """A linear head (width 0) or an MLP of `depth` hidden layers of `width` units."""
    if width == 0:
        return nn.Linear(dim, CLASSES)
    layers: list[nn.Module] = []
    d_in = dim
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.GELU(), nn.Dropout(dropout)]
        d_in = width
    return nn.Sequential(*layers, nn.Linear(d_in, CLASSES))


def suggest(trial: optuna.Trial) -> dict:
    """The search space: the learning rate matters most, several axes barely at all."""
    width = trial.suggest_categorical("width", WIDTHS)
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "width": width,
        "depth": trial.suggest_int("depth", 1, 2) if width else 0,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "batch": trial.suggest_categorical("batch", BATCHES),
        "smoothing": trial.suggest_float("smoothing", 0.0, 0.2),
    }


def train(data, p: dict, epochs: int) -> tuple[list[float], int]:
    """Validation accuracy after every epoch, and the head's parameter count."""
    Xtr, ytr, Xva, yva = data
    torch.manual_seed(0)
    model = head(Xtr.shape[1], p["width"], p["depth"], p["dropout"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    curve = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ytr))
        for i in range(0, len(order), p["batch"]):
            idx = order[i : i + p["batch"]]
            loss = F.cross_entropy(
                model(Xtr[idx]), ytr[idx], label_smoothing=p["smoothing"]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            curve.append(float((model(Xva).argmax(1) == yva).float().mean()))
    return curve, sum(w.numel() for w in model.parameters())


def search(
    data, sampler: optuna.samplers.BaseSampler, trials: int, epochs: int
) -> list[dict]:
    """One study; the objective is the validation accuracy after the last epoch."""
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        p = suggest(trial)
        start = time.perf_counter()
        curve, n_params = train(data, p, epochs)
        seconds = time.perf_counter() - start
        rows.append(
            {
                "number": trial.number,
                "params": p,
                "curve": curve,
                "seconds": seconds,
                "n_params": n_params,
            }
        )
        print(f"{trial.number:3d} {curve[-1]:.3f} {seconds:5.1f}s {p}", flush=True)
        return curve[-1]

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "search_budget.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load()
    result = {"epochs": args.epochs, "train": len(data[1]), "valid": len(data[3])}
    for name, sampler in [
        ("random", optuna.samplers.RandomSampler(seed=0)),
        ("tpe", optuna.samplers.TPESampler(seed=0)),
    ]:
        print(name, flush=True)
        result[name] = search(data, sampler, args.trials, args.epochs)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
