"""The tools of `02_scaling_curves` and `eurosat_data.py`: load a cache, train a head, repeat
it over seeds, and measure the data axis.

A head is trained with Adam (learning rate 1e-3, batch 128) for a number of epochs or a
number of optimizer steps. `train` records the train and valid loss and the valid and test
accuracy after every epoch when `curve=True`, and only after the last step otherwise.
"""

import logging
import sys

import datasets
import huggingface_hub.utils
import numpy as np
import torch
from loguru import logger
from torch import nn

# Quiet output: no download bars, no cache-loaded lines, no note about macOS redirects.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)
from vectormesh import VectorCache
from vectormesh.components import NeuralNet

datasets.disable_progress_bars()
huggingface_hub.utils.disable_progress_bars()
logger.remove()
logger.add(sys.stderr, level="WARNING")

BATCH, LR = 128, 1e-3


def balanced_order(y: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """An order of the rows in which every prefix holds each class about equally often.

    The rows are ranked within their class at random and sorted on that rank, so rows[:n]
    is a stratified subset for every n, and a smaller subset is part of every larger one.
    """
    g = torch.Generator().manual_seed(seed)
    rank = torch.empty(len(y))
    for c in y.unique():
        idx = (y == c).nonzero().squeeze(1)
        rank[idx[torch.randperm(len(idx), generator=g)]] = torch.arange(
            len(idx), dtype=torch.float
        )
    return torch.argsort(rank + 0.5 * torch.rand(len(y), generator=g))


def load(dataset: str, encoder: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The cache's train split cut 80/20 into train and valid, and its own test split.

    The train rows come in the balanced order of `balanced_order`.
    """
    hub = f"pttrn-io/{dataset}-{encoder}"
    train = VectorCache.from_hub(hub, split="train").dataset
    test = VectorCache.from_hub(hub, split="test").dataset
    split = train.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    out = {}
    for name, part in [
        ("train", split["train"]),
        ("valid", split["test"]),
        ("test", test),
    ]:
        part = part.with_format("numpy")
        X = torch.tensor(np.asarray(part["embed"], dtype="float32"))
        y = torch.tensor(np.asarray(part["label"]))
        out[name] = (X, y)
    order = balanced_order(out["train"][1])
    out["train"] = (out["train"][0][order], out["train"][1][order])
    return out


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(
    make,
    d: dict,
    epochs: int | None = None,
    steps: int | None = None,
    rows: int | None = None,
    curve: bool = True,
) -> dict[str, list]:
    """Train make() on the first `rows` train rows for `epochs` epochs or `steps` steps.

    One step is one batch: a forward pass, a backward pass, one update of the weights. The
    record holds the steps so far, the train and valid loss and the valid and test
    accuracy: after every epoch with `curve=True`, after the last step otherwise.
    """
    X, y = d["train"]
    if rows is not None:
        X, y = X[:rows], y[:rows]
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    record = {k: [] for k in ("steps", "train_loss", "valid_loss", "valid", "test")}
    done, epoch = 0, 0
    while (epochs is None or epoch < epochs) and (steps is None or done < steps):
        model.train()
        total, seen = 0.0, 0
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            done += 1
            total, seen = total + loss.item() * len(idx), seen + len(idx)
            if steps is not None and done == steps:
                break
        epoch += 1
        last = (epochs is not None and epoch == epochs) or done == steps
        if curve or last:
            Xv, yv = d["valid"]
            model.eval()
            with torch.no_grad():
                valid_loss = float(lossf(model(Xv), yv))
            for k, v in [
                ("steps", done),
                ("train_loss", total / seen),
                ("valid_loss", valid_loss),
                ("valid", accuracy(model, Xv, yv)),
                ("test", accuracy(model, *d["test"])),
            ]:
                record[k].append(v)
    return record


def repeat(make, d: dict, seeds, **budget) -> dict:
    """train() once per seed; every record becomes an array of (seeds, records)."""
    runs = []
    for s in seeds:
        torch.manual_seed(s)
        runs.append(train(make, d, **budget))
    out: dict = {k: np.array([r[k] for r in runs]) for k in runs[0]}
    out["params"] = n_params(make())
    return out


def choose_epochs(valid: np.ndarray) -> int:
    """The epoch with the highest valid accuracy, averaged over the seeds; numbered from 1."""
    return int(valid.mean(0).argmax()) + 1


def mean_std(values, scale: float = 1.0) -> str:
    """`mean ± std` of `values` times `scale`, rounded to the first digit of the std."""
    v = np.asarray(values, dtype=float) * scale
    std = float(v.std())
    digits = max(0, -int(np.floor(np.log10(std)))) if std > 0 else 4
    return f"{v.mean():.{digits}f} ± {std:.{digits}f}"


def data_axis(d: dict, base: dict, epochs: int, fractions, seeds) -> list[dict]:
    """The baseline head on fractions of the train rows, at equal epochs and equal steps.

    Equal epochs: every subset gets `epochs` passes. Equal steps: every subset gets the
    optimizer steps of the full set at `epochs` epochs. At the full set both are the
    baseline runs `base`, read at `epochs`. Per fraction: the rows, the steps of each arm
    and the test accuracy of each seed in each arm.
    """
    n_full = len(d["train"][0])
    dim, k = d["train"][0].shape[1], int(d["train"][1].max()) + 1
    steps_full = int(base["steps"][0, epochs - 1])
    out = []
    for frac in fractions:
        n = int(frac * n_full)
        if n == n_full:
            full = base["test"][:, epochs - 1].tolist()
            arms = {"epochs": (steps_full, full), "steps": (steps_full, full)}
        else:
            arms = {}
            for arm, budget in [
                ("epochs", {"epochs": epochs}),
                ("steps", {"steps": steps_full}),
            ]:
                r = repeat(
                    lambda: NeuralNet(dim, k), d, seeds, rows=n, curve=False, **budget
                )
                arms[arm] = (int(r["steps"][0, -1]), r["test"][:, -1].tolist())
        out.append(
            {
                "rows": n,
                "steps_equal_epochs": arms["epochs"][0],
                "steps_equal_steps": arms["steps"][0],
                "test_equal_epochs": arms["epochs"][1],
                "test_equal_steps": arms["steps"][1],
            }
        )
    return out
