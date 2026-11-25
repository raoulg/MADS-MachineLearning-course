"""The tools of `02_scaling_curves` and its stored sweep: heads on Food-101's dinov2-large
vectors, trained on a growing share of the train images.

The data is `pttrn-io/food101-dinov2-large`: one 1024-dim vector per image from a frozen
dinov2-large, 101 classes. Its split "train" (750 images per class) gives the training
images and a validation set of `VALID_PER_CLASS` images per class; its split "test"
(250 per class, labelled by hand) is the held-out set every curve is read on. A training
set of `per_class` images takes the first `per_class` of every class in one fixed order, so
a smaller set is part of every larger one. The vectors are standardised with the mean and
std of the whole train split.

`train` runs Adam (learning rate 1e-3, batch 128) and evaluates twice per pass over the
training images. It stops when the validation loss has not improved for 1.5 times the
steps it took to reach its lowest point (and after at least `MIN_STEPS`), so every run
goes past its own minimum. The step of that minimum is chosen on validation; the held-out
loss and accuracy are read at that step.

Run as a script, it trains every head of `WIDTHS` on every size of `PER_CLASS`, one seed,
and stores the result in `results/food_scaling.json`:

    uv run python food_scaling.py            # the full sweep
    uv run python food_scaling.py --test     # a few seconds, into results/test/
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import datasets
import huggingface_hub.utils
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)
from vectormesh import VectorCache

datasets.disable_progress_bars()
huggingface_hub.utils.disable_progress_bars()
logger.remove()
logger.add(sys.stderr, level="WARNING")

REPO = "pttrn-io/food101-dinov2-large"
CLASSES = 101
VALID_PER_CLASS = 50
BATCH, LR = 128, 1e-3
MIN_STEPS, MAX_STEPS = 400, 20_000
# images per class of the data axis: 1,010 to 70,700 training images
PER_CLASS = [10, 30, 70, 200, 700]
# hidden units of the one-layer heads of the sweep; 0 is the linear head
WIDTHS = [0, 16, 32, 64, 128, 256]
RESULTS = Path(__file__).parent / "results"


@dataclass(frozen=True)
class Food:
    """Standardised vectors and labels: train (with its fixed order per class), the
    validation set carved from it, and the held-out set."""

    X: torch.Tensor
    y: torch.Tensor
    order: dict[int, np.ndarray]
    X_valid: torch.Tensor
    y_valid: torch.Tensor
    X_held: torch.Tensor
    y_held: torch.Tensor

    def subset(self, per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The first `per_class` training images of every class."""
        rows = np.concatenate(
            [
                self.order[c][VALID_PER_CLASS : VALID_PER_CLASS + per_class]
                for c in range(CLASSES)
            ]
        )
        return self.X[rows], self.y[rows]


def _arrays(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    table = VectorCache.from_hub(REPO, split=split).dataset.data
    # the vectors as one (n, 1024) array, straight from the Arrow column
    X = (
        table.column("embed")
        .combine_chunks()
        .flatten()
        .to_numpy()
        .reshape(len(table), -1)
    )
    y = table.column("label").to_numpy()
    return torch.from_numpy(X.copy()).float(), torch.from_numpy(y).long()


@cache
def load() -> Food:
    """Download (once) and load the cache, carve the validation set, standardise."""
    X, y = _arrays("train")
    X_held, y_held = _arrays("test")
    rng = np.random.default_rng(0)
    order = {c: rng.permutation(np.flatnonzero(y.numpy() == c)) for c in range(CLASSES)}
    valid = np.concatenate([order[c][:VALID_PER_CLASS] for c in range(CLASSES)])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, X_held = (X - mu) / sd, (X_held - mu) / sd
    return Food(X, y, order, X[valid], y[valid], X_held, y_held)


def head(width: int, dim: int = 1024) -> nn.Module:
    """A linear head (`width` 0) or one hidden layer of `width` units with GELU."""
    if width == 0:
        return nn.Linear(dim, CLASSES)
    return nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, CLASSES))


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(
    width: int,
    X: torch.Tensor,
    y: torch.Tensor,
    food: Food,
    seed: int = 0,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Train `head(width)`, its weights drawn from `seed`, on (X, y) past its lowest
    validation loss. Returns the curve (step, validation loss, test loss and test accuracy
    at every evaluation), the step with the lowest validation loss, and the test loss and
    accuracy at that step."""
    torch.manual_seed(seed)
    model = head(width)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    every = max(1, len(X) // BATCH // 2)
    curve = {"step": [], "valid": [], "held": [], "held_acc": []}
    best, best_step, step = float("inf"), 0, 0
    t = time.time()
    while step < max(MIN_STEPS, 2.5 * best_step) and step < max_steps:
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            j = perm[i : i + BATCH]
            loss = F.cross_entropy(model(X[j]), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % every == 0:
                with torch.no_grad():
                    v = F.cross_entropy(model(food.X_valid), food.y_valid).item()
                    out = model(food.X_held)
                curve["step"].append(step)
                curve["valid"].append(v)
                curve["held"].append(F.cross_entropy(out, food.y_held).item())
                curve["held_acc"].append(
                    (out.argmax(1) == food.y_held).float().mean().item()
                )
                if v < best:
                    best, best_step = v, step
    at = curve["step"].index(best_step)
    return {
        "n": len(X),
        "params": params(model),
        "best_step": best_step,
        "held": curve["held"][at],
        "held_acc": curve["held_acc"][at],
        "secs": time.time() - t,
        "curve": curve,
    }


def sweep(
    per_class: list[int], widths: list[int], max_steps: int = MAX_STEPS
) -> list[dict]:
    """Every head of `widths` on every size of `per_class`, one seed."""
    food = load()
    rows = []
    for pc in per_class:
        X, y = food.subset(pc)
        for w in widths:
            r = train(w, X, y, food, max_steps=max_steps)
            rows.append({"per_class": pc, "width": w, **r})
            print(
                f"{len(X):6d} images  width {w:3d}  {r['params']:>7,} params  "
                f"held-out loss {r['held']:.3f}  at step {r['best_step']}",
                flush=True,
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--test", action="store_true", help="a few seconds, into results/test/"
    )
    test = parser.parse_args().test
    out = RESULTS / "test" if test else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    rows = sweep([10, 30], [0, 8], max_steps=200) if test else sweep(PER_CLASS, WIDTHS)
    (out / "food_scaling.json").write_text(json.dumps(rows))
