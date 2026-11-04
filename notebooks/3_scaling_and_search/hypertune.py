"""A hypertuning script you can copy and point at your own problem.

The notebooks run searches inline so you can read them. Real searches belong in a script:
they run for a while, they run on a machine you are not sitting in front of, and you want
the result on disk rather than in a kernel that might die.

Run it with:

    uv run python hypertune.py

Three things here are worth copying rather than the specifics of the model:

1. **The trainable is self-contained.** Ray runs it in a separate process, so anything it
   needs it loads itself. Closing over a dataframe defined in your notebook is the single
   most common way a ray search fails.
2. **The cache is loaded once per worker and reused**, guarded by a file lock so parallel
   workers do not race each other to download the same thing.
3. **Search and reporting are separated.** `HyperOptSearch` proposes, `ASHAScheduler`
   prunes, and the result is written to disk as a dataframe you analyse afterwards.
"""

from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch import nn
from vectormesh import VectorCache
from vectormesh.components import NeuralNet

CACHE = "pttrn-io/dtd-dinov2-small"
N_CLASSES = 47
MAX_EPOCHS = 20
N_TRIALS = 30
BATCH = 128
RESULTS = Path("ray_results").resolve()


def load_split() -> tuple[torch.Tensor, ...]:
    """Cache -> (train, val) tensors. Called inside each trial, hence the lock.

    Shuffle before slicing: the DTD caches are stored grouped by label, so an unshuffled
    prefix is a class-truncated subset rather than a smaller sample of the same task.
    """
    with FileLock(Path.home() / ".cache" / "dtd_cache.lock"):
        cache = VectorCache.from_hub(CACHE)
    d = cache.dataset.shuffle(seed=0).with_format("numpy")
    X = torch.tensor(np.asarray(d["embed"], dtype="float32"))
    y = torch.tensor(np.asarray(d["label"]))
    n = int(0.75 * len(X))
    return X[:n], y[:n], X[n:], y[n:]


def trainable(config: dict) -> None:
    """One trial. Reports after every epoch so ASHA has something to prune on."""
    Xtr, ytr, Xva, yva = load_split()
    torch.manual_seed(config.get("seed", 0))

    widths = [Xtr.shape[1]] + [config["width"]] * config["depth"]
    model = NeuralNet(widths, N_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - 1, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = float((model(Xva).argmax(1) == yva).float().mean())
        # One report per epoch: ASHA compares trials at matched epochs and stops the
        # laggards. Reporting only at the end would leave it nothing to act on.
        tune.report({"accuracy": acc})


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    search_space = {
        "width": tune.choice([64, 128, 256, 512, 1024]),
        "depth": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-2),
    }

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=HyperOptSearch(),
            scheduler=ASHAScheduler(grace_period=3, reduction_factor=3),
            num_samples=N_TRIALS,
        ),
        run_config=tune.RunConfig(storage_path=str(RESULTS), name="dtd_head_search"),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="accuracy", mode="max")
    metrics = best.metrics or {}
    logger.info(f"best config: {best.config}")
    logger.info(
        f"best validation accuracy: {metrics.get('accuracy', float('nan')):.4f}"
    )

    df = results.get_dataframe()
    out = RESULTS / "dtd_head_search.csv"
    df.to_csv(out, index=False)
    logger.info(f"{len(df)} trials written to {out}")
    logger.info(
        "That number is a validation score, and it is the maximum of many noisy trials. "
        "Retrain the winner and score it once on a test split before reporting it."
    )


if __name__ == "__main__":
    main()
