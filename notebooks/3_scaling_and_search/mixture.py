"""The tools of `01_mixture_of_experts`: load the four-domain mix, train a head on it, and
time a head per image.

A head is trained with AdamW (learning rate 1e-3, weight decay 0.05), label smoothing 0.1,
batch 128, on inputs standardised with the mean and std of the train rows. Every seed cuts
its own validation split, 10% of the train rows of every class. The parameters of the epoch
with the best validation accuracy are kept, and the test split is scored once, with those.
"""

import logging
import sys
import time

import datasets
import huggingface_hub.utils
import jaxtyping
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

# Quiet output: no download bars, no cache-loaded lines, no note about macOS redirects.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)
from vectormesh import VectorCache

datasets.disable_progress_bars()
huggingface_hub.utils.disable_progress_bars()
logger.remove()
logger.add(sys.stderr, level="WARNING")

DOMAINS = ["dtd", "eurosat", "fashion-mnist", "oxford-flowers"]
BATCH, LR, WEIGHT_DECAY, SMOOTHING = 128, 1e-3, 0.05, 0.1


def load_mix(split: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """The vectors, labels and domain of every row of one split of the four caches.

    Each domain's labels are shifted past the classes of the domains before it, so no two
    domains share a class number. A cache stored per patch is averaged over its patches.
    """
    X, y, domain, offset = [], [], [], 0
    for i, name in enumerate(DOMAINS):
        part = VectorCache.from_hub(f"pttrn-io/{name}-dinov2-small", split=split)
        part = part.dataset.with_format("numpy")
        e = np.asarray(part["embed"], dtype=np.float32)
        if e.ndim == 3:
            e = e.mean(1)
        labels = np.asarray(part["label"])
        X.append(e)
        y.append(labels + offset)
        domain.append(np.full(len(labels), i))
        offset += int(labels.max()) + 1
    return (
        torch.tensor(np.concatenate(X)),
        torch.tensor(np.concatenate(y)).long(),
        np.concatenate(domain),
    )


def split_valid(y: torch.Tensor, seed: int, share: float = 0.1):
    """Indices of train and validation rows: `share` of every class goes to validation."""
    g = np.random.default_rng(seed)
    train, valid = [], []
    for c in np.unique(y.numpy()):
        rows = np.flatnonzero(y.numpy() == c)
        g.shuffle(rows)
        n = max(1, round(share * len(rows)))
        valid += rows[:n].tolist()
        train += rows[n:].tolist()
    return np.array(train), np.array(valid)


def train(make, train_rows, X_test: torch.Tensor, seed: int, epochs: int):
    """Train make() on the train rows of `load_mix` and predict the test vectors once.

    Returns the trained model (in eval mode, on standardised inputs), the standardised test
    vectors, the test predictions and the validation accuracy after every epoch.
    """
    X_all, y_all = train_rows[0], train_rows[1]
    tr, va = split_valid(y_all, seed)
    mean, std = X_all[tr].mean(0), X_all[tr].std(0) + 1e-6
    X, y = (X_all[tr] - mean) / std, y_all[tr]
    X_valid, y_valid = (X_all[va] - mean) / std, y_all[va]
    X_test = (X_test - mean) / std

    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    valid_curve, best, kept = [], -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i : i + BATCH]
            loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=SMOOTHING)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(X_valid).argmax(1) == y_valid).float().mean())
        valid_curve.append(acc)
        if acc > best:
            best, kept = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(kept)
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(1)
    return model, X_test, pred, valid_curve


def time_per_image(model: nn.Module, batch: int, dim: int, calls: int) -> float:
    """Median microseconds per image of one forward pass over `batch` images, one thread.

    Ten untimed passes warm up first; then `calls` passes are timed. vectormesh checks the
    shapes of the tensors on every call while you develop; a deployed model runs without that
    check, so the timing turns it off and restores it afterwards.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checking = jaxtyping.config.jaxtyping_disable
    jaxtyping.config.update("jaxtyping_disable", True)
    x = torch.randn(batch, dim)
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        for _ in range(calls):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    torch.set_num_threads(threads)
    jaxtyping.config.update("jaxtyping_disable", checking)
    return float(np.median(times)) * 1e6 / batch
