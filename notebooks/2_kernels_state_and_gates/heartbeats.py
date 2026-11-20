"""The heartbeats of lesson 2's notebooks: loading, training, scoring, and the spread.

The data is MIT-BIH, split by patient (`pttrn-io/heartbeats-mitbih-raw`). The notebooks
compare designs on a validation split carved from train by record, so that it holds
patients the model never trained on, as test does; test is left for the class problem.

Three classes: N, S and V. The notebooks read the macro recall, the mean of the three
recalls, so a missed S counts as much as a missed N.

A beat is `(187, 1)`: 187 steps at 125 Hz, one channel. A batch is `(B, S, D)`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from datasets import disable_progress_bars
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from vectormesh import VectorCache
from vectormesh.components import Conv1D, Pool1D

REPO = "pttrn-io/heartbeats-mitbih-raw"
REVISION = "b03962404ac3081ec1e3679171ea3a108d5e3601"
CLASSES = ["N", "S", "V"]
# six patients of train with N, S and V among their beats: 12.377 beats, 382 of them S
VAL_RECORDS = (118, 124, 201, 207, 215, 223)
STEPS = 187


@dataclass(frozen=True)
class Split:
    """One split as tensors: beats `(n, 187, 1)`, labels `(n,)`, records `(n,)`."""

    beats: Tensor
    labels: Tensor
    records: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        found = np.bincount(self.labels.numpy(), minlength=len(CLASSES))
        return dict(zip(CLASSES, found.tolist(), strict=True))

    def loader(self, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            TensorDataset(self.beats, self.labels),
            batch_size=batch_size,
            shuffle=shuffle,
        )


def _split(cache: VectorCache, keep: np.ndarray | None = None) -> Split:
    columns = cache.dataset.with_format("numpy")[:]
    beats, labels, records = columns["beat"], columns["label"], columns["record"]
    if keep is not None:
        beats, labels, records = beats[keep], labels[keep], records[keep]
    return Split(torch.from_numpy(beats), torch.from_numpy(labels).long(), records)


def load(max_rows: int | None = None) -> tuple[Split, Split]:
    """The fit and validation splits: train without and with `VAL_RECORDS`.

    `max_rows` keeps that many fit rows, drawn at random with a fixed seed, so a run
    under notebooktester stays cheap; validation is always whole.
    """
    disable_progress_bars()
    train = VectorCache.from_hub(REPO, split="train", revision=REVISION)
    records = np.asarray(train.dataset.with_format(None)["record"])
    val_rows = np.isin(records, VAL_RECORDS)
    fit_idx = np.flatnonzero(~val_rows)
    if max_rows is not None and max_rows < len(fit_idx):
        fit_idx = np.sort(np.random.default_rng(0).choice(fit_idx, max_rows, False))
    return _split(train, fit_idx), _split(train, np.flatnonzero(val_rows))


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, the macro recall and the recall of every class with beats in the split,
    from one confusion matrix over the whole split. A class without beats in the split is
    left out of the macro recall."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    present = cm.sum(axis=1) > 0
    recall = np.divide(
        np.diag(cm), cm.sum(axis=1), out=np.zeros(len(CLASSES)), where=present
    )
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_recall": float(recall[present].mean()),
        "recall": {c: float(r) for c, r, p in zip(CLASSES, recall, present) if p},
    }


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, split: Split, batch_size: int = 512) -> np.ndarray:
    model.eval()
    on = next(model.parameters()).device
    out = [
        model(x.to(on)).argmax(-1).cpu() for x, _ in split.loader(batch_size=batch_size)
    ]
    return torch.cat(out).numpy()


def train(
    model: nn.Module,
    fit: Split,
    val: Split,
    epochs: int,
    lr: float = 1e-3,
    loss_fn: nn.Module | None = None,
    batch_size: int = 128,
) -> tuple[dict, float]:
    """Train `model` on `fit` with Adam and cross-entropy, and score it on `val`.

    Returns the scores of the last epoch and the seconds spent training.
    """
    on = device()
    model = model.to(on)
    loss_fn = (loss_fn or nn.CrossEntropyLoss()).to(on)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in fit.loader(batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(on)), y.to(on))
            loss.backward()
            optimizer.step()
    seconds = time.perf_counter() - start
    return scores(val.labels.numpy(), predict(model, val)), seconds


def spread(
    make: Callable[[], nn.Module],
    fit: Split,
    val: Split,
    epochs: int,
    repeats: int,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """The macro recall of `repeats` runs, each from other random weights, and the mean
    seconds of one run."""
    found, seconds = [], []
    for _ in range(repeats):
        result, spent = train(make(), fit, val, epochs, **kwargs)
        found.append(result["macro_recall"])
        seconds.append(spent)
    return np.asarray(found), float(np.mean(seconds))


def line(name: str, values: np.ndarray, seconds: float, versus=None) -> str:
    """One result as `mean ± std`, and its distance to `versus` in sigmas of the larger
    spread of the two."""
    text = f"{name:34s} macro recall {values.mean():.3f} ± {values.std():.3f}"
    text += f"   {seconds:5.1f}s"
    if versus is not None:
        sd = max(values.std(), versus.std(), 1e-3)
        text += f"   {(values.mean() - versus.mean()) / sd:+.1f} sigma"
    return text


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field(layers: list[nn.Module]) -> int:
    """How many input positions one output position of `layers`, applied in order,
    depends on. Only the kernels and pools move the field; every other layer leaves it."""
    field, jump = 1, 1
    for layer in layers:
        if isinstance(layer, (Conv1D, Pool1D)):
            size = layer.kernel_size
            stride = layer.stride if isinstance(layer, Conv1D) else layer.kernel_size
            dilation = layer.dilation if isinstance(layer, Conv1D) else 1
            field += (size - 1) * dilation * jump
            jump *= stride
    return field


def reach(
    features: nn.Module, centre: int, samples: int = 32, seed: int = 0
) -> np.ndarray:
    """How strongly each input step can move one output position of `features`.

    The output position is the one whose field is centred nearest input step `centre`.
    The measure is the absolute gradient of that position with respect to each of the
    187 input steps, averaged over `samples` random inputs: one input alone shows only
    the paths a max pool happened to keep. A step outside the receptive field gets zero.
    """
    features = features.cpu().eval()
    x = torch.randn(
        samples, STEPS, 1, generator=torch.Generator().manual_seed(seed)
    ).requires_grad_(True)
    out = features(x)  # (samples, S', D')
    steps = np.arange(STEPS)
    best, best_gap = np.zeros(STEPS), np.inf
    for position in range(out.shape[1]):
        (grad,) = torch.autograd.grad(out[:, position].sum(), x, retain_graph=True)
        found = grad[..., 0].abs().mean(0).numpy()
        gap = abs((found * steps).sum() / max(found.sum(), 1e-12) - centre)
        if gap < best_gap:
            best, best_gap = found, gap
    return best
