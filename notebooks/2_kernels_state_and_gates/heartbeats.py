"""The heartbeats of lesson 2's notebooks: loading, training, scoring, and the spread.

The data is MIT-BIH, split by patient (`pttrn-io/heartbeats-mitbih-raw`). The notebooks
compare designs on a validation split carved from the dataset's train split by record, so that it holds
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
    """The train and validation splits: the dataset's train split without and with
    `VAL_RECORDS`.

    `max_rows` keeps that many train rows, drawn at random with a fixed seed, so a run
    under notebooktester stays cheap; validation is always whole.
    """
    disable_progress_bars()
    hub = VectorCache.from_hub(REPO, split="train", revision=REVISION)
    records = np.asarray(hub.dataset.with_format(None)["record"])
    val_rows = np.isin(records, VAL_RECORDS)
    train_idx = np.flatnonzero(~val_rows)
    if max_rows is not None and max_rows < len(train_idx):
        train_idx = np.sort(np.random.default_rng(0).choice(train_idx, max_rows, False))
    return _split(hub, train_idx), _split(hub, np.flatnonzero(val_rows))


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


@dataclass
class Run:
    """One training run: its scores on `valid`, its seconds, and a curve per epoch.

    `score` is the mean validation macro recall over the last `LAST` epochs: the score
    of one epoch alone swings by a few hundredths from one epoch to the next.
    """

    scores: dict
    seconds: float
    train_loss: list[float]
    val_loss: list[float]
    val_recall: list[float]

    @property
    def score(self) -> float:
        return float(np.mean(self.val_recall[-LAST:]))


LAST = 3


@torch.no_grad()
def _val_loss(model: nn.Module, split: Split, loss_fn: nn.Module) -> float:
    model.eval()
    on = next(model.parameters()).device
    total, n = 0.0, 0
    for x, y in split.loader(batch_size=512):
        total += loss_fn(model(x.to(on)), y.to(on)).item() * len(y)
        n += len(y)
    return total / n


def train(
    model: nn.Module,
    train: Split,
    valid: Split,
    epochs: int,
    lr: float = 1e-3,
    loss_fn: nn.Module | None = None,
    batch_size: int = 128,
    seed: int = 0,
) -> Run:
    """Train `model` on `train` with Adam and cross-entropy, and score it on `valid` after
    every epoch. `seed` fixes the order of the batches; the weights were drawn when the
    model was made."""
    on = device()
    model = model.to(on)
    loss_fn = (loss_fn or nn.CrossEntropyLoss()).to(on)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    train_loss, val_loss, val_recall = [], [], []
    seconds, scored = 0.0, {}
    for _ in range(epochs):
        start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train.loader(batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(on)), y.to(on))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        seconds += time.perf_counter() - start
        train_loss.append(total / n)
        val_loss.append(_val_loss(model, valid, loss_fn))
        scored = scores(valid.labels.numpy(), predict(model, valid))
        val_recall.append(scored["macro_recall"])
    return Run(scored, seconds, train_loss, val_loss, val_recall)


_train = train  # `spread` names its own split argument `train`


@dataclass
class Spread:
    """Repeated runs of one design, each from other random weights."""

    runs: list[Run]

    @property
    def values(self) -> np.ndarray:
        """The score of every run."""
        return np.asarray([r.score for r in self.runs])

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        """The spread between runs (`ddof=1`); 0 for a single run."""
        return float(self.values.std(ddof=1)) if len(self.runs) > 1 else 0.0

    @property
    def seconds(self) -> float:
        """The mean seconds of training of one run."""
        return float(np.mean([r.seconds for r in self.runs]))

    def recall(self) -> dict[str, float]:
        """The recall per class of the last epoch, averaged over the runs."""
        return {
            c: float(np.mean([r.scores["recall"][c] for r in self.runs]))
            for c in self.runs[0].scores["recall"]
        }


def spread(
    make: Callable[[], nn.Module],
    train: Split,
    valid: Split,
    epochs: int,
    repeats: int,
    **kwargs,
) -> Spread:
    """`repeats` runs of the model `make` builds; run `i` draws its weights and its
    batches with seed `i`."""
    runs = []
    for i in range(repeats):
        torch.manual_seed(i)
        runs.append(_train(make(), train, valid, epochs, seed=i, **kwargs))
    return Spread(runs)


def points(a: Spread, b: Spread) -> float:
    """The mean of `a` minus the mean of `b`, in points of macro recall (hundredths)."""
    return float(100 * (a.mean - b.mean))


def line(name: str, result: Spread, versus: Spread | None = None) -> str:
    """One result as `mean ± std` of the runs, the seconds of one run, and, against
    `versus`, the difference of the means in points."""
    text = f"{name:36s} macro recall {result.mean:.3f} ± {result.std:.3f}"
    text += f"   {result.seconds:5.1f}s"
    if versus is not None:
        text += f"   {points(result, versus):+.1f} pt"
    return text


def curves(results: dict[str, Spread], title: str, floor: float | None = None):
    """Train loss and validation macro recall per epoch: the mean over the runs, with
    the range of the runs shaded. `floor` draws a dashed line on the recall panel."""
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, result) in enumerate(results.items()):
        colour = f"C{i}"
        for ax, key in ((left, "train_loss"), (right, "val_recall")):
            rows = np.asarray(
                [getattr(r, key) for r in result.runs], dtype=float
            ).reshape(len(result.runs), -1)
            epochs = np.arange(1, rows.shape[1] + 1)
            ax.plot(epochs, rows.mean(0), color=colour, label=name)
            ax.fill_between(epochs, rows.min(0), rows.max(0), color=colour, alpha=0.15)
    if floor is not None:
        right.axhline(floor, color="0.5", ls="--", lw=1)
    left.set_title("train loss: has it stopped falling?")
    left.set_xlabel("epoch")
    left.set_ylabel("cross-entropy on train")
    right.set_title("macro recall on unseen patients")
    right.set_xlabel("epoch")
    right.set_ylabel("macro recall (valid)")
    left.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


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


def stack(
    depth: int, width: int = 32, kernel_size: int = 5, pool: bool = True
) -> list[nn.Module]:
    """`depth` blocks of a kernel and a GELU, each followed by a max pool of 2 when
    `pool`: notebook 2.1.4. The padding keeps the length through every kernel."""
    layers, channels = [], 1
    for _ in range(depth):
        layers += [
            Conv1D(channels, width, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        ]
        if pool:
            layers.append(Pool1D(2))
        channels = width
    return layers


class Kernels(nn.Module):
    """Notebook 2.1.5's design: a stack of kernels, flattened into a linear head,
    `(B, 187, 1) -> (B, S', width) -> (B, S' * width) -> (B, 3)`."""

    def __init__(self, depth: int, width: int = 32, kernel_size: int = 5):
        super().__init__()
        self.features = nn.Sequential(*stack(depth, width, kernel_size))
        with torch.no_grad():
            flat = self.features(torch.zeros(1, STEPS, 1)).flatten(1).shape[1]
        self.head = nn.Linear(flat, len(CLASSES))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x).flatten(1))
