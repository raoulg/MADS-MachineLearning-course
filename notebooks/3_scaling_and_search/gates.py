"""The models, the training and the routing of `01_gates_highways_moe`.

One module, imported by the notebook and by the two scripts that train its families
(`gates_family.py`, `gates_balance.py`), so all three build and train a model the same way.

Every model is the same frame with a different middle:

    (B, 192) -> Linear(192, hidden) -> GELU -> middle -> Linear(hidden, 48) -> (B, 48)

The table of middles is in section 1.1 of the notebook.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# a torch module that vectormesh imports warns on macOS that it cannot redirect output;
# nothing here uses it
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)

import torch
from australian_electricity import AusElec, load_ausalia
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from vectormesh.components import (
    Gate,
    Highway,
    MoE,
    NeuralNet,
    SparseMoE,
    load_balancing_loss,
)
from vectormesh.components.metrics import MAE
from vectormesh.training import Trainer, TrainerSettings
from vectormesh.training.trainer import BatchTensor
from windowing import window_index

# the Trainer logs every epoch at INFO; keep warnings only
logger.remove()
logger.add(sys.stderr, level="WARNING")

HERE = Path(__file__).resolve().parent
DAY = 48  # half-hourly steps per day
W, H = 4 * DAY, DAY  # four days in, one day out
STRIDE = 24  # train windows start every 24 half-hours; val and test every 48
HIDDEN = 64
LR = 1e-3
BATCH = 64

MODEL_CONFIGS: list[tuple[str, str, dict, str | None]] = [
    # (label, mechanism, make_model kwargs, control)
    ("plain", "plain", {}, None),
    ("deeper", "deeper", {}, None),
    ("deeper (hidden 86)", "deeper", {"hidden": 86}, None),
    ("deeper (hidden 96)", "deeper", {"hidden": 96}, None),
    ("gate", "gate", {}, "deeper"),
    ("highway", "highway", {}, "deeper (hidden 86)"),
    ("moe", "moe", {"n_experts": 4}, "deeper (hidden 96)"),
    ("sparse_moe (N=4)", "sparse_moe", {"n_experts": 4, "k": 1}, "deeper (hidden 96)"),
    ("sparse_moe (N=64)", "sparse_moe", {"n_experts": 64, "k": 1}, None),
]
CONTROL = {label: control for label, _, _, control in MODEL_CONFIGS}


@dataclass(frozen=True)
class Windows:
    """The series cut into (input, target) windows per split, scaled with the train span."""

    series: AusElec
    mu: float
    sigma: float
    xtr: torch.Tensor
    ytr: torch.Tensor
    xva: torch.Tensor
    yva: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor


def load_windows() -> Windows:
    """Train windows every STRIDE half-hours, val and test windows one per day."""
    series = load_ausalia()
    train = series.values[series.splits["train"]]
    mu, sigma = float(train.mean()), float(train.std())

    def cut(split: str, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = (series.values[series.splits[split]] - mu) / sigma
        win = values[window_index(len(values), W + H, stride)]
        return win[:, :W], win[:, W:]

    return Windows(
        series,
        mu,
        sigma,
        *cut("train", STRIDE),
        *cut("val", 2 * STRIDE),
        *cut("test", 2 * STRIDE),
    )


def make_model(
    mechanism: str,
    hidden: int = HIDDEN,
    n_experts: int = 4,
    k: int = 1,
    normalize: bool = False,
) -> nn.Module:
    """Linear(192, hidden) -> GELU -> middle -> Linear(hidden, 48), the middle named by
    `mechanism`."""
    if mechanism == "plain":
        return NeuralNet(hidden_size=[W, hidden], out_size=H)
    if mechanism == "deeper":
        return NeuralNet(hidden_size=[W, hidden, hidden], out_size=H)
    inp, act, out = nn.Linear(W, hidden), nn.GELU(), nn.Linear(hidden, H)
    if mechanism == "gate":
        mid = Gate(hidden_size=hidden)
    elif mechanism == "highway":
        # NeuralNet(hidden, hidden) is Linear(hidden, hidden) -> GELU -> Linear(hidden, hidden)
        mid = Highway(transform=NeuralNet(hidden, hidden), hidden_size=hidden)
    elif mechanism == "moe":
        mid = MoE(
            experts=[nn.Linear(hidden, hidden) for _ in range(n_experts)],
            hidden_size=hidden,
            out_size=hidden,
        )
    elif mechanism == "sparse_moe":
        mid = SparseMoE(
            experts=[nn.Linear(hidden, hidden) for _ in range(n_experts)],
            hidden_size=hidden,
            out_size=hidden,
            k=k,
            normalize=normalize,
        )
    else:
        raise ValueError(mechanism)
    return nn.Sequential(inp, act, mid, out)


def n_params(model: nn.Module) -> int:
    """Weights and biases: every number the optimizer changes."""
    return sum(p.numel() for p in model.parameters())


def mae(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float((y - yhat).abs().mean())


@dataclass
class Run:
    """One training run: the val and test MAE of the kept epoch, that epoch (counted from
    1), the val curves of every epoch, and the model holding the kept epoch's weights."""

    val_mae: float
    test_mae: float
    best_epoch: int
    val_mae_curve: list[float]
    val_loss_curve: list[float]
    model: nn.Module


def run(
    data: Windows,
    mechanism: str,
    seed: int,
    epochs: int,
    balance: float = 0.0,
    **model_kwargs,
) -> Run:
    """Train one model with Adam (lr LR, batches of BATCH) for the whole `epochs`, record
    val MAE and val loss (MSE) after every epoch, and keep the weights of the epoch with
    the lowest val loss. `balance` > 0 adds that multiple of the load-balancing loss of
    every SparseMoE to the training loss; the val loss never has it."""
    torch.manual_seed(seed)
    model = make_model(mechanism, **model_kwargs)
    trainloader = DataLoader(
        TensorDataset(data.xtr, data.ytr), batch_size=BATCH, shuffle=True
    )
    validloader = DataLoader(
        TensorDataset(data.xva, data.yva), batch_size=len(data.xva)
    )
    val_mae_curve: list[float] = []
    val_loss_curve: list[float] = []

    def track(epoch, train_loss, test_loss, metric_dict):
        # the Trainer passes the loss on its validation loader as `test_loss`
        val_mae_curve.append(float(metric_dict["MAE"]))
        val_loss_curve.append(float(test_loss))

    # patience = the whole budget: the Trainer never stops early, and keeps (in memory)
    # the weights of the epoch with the lowest val loss
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[MAE()],
        logdir=HERE / "modellogs",
        train_steps=len(trainloader),
        valid_steps=len(validloader),
        optimizer_kwargs={"lr": LR},
        earlystop_kwargs={"patience": epochs, "save": False, "verbose": False},
    )

    def balanced_step(
        model: nn.Module, x: BatchTensor, y: torch.Tensor
    ) -> torch.Tensor:
        loss = nn.functional.mse_loss(model(x), y)
        if model.training:
            loss = loss + balance * load_balancing_loss(model)
        return loss

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        traindataloader=trainloader,
        validdataloader=validloader,
        step=balanced_step if balance else None,
        reporters=[track],
        progress=False,
    )
    result = trainer.loop()
    trainer.model.eval()
    with torch.no_grad():
        test_mae = mae(data.yte, trainer.model(data.xte))
    return Run(
        val_mae=float(result.metric_dict["MAE"]),
        test_mae=test_mae,
        best_epoch=result.epoch + 1,  # TrainResult counts epochs from 0
        val_mae_curve=val_mae_curve,
        val_loss_curve=val_loss_curve,
        model=trainer.model,
    )


def routing(model: nn.Module, x: torch.Tensor) -> tuple[list[int], float]:
    """For a Linear -> GELU -> SparseMoE -> Linear model: how many windows of `x` the
    router sends to each expert (at k = 2 a window counts for both its experts), and the
    mean weight of the first chosen expert (rescaled over the k chosen when `normalize`
    is on)."""
    if not isinstance(model, nn.Sequential) or not isinstance(model[2], SparseMoE):
        raise TypeError(
            "routing reads a Linear -> GELU -> SparseMoE -> Linear Sequential"
        )
    inp, act, moe = model[0], model[1], model[2]
    assert isinstance(moe, SparseMoE)
    with torch.no_grad():
        g = torch.softmax(moe.router(act(inp(x))), dim=-1)
        top, chosen = g.topk(moe.k, dim=-1)
        if moe.normalize:
            top = top / top.sum(dim=-1, keepdim=True)
    counts = torch.bincount(chosen.reshape(-1), minlength=moe.num_experts)
    return counts.tolist(), float(top[:, 0].mean())


def early_stop_epochs(val_loss: list[float], patience: int) -> tuple[int, int]:
    """Where early stopping with `patience` would keep and stop on a recorded val-loss
    curve: the epoch with the lowest loss so far, and the epoch `patience` epochs without
    improvement later (or the last epoch). Epochs count from 1."""
    best, best_epoch, waited = float("inf"), 0, 0
    for epoch, loss in enumerate(val_loss, start=1):
        if loss < best:
            best, best_epoch, waited = loss, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                return best_epoch, epoch
    return best_epoch, len(val_loss)
