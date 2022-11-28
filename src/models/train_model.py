import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import gin
import numpy as np
import tensorflow as tf  # noqa: F401

# needed to make summarywriter load without error
import torch
from loguru import logger
from numpy import Inf
from ray import tune
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from src.data import data_tools
from src.models.metrics import Metric
from src.typehinting import GenericModel


def write_gin(dir: Path, txt) -> None:
    path = dir / "saved_config.gin"
    with open(path, "w") as file:
        file.write(txt)


def trainbatches(
    model: GenericModel,
    traindatastreamer: Iterator,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    train_steps: int,
) -> float:
    model.train()
    train_loss: float = 0.0
    for _ in tqdm(range(train_steps)):
        x, y = next(iter(traindatastreamer))
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().numpy()
    train_loss /= train_steps
    return train_loss


def evalbatches(
    model: GenericModel,
    testdatastreamer: Iterator,
    loss_fn: Callable,
    metrics: List[Metric],
    eval_steps: int,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    test_loss: float = 0.0
    metric_dict: Dict[str, float] = {}
    for _ in range(eval_steps):
        x, y = next(iter(testdatastreamer))
        yhat = model(x)
        test_loss += loss_fn(yhat, y).detach().numpy()
        for m in metrics:
            metric_dict[str(m)] = (
                metric_dict.get(str(m), 0.0) + m(y, yhat).detach().numpy()
            )

    test_loss /= eval_steps
    for key in metric_dict:
        metric_dict[str(key)] = metric_dict[str(key)] / eval_steps
    return metric_dict, test_loss


@gin.configurable
def trainloop(
    epochs: int,
    model: GenericModel,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    loss_fn: Callable,
    metrics: List[Metric],
    train_dataloader: Iterator,
    test_dataloader: Iterator,
    log_dir: Path,
    train_steps: int,
    eval_steps: int,
    patience: int = 10,
    factor: float = 0.9,
    tunewriter: bool = False,
) -> GenericModel:
    """

    Args:
        epochs (int) : Amount of runs through the dataset
        model: A generic model with a .train() and .eval() method
        optimizer : an uninitialized optimizer class. Eg optimizer=torch.optim.Adam
        learning_rate (float) : floating point start value for the optimizer
        loss_fn : A loss function
        metrics (List[Metric]) : A list of callable metrics.
            Assumed to have a __repr__ method implemented, see src.models.metrics
            for Metric details
        train_dataloader, test_dataloader (Iterator): data iterators
        log_dir (Path) : where to log stuff when not using the tunewriter
        train_steps, eval_steps (int) : amount of times the Iterators are called for a
            new batch of data.
        patience (int): used for the ReduceLROnPlatues scheduler. How many epochs to
            wait before dropping the learning rate.
        factor (float) : fraction to drop the learning rate with, after patience epochs
            without improvement in the loss.
        tunewriter (bool) : when running experiments manually, this should
            be False (default). If false, a subdir is created
            with a timestamp, and a SummaryWriter is invoked to write in
            that subdir for Tensorboard use.
            If True, the logging is left to the ray.tune.report


    Returns:
        _type_: _description_
    """

    optimizer_: torch.optim.Optimizer = optimizer(
        model.parameters(), lr=learning_rate
    )  # type: ignore

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_,
        factor=factor,
        patience=patience,
    )

    if not tunewriter:
        log_dir = data_tools.dir_add_timestamp(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        write_gin(log_dir, gin.config_str())

        images, _ = next(iter(train_dataloader))
        if len(images.shape) == 4:
            grid = make_grid(images)
            writer.add_image("images", grid)
        writer.add_graph(model, images)

    for epoch in tqdm(range(epochs)):
        train_loss = trainbatches(
            model, train_dataloader, loss_fn, optimizer_, train_steps
        )

        metric_dict, test_loss = evalbatches(
            model, test_dataloader, loss_fn, metrics, eval_steps
        )

        scheduler.step(test_loss)

        if tunewriter:
            tune.report(
                iterations=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                **metric_dict,
            )
        else:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            for m in metric_dict:
                writer.add_scalar(f"metric/{m}", metric_dict[m], epoch)
            lr = [group["lr"] for group in optimizer_.param_groups][0]
            writer.add_scalar("learning_rate", lr, epoch)
            metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
            )

    return model


def count_parameters(model: GenericModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_lr(
    model: GenericModel,
    loss_fn: Callable,
    optimizer: Optimizer,
    data_loader: DataLoader,
    smooth_window: int = 10,
    init_value: float = 1e-8,
    final_value: float = 10.0,
) -> Tuple[List[float], List[float]]:
    num_epochs = len(data_loader) - 1
    update_step = (final_value / init_value) ** (1 / num_epochs)
    lr = init_value
    optimizer.param_groups[0]["lr"] = init_value
    best_loss = Inf
    batch_num = 0
    losses = []
    smooth_losses: List[float] = []
    log_lrs: List[float] = []
    for x, y in tqdm(data_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)

        if loss < best_loss:
            best_loss = loss

        if loss > 4 * best_loss:
            return log_lrs[10:-5], smooth_losses[10:-5]

        losses.append(loss.item())
        batch_num += 1
        start = max(0, batch_num - smooth_window)
        smooth = np.mean(losses[start:batch_num])
        smooth_losses.append(smooth)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], smooth_losses[10:-5]
