import math
from pathlib import Path
from typing import Callable, List, Tuple, Union

import gin
import numpy as np
import tensorflow as tf  # noqa: F401

# needed to make summarywriter load without error
import torch
from loguru import logger
from numpy import Inf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import data_tools
from src.typehinting import GenericModel


def write_gin(dir: Path) -> None:
    path = dir / "saved_config.gin"
    with open(path, "w") as file:
        file.write(gin.operative_config_str())


@gin.configurable
def trainloop(
    epochs: int,
    model: GenericModel,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    loss_fn: Callable,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    log_dir: Union[Path, str],
    eval_steps: int,
) -> GenericModel:
    optimizer_: torch.optim.Optimizer = optimizer(
        model.parameters(), lr=learning_rate
    )  # type: ignore
    log_dir = Path(log_dir)
    data_tools.clean_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer_.zero_grad()
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer_.step()
            train_loss += loss.data.item()
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        for _ in range(eval_steps):
            input, target = next(iter(test_dataloader))
            output = model(input)
            loss = loss_fn(output, target)
            test_loss += loss.data.item()
            test_accuracy += (output.argmax(dim=1) == target).sum()
        datasize = eval_steps * test_dataloader.batch_size
        test_loss /= datasize
        test_accuracy /= datasize
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Loss/accuracy", test_accuracy, epoch)
        logger.info(
            f"Epoch {epoch} train {train_loss:.4f} | test {test_loss:.4f} acc {test_accuracy:.4f}"
        )
    write_gin(log_dir)
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
) -> Tuple[List[float], List[float], List[Tuple]]:
    num_epochs = len(data_loader) - 1
    update_step = (final_value / init_value) ** (1 / num_epochs)
    lr = init_value
    optimizer.param_groups[0]["lr"] = init_value
    best_lr = 0.0
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
