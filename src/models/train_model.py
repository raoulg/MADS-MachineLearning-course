import copy
import math
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from numpy import Inf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gin
from pathlib import Path

from src.typehinting import GenericModel
from src.data import data_tools


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
    log_dir: Union[Path, str]
) -> GenericModel:
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    log_dir = Path(log_dir)
    data_tools.clean_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.eval()
        test_loss = 0.0
        for batch in test_dataloader:
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            test_loss += loss.data.item()
        test_loss /= len(test_dataloader.dataset)
        writer.add_scalar("Loss/test", test_loss, epoch)
        logger.info(f"Epoch {epoch} train {train_loss:.4f} | test {test_loss:.4f}")
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
) -> Tuple[List[float], List[float]]:
    num_epochs = len(data_loader) - 1
    update_step = (final_value / init_value) ** (1 / num_epochs)
    lr = init_value
    best_loss = Inf
    best_diff = Inf
    batch_num = 0
    losses = []
    smooth_losses = []
    momentum = 0.0
    momentum_list = []
    log_lrs = []
    diff_coords = []
    for x, y in tqdm(data_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)

        if loss < best_loss:
            best_loss = loss

        if loss > 4 * best_loss:
            logger.info(f"Best diff {best_diff:.2f} at {best_lr:.2f}")
            return log_lrs[10:-5], smooth_losses[10:-5], diff_coords, momentum_list

        losses.append(loss.item())
        batch_num += 1
        start = max(0, batch_num - smooth_window)
        smooth = np.mean(losses[start:batch_num])
        smooth_losses.append(smooth)
        log_lrs.append(math.log10(lr))

        if batch_num > 2:
            diff = smooth_losses[-1] - smooth_losses[-2]
            momentum += 0.7 * diff
            momentum_list.append(momentum)
            if diff < best_diff:
                best_diff = diff
                best_lr = log_lrs[-1]
                diff_coords.append((best_lr, smooth))

        loss.backward()
        optimizer.step()

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    logger.info(f"Best diff {best_diff:.4f} at {best_lr:.4f}")
    return log_lrs[10:-5], smooth_losses[10:-5], diff_coords, momentum_list
