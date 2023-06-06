import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import gin
import mlflow
import numpy as np

# needed to make summarywriter load without error
import torch
from loguru import logger
from numpy import Inf
from ray import tune
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import data_tools
from src.models.metrics import Metric
from src.settings import ReportTypes, TrainerSettings
from src.typehinting import GenericModel


def write_gin(dir: Path, txt: str) -> None:
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
    for _ in tqdm(range(train_steps), colour="#1e4706"):
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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        settings: TrainerSettings,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        traindataloader: Iterator,
        validdataloader: Iterator,
        scheduler: Optional[Callable],

    ) -> None:
        self.model = model
        self.settings = settings
        self.log_dir = data_tools.dir_add_timestamp(settings.logdir)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader

        self.optimizer = optimizer(  # type: ignore
            model.parameters(), **settings.optimizer_kwargs
        )
        self.last_epoch = 0

        if scheduler:
            if settings.scheduler_kwargs is None:
                raise ValueError("Missing 'scheduler_kwargs' in TrainerSettings.")
            self.scheduler = scheduler(self.optimizer, **settings.scheduler_kwargs)

        if settings.earlystop_kwargs is not None:
            logger.info(
                "Found earlystop_kwargs in settings."
                "Set to None if you dont want earlystopping.")
            self.early_stopping = EarlyStopping(
                self.log_dir,
                **settings.earlystop_kwargs
            )
        else:
            self.early_stopping = None

        if ReportTypes.TENSORBOARD in self.settings.reporttypes:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        if ReportTypes.GIN in self.settings.reporttypes:
            write_gin(self.log_dir, gin.config_str())

    def loop(self) -> None:
        for epoch in tqdm(range(self.settings.epochs), colour="#1e4706"):
            train_loss = self.trainbatches()
            metric_dict, test_loss = self.evalbatches()
            self.report(epoch, train_loss, test_loss, metric_dict)

            if self.early_stopping:
                self.early_stopping(test_loss, self.model)  # type: ignore

            if self.early_stopping is not None and self.early_stopping.early_stop:
                logger.info("Interrupting loop due to early stopping patience.")
                self.last_epoch = epoch
                if self.early_stopping.save:
                    logger.info("retrieving best model.")
                    self.model = self.early_stopping.get_best()  # type: ignore
                else:
                    logger.info(
                        "early_stopping_save was false, using latest model."
                        "Set to true to retrieve best model.")
                break
        self.last_epoch = epoch

    def trainbatches(self) -> float:
        self.model.train()
        train_loss: float = 0.0
        train_steps = self.settings.train_steps
        for _ in tqdm(range(train_steps), colour="#1e4706"):
            x, y = next(iter(self.traindataloader))
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.detach().numpy()
        train_loss /= train_steps
        return train_loss

    def evalbatches(self) -> Tuple[Dict[str, float], float]:
        self.model.eval()
        valid_steps = self.settings.valid_steps
        test_loss: float = 0.0
        metric_dict: Dict[str, float] = {}
        for _ in range(valid_steps):
            x, y = next(iter(self.validdataloader))
            yhat = self.model(x)
            test_loss += self.loss_fn(yhat, y).detach().numpy()
            for m in self.settings.metrics:
                metric_dict[str(m)] = (
                    metric_dict.get(str(m), 0.0) + m(y, yhat).detach().numpy()
                )

        test_loss /= valid_steps
        for key in metric_dict:
            metric_dict[str(key)] = metric_dict[str(key)] / valid_steps
        return metric_dict, test_loss

    def report(self, epoch: int, train_loss: float, test_loss: float, metric_dict: Dict) -> None:
        epoch = epoch + self.last_epoch
        reporttypes = self.settings.reporttypes
        self.test_loss = test_loss

        if ReportTypes.RAY in reporttypes:
            tune.report(
                iterations=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                **metric_dict,
            )

        if ReportTypes.MLFLOW in reporttypes:
            mlflow.log_metric("Loss/train", train_loss, step=epoch)
            mlflow.log_metric("Loss/test", test_loss, step=epoch)
            for m in metric_dict:
                mlflow.log_metric(f"metric/{m}", metric_dict[m], step=epoch)
            lr = [group["lr"] for group in self.optimizer.param_groups][0]
            mlflow.log_metric("learning_rate", lr, step=epoch)

        if ReportTypes.TENSORBOARD in reporttypes:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            for m in metric_dict:
                self.writer.add_scalar(f"metric/{m}", metric_dict[m], epoch)
            lr = [group["lr"] for group in self.optimizer.param_groups][0]
            self.writer.add_scalar("learning_rate", lr, epoch)

        metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
        logger.info(
            f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
        )


class EarlyStopping:
    def __init__(self, log_dir: Path, patience: int = 7, verbose: bool = False, delta: float = 0.0,
                 save: bool = False) -> None:
        """
        Args:
            log_dir (Path): location to save checkpoint to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = Path(log_dir) / 'checkpoint.pt'
        self.save = save

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        # first epoch best_loss is still None
        if self.best_loss is None:
            self.best_loss = val_loss  # type: ignore
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif val_loss >= self.best_loss + self.delta:  # type: ignore
            # we minimize loss. If current loss did not improve
            # the previous best (with a delta) it is considered not to improve.
            self.counter += 1
            logger.info(
                f'best loss: {self.best_loss:.4f}, current loss {val_loss:.4f}. Counter {self.counter}/{self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # if not the first run, and val_loss is smaller, we improved.
            self.best_loss = val_loss
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(
                f'Validation loss ({self.best_loss:.4f} --> {val_loss:.4f}). Saving {self.path} ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss

    def get_best(self) -> torch.nn.Module:
        return torch.load(self.path)


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
