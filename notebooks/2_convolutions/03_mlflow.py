"""The 03_mlflow notebook, as a script.

Same experiment, run from the terminal instead of a notebook:

    uv run python 03_mlflow.py

A script is the right shape once a search takes long enough that you do not want to babysit a
kernel. It is also where the dataloader settings differ from the notebooks (see `DataLoader`
below) and where the multiprocessing gotcha bites -- both are commented at the point they matter.
"""

from datetime import datetime
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings
from mltrainer.imagemodels import CNNConfig, CNNblocks
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from vectormesh.components.metrics import Accuracy
from vectormesh.data.vectorizers import detect_device

BATCHSIZE = 64
EPOCHS = 3
NUM_CLASSES = 10
MLRUNS = Path(__file__).resolve().parents[2] / "mlruns.db"


def get_fashion_loaders(batchsize: int) -> tuple[DataLoader, DataLoader]:
    """Fashion MNIST from the hub, as two dataloaders.

    `num_workers=2, persistent_workers=True` is the setting for scripts, not for notebooks. The
    measured gain is a fraction of a millisecond per batch -- invisible next to a training step --
    but a script runs long enough for it to be free, and `persistent_workers` matters more than
    the worker count, because without it the pool is rebuilt every epoch.
    """
    # A huggingface `Dataset` implements __getitem__/__len__, so a torch DataLoader is perfectly
    # happy with it -- but it does not inherit from torch's generic `Dataset[T]`, so a type
    # checker cannot see that. The cast says "I checked"; it changes nothing at runtime.
    fashion = cast(
        dict[str, TorchDataset], load_dataset("zalando-datasets/fashion_mnist")
    )

    def collate(items):
        X = [
            torch.from_numpy(
                np.asarray(i["image"], dtype=np.float32) / 255.0
            ).unsqueeze(0)
            for i in items
        ]
        y = torch.nn.functional.one_hot(
            torch.tensor([i["label"] for i in items]), NUM_CLASSES
        ).float()
        return torch.stack(X), y

    train = DataLoader(
        fashion["train"],
        batch_size=batchsize,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        persistent_workers=True,
    )
    valid = DataLoader(
        fashion["test"],
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        persistent_workers=True,
    )
    return train, valid


def setup_mlflow(experiment_path: str) -> None:
    # one store at the repo root, so runs from every lesson are comparable
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS}")
    mlflow.set_experiment(experiment_path)


def objective(params):
    modeldir = Path("models").resolve()
    modeldir.mkdir(parents=True, exist_ok=True)

    trainloader, validloader = get_fashion_loaders(BATCHSIZE)
    device = detect_device()

    settings = TrainerSettings(
        epochs=EPOCHS,
        metrics=[Accuracy()],
        logdir=Path("modellog"),
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.MLFLOW],
    )

    with mlflow.start_run():
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "raoul")
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{BATCHSIZE}")

        model = CNNblocks(
            CNNConfig(
                matrixshape=(28, 28),
                batchsize=BATCHSIZE,
                input_channels=1,
                maxpool=2,
                num_classes=NUM_CLASSES,
                **params,
            )
        )
        model.to(device)

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=optim.Adam,  # type: ignore
            traindataloader=trainloader,
            validdataloader=validloader,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main():
    setup_mlflow("l2_representations")

    search_space = {
        "hidden": scope.int(hp.quniform("hidden", 16, 64, 8)),
        "kernel_size": scope.int(hp.quniform("kernel_size", 3, 5, 2)),
        "num_layers": scope.int(hp.quniform("num_layers", 2, 4, 1)),
    }

    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=3, trials=Trials()
    )

    logger.info(f"Best result: {best_result}")


if __name__ == "__main__":
    # This guard is load-bearing, not decoration. With num_workers > 0 the DataLoader spawns
    # worker processes, and on macOS and Windows a spawned worker re-imports this module. Without
    # the guard it would re-run main() on import, and the workers would crash on startup.
    main()
