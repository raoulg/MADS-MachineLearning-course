"""A hyperparameter search as a script, rather than in a notebook.

    uv run python 03_mlflow.py

A script is the right shape once a search takes long enough that you do not want to babysit a
kernel, and it is where two things differ from the notebooks: the dataloader settings (see
`get_fashion_loaders`) and the multiprocessing entry-point guard (see the bottom of the file).
Both are commented at the point they matter.

This is deliberately *not* a copy of `03_mlflow.ipynb`. The notebook searches a head on cached
flower vectors, because that is where the cost argument lives; this searches the CNN from
`01_convolutions.ipynb` on raw fashion pixels, which is the slow side of that same argument and
the one worth running unattended. Logs to the same MLflow store, so the two end up in one
experiment and are directly comparable.

It imports that CNN from `cnn.py` rather than using `mltrainer.imagemodels.CNNblocks`, and the
reason is worth knowing before you search anything. `CNNblocks` derives its flatten size from the
pooling alone, so changing `kernel_size` silently invalidates the arithmetic and the run dies with
a shape error partway through the search. Our `CNN` measures the activation map with `_conv_test`
instead of predicting it, so every point in the search space builds. **A model that cannot be
reconfigured cannot be searched** -- which is a lesson about writing models, not about hyperopt.
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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from cnn import CNN  # the model from 01_convolutions.ipynb
from vectormesh.components.metrics import Accuracy
from vectormesh.data.vectorizers import detect_device

BATCHSIZE = 64
EPOCHS = 3
NUM_CLASSES = 10
MLRUNS = Path(__file__).resolve().parents[2] / "artefacts" / "mlflow" / "mlruns.db"


def collate(items):
    """Turn a list of rows into a batch: PIL -> (1, 28, 28) float, label -> one-hot.

    This lives at module level, and that is load-bearing. With `num_workers > 0` the DataLoader
    has to *pickle* its `collate_fn` to hand it to the worker processes, and a function defined
    inside another function cannot be pickled -- you get

        AttributeError: Can't get local object 'get_fashion_loaders.<locals>.collate'

    which is the same multiprocessing constraint as the `if __name__ == "__main__"` guard at the
    bottom of this file, in a second disguise. Notebooks never hit either, because they run with
    `num_workers=0`.
    """
    X = [
        torch.from_numpy(np.asarray(i["image"], dtype=np.float32) / 255.0).unsqueeze(0)
        for i in items
    ]
    y = torch.nn.functional.one_hot(
        torch.tensor([i["label"] for i in items]), NUM_CLASSES
    ).float()
    return torch.stack(X), y


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
    # One store for the whole course, so runs from every lesson are comparable -- and the same
    # one 03_mlflow.ipynb writes to, which is what makes the notebook's head search and this
    # script's CNN search show up side by side. sqlite creates the file but not its directory.
    MLRUNS.parent.mkdir(parents=True, exist_ok=True)
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

        model = CNN(
            input_size=(BATCHSIZE, 1, 28, 28),
            num_classes=NUM_CLASSES,
            **params,
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
        "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
        "units1": scope.int(hp.quniform("units1", 32, 128, 8)),
        "units2": scope.int(hp.quniform("units2", 32, 128, 8)),
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
