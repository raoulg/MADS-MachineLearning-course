from datetime import datetime
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device


# There are more models in mltrainer.imagemodels for inspiration.
# You can import them, or create your own like here.
class CNN(nn.Module):
    def __init__(self, filters, units1, units2, input_size=(32, 1, 28, 28)):
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.units1 = units1
        self.units2 = units2

        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, 10),
        )

    def _conv_test(self, input_size=(32, 1, 28, 28)):
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits


def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


def objective(params):
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created {modeldir}")
    batchsize = 64
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=3,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.MLFLOW],
    )
    # Start a new MLflow run for tracking the experiment
    device = get_device()
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "raoul")
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate the CNN model with the given hyperparameters
        model = CNN(**params)
        model.to(device)
        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main():
    setup_mlflow("mlflow_database")

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
    main()
