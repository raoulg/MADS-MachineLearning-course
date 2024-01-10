import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from loguru import logger
import torch.optim as optim
import torch.nn as nn
import sys
from pydantic import BaseModel
from pathlib import Path
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes


logger.add("/tmp/explainer.log")
logger.add("explainer.log")


class NeuralNetworkExplainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        out = self.fc_layers(x)
        return out


class ImageExplainerSettings(BaseModel):
    data_dir: Path = Path("data")
    epochs: int = 10
    modelname: Path = Path("trained_model.pt")
    imgpath: Path = Path("img")
    num_classes: int = 10


if __name__ == "__main__":
    logger.info("start image_explainer.py")
    presets = ImageExplainerSettings()

    logger.info(f"Importing {sys.argv[1]}")

    if sys.argv[1] == "mnist":
        dataset_train = datasets.MNIST(
            root=presets.data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        dataset_test = datasets.MNIST(
            root=presets.data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
        )
    else:
        dataset_train = datasets.FashionMNIST(
            root=presets.data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        dataset_test = datasets.FashionMNIST(
            root=presets.data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

    X, y = next(iter(train_dataloader))

    logger.info(
        f"Length trainset: {len(dataset_train)}, length testset: {len(dataset_test)}"
    )

    model = NeuralNetworkExplainer()

    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()

    log_dir = "models/explainer"

    logger.info(f"starting training for {presets.epochs} epochs")

    settings = TrainerSettings(
        epochs=presets.epochs,
        metrics=[accuracy],
        logdir=log_dir,
        train_steps=50,
        valid_steps=50,
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 1e-2},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_dataloader,
        validdataloader=test_dataloader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    )
    trainer.loop()

    torch.save(model, presets.modelname)
    logger.success("finished making model.py")
