import torch
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
import torch.optim as optim
import sys


logger.add("/tmp/explainer.log")
logger.add("explainer.log")

if __name__ == "__main__":
    from src.settings import ImageExplainerSettings
    from src.models.imagemodels import NeuralNetworkExplainer
    from src.models import metrics, train_model
    from src.settings import TrainerSettings


    logger.info("start image_explainer.py")
    presets = ImageExplainerSettings()

    logger.info(f"Importing {sys.argv[1]}")

    if sys.argv[1] == 'mnist':
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

    train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

    X, y = next(iter(train_dataloader))

    logger.info(
        f"Length trainset: {len(dataset_train)}, length testset: {len(dataset_test)}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Using {device} device")

    model = NeuralNetworkExplainer().to(device)
    logger.info(f"{model}")

    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()

    log_dir = "../../../models/explainer"

    logger.info(f"starting training for {presets.epochs} epochs")

    settings = TrainerSettings(
        epochs=presets.epochs,
        metrics=[accuracy],
        logdir=log_dir,
        train_steps=50,
        valid_steps=50,
        tunewrite=["tensorboard"],
        optimizer_kwargs = {"lr" : 1e-2}
    )

    trainer = train_model.Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_dataloader,
        validdataloader=test_dataloader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau
    )
    trainer.loop()

    torch.save(model, presets.modelname)
    logger.success("finished making model.py")
