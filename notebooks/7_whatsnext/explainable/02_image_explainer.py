import torch
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
import torch.optim as optim


logger.add("/tmp/explainer.log")
logger.add("explainer.log")

if __name__ == "__main__":
    from src.settings import ImageExplainerSettings
    from src.models.imagemodels import NeuralNetworkExplainer
    from src.models import metrics, train_model

    
    logger.info("starting autoencode.py")
    presets = ImageExplainerSettings()

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

    log_dir="../../models/test"

    logger.info(f"starting training for {presets.epochs} epochs")

    model = train_model.trainloop(
        epochs=presets.epochs,
        model=model,
        optimizer=optim.Adam,
        learning_rate=1e-2,
        loss_fn=loss_fn,
        metrics=[accuracy],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_dir=log_dir,
        train_steps=50,
        eval_steps=50,
    )

    torch.save(model, presets.modelname)
    logger.success("finished making model.py")
