# import torch
import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor
from mads_datasets.base import BaseDatastreamer
from typing import Iterator
from pydantic import BaseModel
from pathlib import Path

logger.add("/tmp/autoencoder.log")
logger.add("vae.log")


class VAEstreamer(BaseDatastreamer):
    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            # we throw away the Y
            X_, _ = zip(*batch)  # noqa N806
            X = torch.stack(X_)  # noqa N806
            # change the channel to channel-last
            X = torch.moveaxis(X, 1, 3)  # noqa N806
            # and yield X, X
            yield X, X


class VAESettings(BaseModel):
    data_dir: Path = Path("data")
    h1: int = 250
    h2: int = 100
    insize: int = 784
    latent: int = 10
    batchsize: int = 32
    epochs: int = 100
    modelname: Path = Path("vaemodel.pt")
    imgpath: Path = Path("img")
    samplesize: int = 512


if __name__ == "__main__":
    logger.info("starting autoencode.py")
    from mltrainer import Trainer, vae, TrainerSettings, ReportTypes

    presets = VAESettings()

    logger.info("loading MNIST datasets")
    training_data = datasets.MNIST(
        root=presets.data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    logger.info(
        f"Length trainset: {len(training_data)}, length testset: {len(test_data)}"
    )

    logger.info("creating datastreamers")
    trainstreamer = VAEstreamer(training_data, batchsize=presets.batchsize).stream()
    teststreamer = VAEstreamer(test_data, batchsize=32).stream()

    X1, X2 = next(trainstreamer)

    config = presets.dict()
    logger.info(f"the shape before: {X1.shape}")

    encoder = vae.Encoder(config)
    decoder = vae.Decoder(config)

    latent = encoder(X1)
    logger.info(f"the latent shape : {latent.shape}")

    x = decoder(latent)
    logger.info(f"the shape after: {x.shape}")

    lossfn = vae.RecostructionLoss()
    loss = lossfn(x, X2)
    logger.info(f"Untrained loss: {loss}")

    logger.info(f"starting training for {presets.epochs} epochs")
    autoencoder = vae.AutoEncoder(config)

    settings = TrainerSettings(
        epochs=presets.epochs,
        metrics=[lossfn],
        logdir="vaemodels",
        train_steps=200,
        valid_steps=200,
        reporttypes=[ReportTypes.TENSORBOARD],
        scheduler_kwargs={"factor": 0.5, "patience": 10},
    )

    trainer = Trainer(
        model=autoencoder,
        settings=settings,
        loss_fn=lossfn,
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer,
        validdataloader=teststreamer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    trainer.loop()

    torch.save(autoencoder, presets.modelname)

    logger.success("finished autoencode.py")
