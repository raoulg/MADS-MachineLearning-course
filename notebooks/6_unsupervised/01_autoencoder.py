# import torch
import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor

logger.add("/tmp/autoencoder.log")
logger.add("vae.log")

if __name__ == "__main__":
    logger.info("starting autoencode.py")
    from src.data import data_tools
    from src.models import train_model, vae
    from src.settings import VAESettings, TrainerSettings, ReportTypes

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
    trainstreamer = data_tools.VAEstreamer(
        training_data, batchsize=presets.batchsize
    ).stream()
    teststreamer = data_tools.VAEstreamer(test_data, batchsize=32).stream()

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

    trainer = train_model.Trainer(
        model=autoencoder,
        settings=settings,
        loss_fn=lossfn,
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer,
        validdataloader=teststreamer,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau
    )
    trainer.loop()

    torch.save(autoencoder, presets.modelname)

    logger.success("finished autoencode.py")
