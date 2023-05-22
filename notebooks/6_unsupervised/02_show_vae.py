from pathlib import Path

import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor

logger.add("vae.log")
logger.add("/tmp/autoencoder.log")


if __name__ == "__main__":
    logger.info("Starting show_vae.py")
    from src.data import data_tools
    from src.models.vae import build_latent_grid, sample_range
    from src.settings import VAESettings
    from src.visualization.visualize import plot_grid

    presets = VAESettings()

    logger.info("loading data")
    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = data_tools.VAEstreamer(test_data, batchsize=32).stream()

    logger.info(f"loading pretrained model {presets.modelname}")
    model, modelname = torch.load(presets.modelname)

    X, Y = next(teststreamer)

    img = model(X)
    if not presets.imgpath.exists():
        presets.imgpath.mkdir(parents=True)

    imgpath = presets.imgpath / Path("vae-output-grid.png")
    plot_grid(img.detach().numpy(), filepath=imgpath)

    if presets.latent == 2:

        minimum, maximum = sample_range(model.encoder, teststreamer)
        logger.info(f"Found range min:{minimum} and max:{maximum}")
        latent_grid = build_latent_grid(model.decoder, minimum, maximum, k=20)
        latentpath = presets.imgpath / Path("latentspace.png")
        plot_grid(latent_grid, filepath=latentpath, k=20)

    else:
        logger.info("To visualize the latent space, set VAESettings.latent=2")
    logger.success("Finished show_vae.py")
