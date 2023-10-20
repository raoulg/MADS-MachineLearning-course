from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from settings import VAESettings, VAEstreamer

logger.add("logs/vae.log")
logger.add("/tmp/autoencoder.log")


def sample_range(encoder, stream, k: int = 10):
    minmax_list = []
    for _ in range(10):
        X, _ = next(stream)
        y = encoder(X).detach().numpy()
        minmax_list.append(y.min())
        minmax_list.append(y.max())
    minmax = np.array(minmax_list)
    return minmax.min(), minmax.max()


def build_latent_grid(decoder, minimum: int, maximum: int, k: int = 20):
    x = np.linspace(minimum, maximum, k)
    y = np.linspace(minimum, maximum, k)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    img = decoder(torch.tensor(grid, dtype=torch.float32))
    return img.detach().numpy()


def plot_grid(
    img: np.ndarray,
    filepath: Path,
    k: int = 3,
    figsize: tuple = (10, 10),
    title: str = "",
) -> None:
    fig, axs = plt.subplots(k, k, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axs = axs.ravel()
    for i in tqdm(range(k * k)):
        axs[i].imshow(img[i], cmap="gray")
        axs[i].axis("off")
    fig.savefig(filepath)
    logger.success(f"saved grid to {filepath}")


def main():
    logger.info("Starting show_vae.py")

    presets = VAESettings()

    logger.info("loading data")
    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = VAEstreamer(test_data, batchsize=32).stream()

    modelpath = presets.modeldir / presets.modelname

    logger.info(f"loading pretrained model {modelpath}")
    model = torch.load(modelpath)

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


if __name__ == "__main__":
    main()
