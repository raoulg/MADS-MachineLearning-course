from pathlib import Path

import torch
from loguru import logger
from scipy.spatial import KDTree
from torchvision import datasets
from torchvision.transforms import ToTensor

from settings import VAESettings
from show_vae import plot_grid

logger.add("logs/vae.log")


def main():
    presets = VAESettings()
    embedfile = "models/embeds.pt"

    img, embeds = torch.load(embedfile, weights_only=False)
    logger.info(f"Loaded {embedfile} with shape {embeds.shape}")
    kdtree = KDTree(embeds)

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    modelpath = presets.modeldir / presets.modelname
    logger.info(f"loading pretrained model {modelpath}")
    model = torch.load(modelpath, weights_only=False)

    x, y = test_data[1]

    other = model.encoder(x)

    dd, ii = kdtree.query(other.detach().numpy(), k=9)

    closest = img[ii[0]]  # type: ignore
    logger.info(f"closest items for label {y}")
    imgpath = presets.imgpath / Path(f"closest-label-{y}.png")
    plot_grid(closest, imgpath, title=f"Label {y}")


if __name__ == "__main__":
    main()
