from pathlib import Path

import torch
from loguru import logger
from scipy.spatial import KDTree
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.data import data_tools
from src.settings import VAESettings
from src.visualization.visualize import plot_grid

import matplotlib.pyplot as plt

logger.add("/tmp/autoencoder.log")
logger.add("vae.log")

if __name__ == "__main__":
    presets = VAESettings()
    embedfile = "embeds.pt"

    img, embeds = torch.load(embedfile)
    logger.info(f"Loaded {embedfile} with shape {embeds.shape}")
    kdtree = KDTree(embeds)

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = data_tools.VAEstreamer(test_data, batchsize=10).stream()

    logger.info(f"loading pretrained model {presets.modelname}")
    model, modelname = torch.load(presets.modelname)
    x, y = test_data[1]

    other = model.encoder(x)

    dd, ii = kdtree.query(other.detach().numpy(), k=9)

    closest = img[ii]
    logger.info(f"closest items for label {y}")
    imgpath = presets.imgpath / Path(f"closest-label-{y}.png")
    plot_grid(closest, imgpath, title=f"Label {y}")
