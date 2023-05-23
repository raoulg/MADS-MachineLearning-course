from pathlib import Path

import tensorboard as tb
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.data import data_tools
from src.settings import VAESettings

cwd = Path(__file__).parent
root = (cwd / "../..").resolve()
logdir = root / Path("models/embeddings/")
writer = SummaryWriter(log_dir=logdir)

logger.add("/tmp/autoencoder.log")
logger.add("vae.log")

if __name__ == "__main__":
    logger.info("starting vae_embeddings.py")

    presets = VAESettings()

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = data_tools.VAEstreamer(
        test_data, batchsize=presets.samplesize
    ).stream()

    logger.info(f"loading pretrained model {presets.modelname}")
    model, modelname = torch.load(presets.modelname)
    X, _ = next(teststreamer)

    img = model(X)

    embs = model.encoder(X)
    logger.info(f"Embeddings shape {embs.shape}")

    embedfile = "embeds.pt"
    torch.save((X, embs.detach().numpy()), embedfile)
    logger.info(f"Saved embeddings and images to {embedfile}")

    # tensorflow has channnel first (dim 1), not channel (dim 3) last
    label_img = torch.moveaxis(X, 3, 1)
    writer.add_embedding(embs, label_img=label_img)
    logger.info(f"added embeddings to {logdir}")

    writer.close()
    logger.success("vae_embeddings.py")
