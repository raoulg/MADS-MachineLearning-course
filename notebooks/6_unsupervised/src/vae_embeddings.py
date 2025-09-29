from pathlib import Path

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

from settings import VAESettings, VAEstreamer

logdir = Path("models/embeddings/")
writer = SummaryWriter(log_dir=logdir)

logger.add("/tmp/autoencoder.log")
logger.add("logs/vae.log")


def main():
    logger.info("starting vae_embeddings.py")

    presets = VAESettings()

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = VAEstreamer(test_data, batchsize=presets.samplesize).stream()  # type: ignore

    modelpath = presets.modeldir / presets.modelname
    logger.info(f"loading pretrained model {modelpath}")
    model = torch.load(modelpath, weights_only=False)
    X, _ = next(teststreamer)

    embs = model.encoder(X)
    logger.info(f"Embeddings shape {embs.shape}")

    embedfile = "models/embeds.pt"
    torch.save((X, embs.detach().numpy()), embedfile)
    logger.info(f"Saved embeddings and images to {embedfile}")

    # tensorflow has channnel first (dim 1), not channel (dim 3) last
    label_img = torch.moveaxis(X, 3, 1)
    writer.add_embedding(embs, label_img=label_img)
    logger.info(f"added embeddings to {logdir}")

    writer.close()
    logger.success("vae_embeddings.py")


if __name__ == "__main__":
    main()
