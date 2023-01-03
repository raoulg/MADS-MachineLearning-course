from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import tensorflow as tf
from loguru import logger
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logdir = Path("../../models/embeddings/").resolve()
writer = SummaryWriter(log_dir=logdir)


if __name__ == "__main__":
    logger.info("starting vae_embeddings.py")

    from src.settings import VAESettings
    from src.models.vae import select_n_random

    presets = VAESettings()

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    logger.info(f"Loaded {len(test_data)} items")
    images, labels = select_n_random(test_data.data, test_data.targets)
    logger.info(f"selected {len(images)} images")

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                         metadata=labels,
                         label_img=images.unsqueeze(1))
    logger.info(f"added embeddings to {logdir}")

    writer.close()
    logger.success("Embeddings")
