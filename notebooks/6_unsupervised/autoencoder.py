from torchvision import datasets
# import torch
import sys
from torchvision.transforms import ToTensor
from pathlib import Path


if __name__ == "__main__":
    cwd = Path(__file__).parent
    cwd = (cwd / "../../").resolve()
    sys.path.insert(0, str(cwd))
    # from src.models import imagemodels
    from src.data import data_tools
    from src.settings import GeneralSettings
    from src.models import vae

    presets = GeneralSettings()

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

    trainstreamer = data_tools.VAEstreamer(
        training_data, batchsize=32).stream()
    teststreamer = data_tools.VAEstreamer(test_data, batchsize=32).stream()

    X1, X2 = next(trainstreamer)
    config = {
        "in": 28*28,
        "h1": 250,
        "h2": 100,
        "latent": 10,
    }

    encoder = vae.Encoder(config)
    latent = encoder(X1)


    print("end")
