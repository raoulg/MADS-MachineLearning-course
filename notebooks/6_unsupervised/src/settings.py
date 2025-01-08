from pathlib import Path
from typing import Iterator

import torch
from mads_datasets.base import BaseDatastreamer
from pydantic import BaseModel


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
    modeldir: Path = Path("models")
    imgpath: Path = Path("img")
    samplesize: int = 512
