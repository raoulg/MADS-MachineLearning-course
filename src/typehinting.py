from typing import Callable, Protocol

import torch


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass
