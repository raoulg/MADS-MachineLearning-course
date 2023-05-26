from typing import List, Tuple

import torch

# import jax.numpy as jnp
from numpy import ndarray

# from trax.layers.combinators import Serial as Traxmodel
# from trax.shapes import signature

Array = ndarray


def init_weights(
    m: int, n: int, scale: float = 1e-1, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return scale * torch.normal(0.0, scale, (m, n)), scale * torch.normal(
        0.0, scale, (n,)
    )


def torch_network(
    sizes: List[int],
    scale: float = 1e-1,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [init_weights(m, n, scale, seed) for m, n in zip(sizes[:-1], sizes[1:])]
