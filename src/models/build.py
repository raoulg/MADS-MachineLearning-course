from typing import List, Tuple

import jax.numpy as jnp
import torch
from jax import random


def random_layer_params(
    m: int, n: int, key: jnp.ndarray, scale: float = 1e-2
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))


def init_network(
    sizes: List[int], key: jnp.ndarray
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def init_weights(
    m: int, n: int, scale: float = 1e-1
) -> Tuple[torch.Tensor, torch.Tensor]:
    return scale * torch.normal(0.0, scale, (m, n)), scale * torch.normal(
        0.0, scale, (n,)
    )


def torch_network(
    sizes: List[int], scale: float = 1e-1
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [init_weights(m, n, scale) for m, n in zip(sizes[:-1], sizes[1:])]
