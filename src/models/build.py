from typing import List, Tuple

import jax.numpy as jnp
import torch
from jax import random
from trax.shapes import signature


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

def summary(model, X, init=1, counter=0):
    output = X
    input = signature(output)
    if init == 1:
        print(f'{"layer":<23} {"input":<19} {"dtype":^7}    {"output":<19} {"dtype":^7}')
    for sub in model.sublayers:
        name = str(sub.name)
        if name == '':
            continue
        elif name == "Serial":
            output = summary(sub, output, init+1, counter)
        else:
            output = sub.output_signature(input)
            print(f'({counter}) {str(sub.name):<19} {str(input.shape):<19}({str(input.dtype):^7}) | {str(output.shape):<19}({str(output.dtype):^7})') 
        input = output
        counter += 1
    return output