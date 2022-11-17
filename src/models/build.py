from typing import List, Tuple, Union

# import jax.numpy as jnp
from numpy import ndarray
import torch
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


def summary(
    model, X: Array, init: int = 1, counter: int = 0  # noqa N803
) -> Array:
    output = X  # noqa N803
    input = None #signature(output)
    if init == 1:
        print(
            f'{"layer":<23} {"input":<19} {"dtype":^7}    {"output":<19} {"dtype":^7}'
        )
    for sub in model.sublayers:
        name = str(sub.name)
        if name == "":
            continue
        elif name == "Serial":
            output = summary(sub, output, init + 1, counter)
        else:
            output = sub.output_signature(input)
            print(
                f"({counter}) {str(sub.name):<19} {str(input.shape):<19}({str(input.dtype):^7}) | {str(output.shape):<19}({str(output.dtype):^7})"  # noqa E501
            )
        input = output
        counter += 1
    return output
