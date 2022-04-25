from typing import List, Tuple

import torch
from loguru import logger

Params = List[Tuple[torch.Tensor, torch.Tensor]]


def predict(params: Params, activations: torch.Tensor) -> torch.Tensor:
    for w, b in params:
        outputs = torch.matmul(activations, w) + b
        logger.info(f"Shape: {outputs.shape}")
        activations = outputs

    return outputs


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor([0.0]), x)


def nn_predict(params: Params, activations: torch.Tensor) -> torch.Tensor:
    for w, b in params[:-1]:
        outputs = torch.matmul(activations, w) + b
        logger.info(f"Shape: {outputs.shape}")
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = torch.matmul(activations, final_w) + final_b
    logger.info(f"Shape: {logits.shape}")
    return logits
