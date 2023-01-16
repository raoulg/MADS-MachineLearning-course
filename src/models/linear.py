from typing import Dict, List, Tuple

import torch
import torch.nn as nn
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


class NeuralNetwork(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(config["input"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(config["h2"], config["output"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits
