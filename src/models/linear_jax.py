from typing import List, Tuple

import jax.numpy as jnp
from loguru import logger

Params = List[Tuple[jnp.ndarray, jnp.ndarray]]


def predict(params: Params, activations: jnp.ndarray) -> jnp.ndarray:
    for w, b in params:
        outputs = jnp.dot(activations, w) + b
        logger.info(f"Shape: {outputs.shape}")
        activations = outputs

    return outputs


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def nn_predict(params: Params, activations: jnp.ndarray) -> jnp.ndarray:
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        logger.info(f"Shape: {outputs.shape}")
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    logger.info(f"Shape: {logits.shape}")
    return logits
