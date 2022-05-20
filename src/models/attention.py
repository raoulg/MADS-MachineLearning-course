from typing import Tuple

import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    """
    pytorch implementation of attention layers.
    mainly a simplified implementation of the tensorflow attention class found here
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/layers/dense_attention.py
    The BahdanauAttention is an implementation of this tensorflow tutorial
    https://www.tensorflow.org/text/tutorials/nmt_with_attention
    I use the query-key-values convention.
    A good explanation of attention can be found here:
    https://www.youtube.com/watch?v=yGTUuEx3GkA

    basic flow is:
    calculate attention scores based on values and query
    apply attention layers to values

    note that all attentions layers are batch first.
    """

    def __init__(self) -> None:
        super(BaseAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def _calculate_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_scores(
        self, scores: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO implement mask
        weights = self.softmax(scores)
        return torch.bmm(weights, value), weights

    def forward(
        self, query: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self._calculate_scores(query=query, key=value)
        result, attention_scores = self._apply_scores(scores, value)
        return result, attention_scores


class Attention(BaseAttention):
    """
    basic dot prodcut attention without learnable weights
    """

    def __init__(self, use_scale: bool = True) -> None:
        super(Attention, self).__init__()
        self.use_scale = use_scale
        if use_scale:
            self.scale = nn.Parameter(torch.ones(1))

    def _calculate_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        scores = torch.bmm(query, key.transpose(1, 2).contiguous())
        if self.use_scale:
            scores *= self.scale
        return scores


class AdditiveAttention(BaseAttention):
    """
    additive attention without learnable weights
    """

    def __init__(self, units: int, use_scale: bool = True) -> None:
        super(AdditiveAttention, self).__init__()
        self.use_scale = use_scale
        if use_scale:
            self.scale = self._initialize_weights(units)

    def _calculate_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Note that this is all batch-first

        Args:
            query [batch, q_len, dim]
            key [batch, v_len, dim]
        Returns:
            [batch, q_len, v_len]
        """
        if self.use_scale:
            scale = self.scale
        else:
            scale = torch.Tensor([1])
        # [batch, q_len, 1, dim]
        query = query.unsqueeze(-2)
        # [batch, 1, v_len, dim]
        key = key.unsqueeze(-3)
        # [batch, q_len, v_len, dim]
        score = torch.tanh(query + key)

        # [batch, q_len, v_len]
        return torch.sum(scale * score, dim=-1)

    def _initialize_weights(self, units: int) -> torch.Tensor:
        return nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, units)))


class BahdanauAttention(BaseAttention):
    """
    additive attention, but first transform the key and query

    """

    def __init__(self, units: int, use_scale: bool = True) -> None:
        super(BahdanauAttention, self).__init__()
        self.query_w = nn.Linear(units, units, bias=False)
        self.key_w = nn.Linear(units, units, bias=False)
        self.use_scale = use_scale
        if use_scale:
            self.scale = self._initialize_weights(units)

    def _calculate_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query [batch, q_len, dim]
            key [batch, v_len, dim]
        Returns:
            [batch, q_len, v_len]
        """
        if self.use_scale:
            scale = self.scale
        else:
            scale = torch.Tensor([1])
        key = self.key_w(key)
        query = self.query_w(query)

        # [batch, q_len, 1, dim]
        query = query.unsqueeze(-2)
        # [batch, 1, v_len, dim]
        key = key.unsqueeze(-3)
        # [batch, q_len, v_len, dim]
        score = torch.tanh(query + key)

        # [batch, q_len, v_len]
        return torch.sum(scale * score, dim=-1)

    def _initialize_weights(self, units: int) -> torch.Tensor:
        return nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, units)))
