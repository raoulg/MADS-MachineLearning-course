"""What one image costs a model, as `Architecture.of` measures it: the FLOPs of its
forward pass and the time that pass takes."""

import torch
from torch import nn
from vectormesh.components import MoE, SparseMoE

from hpo.measure import Architecture

P, K, N_EXPERTS, WIDTH = 64, 321, 8, 8


def dense(width: int) -> nn.Module:
    return nn.Sequential(nn.Linear(P, width), nn.GELU(), nn.Linear(width, K))


def test_a_linear_layer_costs_a_multiply_and_an_add_per_weight():
    arch = Architecture.of(nn.Linear(10, 3), torch.zeros(2, 10))
    assert arch.flops_per_image == 2 * 10 * 3


def test_a_dense_head_counts_both_its_layers():
    arch = Architecture.of(dense(16), torch.zeros(2, P))
    assert arch.flops_per_image == 2 * (P * 16 + 16 * K)


def test_a_sparse_mixture_counts_its_router_and_the_one_expert_that_ran():
    torch.manual_seed(0)
    moe = SparseMoE([dense(WIDTH) for _ in range(N_EXPERTS)], P, K, k=1)
    router, expert = 2 * P * N_EXPERTS, 2 * (P * WIDTH + WIDTH * K)
    assert (router, expert) == (1_024, 6_160)
    for example in (torch.zeros(2, P), torch.randn(2, P)):
        assert Architecture.of(moe, example).flops_per_image == router + expert


def test_a_dense_mixture_counts_every_expert():
    moe = MoE([dense(WIDTH) for _ in range(N_EXPERTS)], P, K)
    arch = Architecture.of(moe, torch.zeros(2, P))
    expert = 2 * (P * WIDTH + WIDTH * K)
    # the softmax-weighted sum of the experts is no matrix product, so it is not counted
    assert arch.flops_per_image == 2 * P * N_EXPERTS + N_EXPERTS * expert


def test_the_latency_is_measured_and_the_mode_restored():
    model = dense(16).train()
    arch = Architecture.of(model, torch.zeros(2, P))
    assert arch.latency_us is not None and arch.latency_us > 0
    assert model.training


def test_a_model_with_two_streams_is_timed_on_both():
    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(6, 2)

        def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return self.head(torch.cat(x, dim=-1))

    arch = Architecture.of(Two(), (torch.zeros(2, 4), torch.zeros(2, 2)))
    assert arch.flops_per_image == 2 * 6 * 2
    assert arch.latency_us is not None and arch.latency_us > 0
