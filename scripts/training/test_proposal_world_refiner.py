#!/usr/bin/env python3
"""CPU checks for the proposal-conditioned BEV world refiner."""

from copy import deepcopy

import torch
from omegaconf import OmegaConf

from navsim.agents.drivoR.layers.proposal_world_refiner import (
    ProposalConditionedWorldRefiner,
)


def make_config(chunk_size: int = 2, refine_gate: float = 0.0, score_gate: float = 0.0):
    return OmegaConf.create(
        {
            "tf_d_model": 32,
            "tf_d_ffn": 64,
            "num_poses": 8,
            "bev_tokenizer": {"num_tokens": 16},
            "proposal_world_refiner": {
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 64,
                "rollout_steps": 1,
                "proposal_chunk_size": chunk_size,
                "init_refine_gate": refine_gate,
                "init_score_gate": score_gate,
                "dropout": 0.0,
            },
        }
    )


def inputs():
    torch.manual_seed(7)
    base = torch.randn(2, 5, 8, 3)
    queries = torch.randn(2, 5, 32)
    bev = torch.randn(2, 16, 32)
    return base, queries, bev


def test_zero_gate_parity() -> None:
    module = ProposalConditionedWorldRefiner(make_config()).eval()
    base, queries, bev = inputs()
    refined, delta, score_residual, diagnostics = module(base, queries, bev)
    torch.testing.assert_close(refined, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(score_residual, torch.zeros_like(score_residual), atol=0.0, rtol=0.0)
    assert delta.shape == base.shape
    assert set(diagnostics) >= {
        "proposal_world_refine_gate",
        "proposal_world_score_gate",
        "proposal_world_context_norm",
    }


def test_nonzero_gate_candidate_wiring() -> None:
    module = ProposalConditionedWorldRefiner(
        make_config(refine_gate=1.0, score_gate=1.0)
    ).eval()
    base, queries, bev = inputs()
    refined, delta, score_residual, _ = module(base, queries, bev)
    assert refined.shape == base.shape
    assert score_residual.shape == queries.shape
    assert not torch.allclose(refined, base)
    assert not torch.allclose(delta[:, 0], delta[:, 1])
    assert not torch.allclose(score_residual[:, 0], score_residual[:, 1])


def test_chunking_is_numerically_equivalent() -> None:
    chunked = ProposalConditionedWorldRefiner(make_config(chunk_size=2)).eval()
    unchunked = ProposalConditionedWorldRefiner(make_config(chunk_size=64)).eval()
    unchunked.load_state_dict(deepcopy(chunked.state_dict()))
    base, queries, bev = inputs()
    out_chunked = chunked(base, queries, bev)
    out_unchunked = unchunked(base, queries, bev)
    for left, right in zip(out_chunked[:3], out_unchunked[:3]):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_two_step_gate_activation() -> None:
    module = ProposalConditionedWorldRefiner(make_config()).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    base, queries, bev = inputs()
    trajectory_target = torch.zeros_like(base)
    score_target = torch.ones_like(queries)

    refined, _, score_residual, _ = module(base, queries, bev)
    loss = (refined - trajectory_target).square().mean()
    loss = loss + (score_residual - score_target).square().mean()
    loss.backward()
    assert module.refine_gate.grad is not None and module.refine_gate.grad.abs() > 0
    assert module.score_gate.grad is not None and module.score_gate.grad.abs() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    refined, _, score_residual, _ = module(base, queries, bev)
    loss = (refined - trajectory_target).square().mean()
    loss = loss + (score_residual - score_target).square().mean()
    loss.backward()
    world_grad = sum(
        parameter.grad.abs().sum()
        for parameter in module.world_model.parameters()
        if parameter.grad is not None
    )
    assert world_grad > 0


def main() -> None:
    checks = [
        test_zero_gate_parity,
        test_nonzero_gate_candidate_wiring,
        test_chunking_is_numerically_equivalent,
        test_two_step_gate_activation,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")

    module = ProposalConditionedWorldRefiner(make_config())
    trainable = sum(parameter.numel() for parameter in module.parameters())
    print(f"trainable_parameters={trainable}")


if __name__ == "__main__":
    main()
