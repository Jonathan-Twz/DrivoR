"""Parity, wiring, and staged-gradient test for the BEV residual refiner."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from navsim.agents.drivoR.layers.bev_residual_proposal_refiner import (  # noqa: E402
    BevResidualProposalRefiner,
)


def _config(init_alpha: float = 0.0):
    return OmegaConf.create(
        {
            "num_poses": 8,
            "tf_d_model": 256,
            "tf_d_ffn": 1024,
            "bev_residual_proposal_refiner": {
                "num_layers": 1,
                "num_heads": 1,
                "init_alpha": init_alpha,
                "dropout": 0.0,
            },
        }
    )


def main() -> int:
    torch.manual_seed(0)
    base = torch.randn(2, 64, 8, 3)
    queries = torch.randn(2, 64, 256)
    bev = torch.randn(2, 64, 256)

    refiner = BevResidualProposalRefiner(_config(init_alpha=0.0))
    refined, delta = refiner(base, queries, bev)
    assert refined.shape == delta.shape == base.shape
    assert torch.equal(refined, base), "alpha=0 must exactly preserve base proposals"

    no_bev, no_bev_delta = refiner(base, queries, None)
    assert torch.equal(no_bev, base)
    assert torch.count_nonzero(no_bev_delta) == 0

    loss = refined.square().mean()
    loss.backward()
    assert refiner.alpha.grad is not None and refiner.alpha.grad.abs().item() > 0
    first_step_inner_grad = sum(
        p.grad.abs().sum().item()
        for name, p in refiner.named_parameters()
        if name != "alpha" and p.grad is not None
    )
    assert first_step_inner_grad == 0.0

    with torch.no_grad():
        refiner.alpha.add_(-1e-2 * refiner.alpha.grad)
    refiner.zero_grad(set_to_none=True)
    refined_step_two, _ = refiner(base, queries, bev)
    refined_step_two.square().mean().backward()
    second_step_inner_grad = sum(
        p.grad.abs().sum().item()
        for name, p in refiner.named_parameters()
        if name != "alpha" and p.grad is not None
    )
    assert second_step_inner_grad > 0.0
    assert not torch.equal(refined_step_two, base)

    print("PASS: alpha=0 parity, no-BEV bypass, nonzero wiring, and staged gradients verified.")
    print(f"alpha after synthetic step: {refiner.alpha.item():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
