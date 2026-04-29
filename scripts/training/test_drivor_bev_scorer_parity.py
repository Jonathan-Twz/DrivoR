"""Parity test for the BEV-aware scorer.

Builds both ``TransformerDecoderScorer`` and ``BevAwareScorer`` with the same
random weights, copies the frozen-branch parameters from the BEV-aware scorer
into the reference, and verifies that at initialisation (new cross_attn_bev
gated to 0, side-LoRA out_proj zero-initialised) the outputs match bit-for-bit
(up to floating-point noise).

Run::

    python scripts/training/test_drivor_bev_scorer_parity.py [--tolerance 1e-5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _scorer_config():
    return OmegaConf.create(
        {
            "scorer_ref_num": 4,
            "tf_d_model": 256,
            "refiner_num_heads": 1,
            "refiner_ls_values": 0.0,
            "scorer_bev": {
                "lora_rank": 8,
                "lora_dropout": 0.0,
                "add_new_cross_attn": True,
                "init_gate": 0.0,
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n_traj", type=int, default=64)
    parser.add_argument("--k_scene", type=int, default=128)
    parser.add_argument("--k_bev", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
    from navsim.agents.drivoR.layers.bev_scorer_blocks import BevAwareScorer

    config = _scorer_config()
    d = int(config.tf_d_model)

    ref = TransformerDecoderScorer(
        num_layers=config.scorer_ref_num, d_model=d, proj_drop=0.0, drop_path=0.0, config=config
    ).eval()
    bev = BevAwareScorer(
        num_layers=config.scorer_ref_num, d_model=d, proj_drop=0.0, drop_path=0.0, config=config
    ).eval()

    # Copy all matching parameter names from ref into bev so frozen branches match.
    ref_state = ref.state_dict()
    bev_state = bev.state_dict()
    copied = 0
    for k, v in ref_state.items():
        if k in bev_state and bev_state[k].shape == v.shape:
            bev_state[k] = v.clone()
            copied += 1
    bev.load_state_dict(bev_state)
    print(f"Copied {copied} tensors from reference scorer into BEV-aware scorer.")

    x = torch.randn(args.batch, args.n_traj, d)
    scene = torch.randn(args.batch, args.k_scene, d)
    bev_tokens = torch.randn(args.batch, args.k_bev, d)

    with torch.no_grad():
        out_ref = ref(x, scene)
        out_bev = bev(x, scene, bev_tokens)

    diff = (out_ref - out_bev).abs().max().item()
    print(f"Max |ref - bev| = {diff:.3e}")

    # Also verify that, with bev_tokens=None, the new branch is fully skipped.
    with torch.no_grad():
        out_bev_no_bev = bev(x, scene, None)
    diff_no_bev = (out_ref - out_bev_no_bev).abs().max().item()
    print(f"Max |ref - bev(no bev tokens)| = {diff_no_bev:.3e}")

    ok = diff < args.tolerance and diff_no_bev < args.tolerance
    if not ok:
        print(f"FAIL: parity not within tolerance {args.tolerance:.1e}.")
        return 1
    print(f"PASS: BevAwareScorer matches reference within {args.tolerance:.1e}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
