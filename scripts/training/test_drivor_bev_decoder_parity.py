"""Parity test for the BEV-aware trajectory decoder.

Builds both ``TransformerDecoder`` and ``BevAwareTrajectoryDecoder`` with the
same random weights, copies the frozen-branch parameters from the BEV-aware
decoder into the reference, and verifies that:

1. every reference state_dict key exists (with matching shape) in the
   BEV-aware decoder, so a pre-trained checkpoint loads cleanly;
2. with ``decoder_bev.init_gate=0`` the stacked intermediate outputs match the
   reference bit-for-bit (up to floating-point noise), both with BEV tokens
   and with ``x_bev=None``;
3. with a non-zero gate and BEV tokens the outputs differ (the new branch is
   actually wired in).

Run::

    python scripts/training/test_drivor_bev_decoder_parity.py [--tolerance 1e-5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _decoder_config(init_gate: float = 0.0, lora_rank: int = 0):
    return OmegaConf.create(
        {
            "ref_num": 4,
            "tf_d_model": 256,
            "refiner_num_heads": 1,
            "refiner_ls_values": 0.0,
            "decoder_bev": {
                "init_gate": init_gate,
                "lora_rank": lora_rank,
                "lora_dropout": 0.0,
            },
        }
    )


def _copy_matching(src: torch.nn.Module, dst: torch.nn.Module) -> int:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = 0
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k] = v.clone()
            copied += 1
    dst.load_state_dict(dst_state)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n_traj", type=int, default=64)
    parser.add_argument("--k_scene", type=int, default=128)
    parser.add_argument("--k_bev", type=int, default=64)
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from navsim.agents.drivoR.transformer_decoder import TransformerDecoder
    from navsim.agents.drivoR.layers.bev_decoder_blocks import BevAwareTrajectoryDecoder

    config = _decoder_config(init_gate=0.0, lora_rank=args.lora_rank)
    d = int(config.tf_d_model)

    ref = TransformerDecoder(proj_drop=0.0, drop_path=0.0, config=config).eval()
    bev = BevAwareTrajectoryDecoder(proj_drop=0.0, drop_path=0.0, config=config).eval()

    # 1. Checkpoint-key compatibility: every reference key must exist in the
    #    BEV-aware decoder with the same shape.
    ref_state = ref.state_dict()
    bev_state = bev.state_dict()
    missing = [
        k for k, v in ref_state.items() if k not in bev_state or bev_state[k].shape != v.shape
    ]
    extra = [k for k in bev_state if k not in ref_state]
    bad_extra = [k for k in extra if "cross_attn_bev" not in k and "_lora" not in k]
    print(f"Reference keys missing in BEV decoder: {len(missing)}")
    print(f"Extra keys in BEV decoder: {len(extra)} (non cross_attn_bev/lora: {len(bad_extra)})")
    if missing or bad_extra:
        print(f"FAIL: state_dict mismatch. missing={missing[:5]} bad_extra={bad_extra[:5]}")
        return 1

    copied = _copy_matching(ref, bev)
    print(f"Copied {copied} tensors from reference decoder into BEV-aware decoder.")

    x = torch.randn(args.batch, args.n_traj, d)
    scene = torch.randn(args.batch, args.k_scene, d)
    bev_tokens = torch.randn(args.batch, args.k_bev, d)

    # 2. Parity at init_gate=0.
    with torch.no_grad():
        out_ref = ref(x, scene)
        out_bev = bev(x, scene, bev_tokens)
        out_bev_none = bev(x, scene, None)

    assert out_ref.shape == out_bev.shape == out_bev_none.shape
    diff = (out_ref - out_bev).abs().max().item()
    diff_none = (out_ref - out_bev_none).abs().max().item()
    print(f"Max |ref - bev| = {diff:.3e}")
    print(f"Max |ref - bev(no bev tokens)| = {diff_none:.3e}")

    # 3. Non-zero gate actually changes the output.
    torch.manual_seed(args.seed)
    bev_gated = BevAwareTrajectoryDecoder(
        proj_drop=0.0,
        drop_path=0.0,
        config=_decoder_config(init_gate=0.1, lora_rank=args.lora_rank),
    ).eval()
    _copy_matching(ref, bev_gated)
    with torch.no_grad():
        out_gated = bev_gated(x, scene, bev_tokens)
        out_gated_none = bev_gated(x, scene, None)
    diff_gated = (out_ref - out_gated).abs().max().item()
    diff_gated_none = (out_ref - out_gated_none).abs().max().item()
    print(f"Max |ref - bev(gate=0.1, bev tokens)| = {diff_gated:.3e} (should be > 0)")
    print(f"Max |ref - bev(gate=0.1, no bev tokens)| = {diff_gated_none:.3e}")

    ok = (
        diff < args.tolerance
        and diff_none < args.tolerance
        and diff_gated_none < args.tolerance
        and diff_gated > args.tolerance
    )
    if not ok:
        print(f"FAIL: parity/wiring checks not satisfied (tolerance {args.tolerance:.1e}).")
        return 1
    print(f"PASS: BevAwareTrajectoryDecoder matches reference within {args.tolerance:.1e} "
          f"and BEV branch is active when gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
