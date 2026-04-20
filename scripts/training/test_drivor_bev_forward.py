#!/usr/bin/env python3
"""Sanity check: DrivoR forward with optional BEV branch (no dataset)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from navsim.agents.drivoR.drivor_model import DrivoRModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="", help="Optional .pth Lightning checkpoint (partial load)")
    p.add_argument("--bev-channels", type=int, default=80)
    args = p.parse_args()

    root = OmegaConf.load(_REPO / "navsim/planning/script/config/common/agent/drivoR.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(root.config, resolve=True))
    cfg.use_bev_feature = True
    cfg.bev_channels = args.bev_channels

    model = DrivoRModel(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("agent._drivor_model.", ""): v for k, v in sd.items() if "drivor_model" in k}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("load_state_dict strict=False:", "missing", len(missing), "unexpected", len(unexpected))

    B = 2
    H, W = cfg.image_size
    n_cams = 4
    features = {
        "image": torch.randn(B, n_cams, 3, W, H),
        "ego_status": torch.randn(B, 4, 11),
        "bev_feature": torch.randn(B, int(cfg.bev_channels), 128, 128),
    }
    model.eval()
    with torch.no_grad():
        out = model(features)
    assert "trajectory" in out and out["trajectory"].shape[0] == B
    print("OK: trajectory shape", tuple(out["trajectory"].shape))


if __name__ == "__main__":
    main()
