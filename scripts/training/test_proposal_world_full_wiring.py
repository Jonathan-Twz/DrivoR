#!/usr/bin/env python3
"""Check baseline loading and the proposal-world freeze whitelist."""

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nuplan-devkit"))
REFERENCE_ROOT = Path(
    os.environ.get("REFERENCE_ROOT", "/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR")
)
os.environ.setdefault("NAVSIM_EXP_ROOT", str(REFERENCE_ROOT / "exp"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/drivor-matplotlib-idea0002")


def main() -> None:
    config_dir = ROOT / "navsim/planning/script/config/training"
    baseline = REFERENCE_ROOT / "weights/checkpoints/drivor_Nav1_25epochs.pth"
    backbone = (
        REFERENCE_ROOT
        / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
    )
    overrides = [
        "agent=drivoR",
        f"agent.checkpoint_path={baseline}",
        "agent.config.use_bev_feature=true",
        "agent.config.use_bev_in_decoder=false",
        "agent.config.use_bev_in_scorer=false",
        "agent.config.use_bev_residual_proposal_refiner=false",
        "agent.config.use_proposal_world_refiner=true",
        "agent.config.use_privileged_future_bev=false",
        "agent.config.freeze_pretrained_except_bev_scorer=true",
        "agent.config.bev_channels=256",
        f"agent.config.image_backbone.model_weights={backbone}",
        f"agent.config.lidar_backbone.model_weights={backbone}",
        "agent.loss=null",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="default_training", overrides=overrides)

    agent = instantiate(cfg.agent)
    agent.initialize()
    params = agent._collect_trainable_params()
    trainable_names = [
        name
        for name, parameter in agent._drivor_model.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name in trainable_names
        if not (
            name.startswith("bev_tokenizer.")
            or name.startswith("proposal_world_refiner.")
        )
    ]
    if invalid:
        raise AssertionError(f"unexpected trainable parameters: {invalid[:20]}")
    if not any(name.startswith("proposal_world_refiner.") for name in trainable_names):
        raise AssertionError("proposal world refiner is not trainable")

    trainable_count = sum(parameter.numel() for parameter in params)
    total_count = sum(parameter.numel() for parameter in agent._drivor_model.parameters())
    print("PASS baseline checkpoint compatibility")
    print("PASS freeze whitelist")
    print(f"trainable_tensors={len(params)}")
    print(f"trainable_parameters={trainable_count}")
    print(f"total_parameters={total_count}")
    print(f"trainable_fraction={trainable_count / total_count:.6f}")


if __name__ == "__main__":
    main()
