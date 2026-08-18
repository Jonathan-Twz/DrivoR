#!/usr/bin/env python3
"""Visualize proposal-conditioned latent BEV rollouts for idea_0002_01."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from omegaconf import OmegaConf

from navsim.agents.drivoR.drivor_model import DrivoRModel


WORKSPACE = Path("/mnt/ws-frb/users/jingyuso/wenzhet")
REFERENCE_ROOT = WORKSPACE / "DrivoR"
DEFAULT_CHECKPOINT = (
    REFERENCE_ROOT
    / "exp/ke/Aug18-idea0002-01-proposal-world-fast/08.18_purrgil_final_pair2"
    / "checkpoints/best-epoch=3-step=64.ckpt"
)
DEFAULT_CACHE = REFERENCE_ROOT / "exp/navsim_cache_future_bev_oracle_decoder_scorer"
DEFAULT_BEV_ROOT = WORKSPACE / "navsim_bev_feature/exports_pretrained"
DEFAULT_LOG = "2021.06.07.12.54.00_veh-35_01843_02314"
DEFAULT_TOKEN = "d973628ca1235533"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bev-root", type=Path, default=DEFAULT_BEV_ROOT)
    parser.add_argument("--log-name", default=DEFAULT_LOG)
    parser.add_argument("--scene-token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/idea0002_01/rollouts"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config() -> object:
    config_path = Path("navsim/planning/script/config/common/agent/drivoR.yaml")
    config = OmegaConf.load(config_path).config
    config.use_bev_feature = True
    config.use_bev_in_decoder = False
    config.use_bev_in_scorer = False
    config.use_bev_residual_proposal_refiner = False
    config.use_proposal_world_refiner = True
    config.use_privileged_future_bev = False
    config.freeze_pretrained_except_bev_scorer = True
    config.bev_feature_type = "decoder_neck"
    config.bev_channels = 256
    config.bev_spatial_hw = [128, 128]
    config.refiner_ls_values = 0.0
    config.image_backbone.model_weights = str(
        REFERENCE_ROOT / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
    )
    config.image_backbone.focus_front_cam = False
    config.one_token_per_traj = True
    config.refiner_num_heads = 1
    config.tf_d_model = 256
    config.tf_d_ffn = 1024
    config.area_pred = False
    config.agent_pred = False
    config.ref_num = 4
    config.long_trajectory_additional_poses = 2
    config.proposal_world_refiner.num_layers = 2
    config.proposal_world_refiner.num_heads = 4
    config.proposal_world_refiner.ffn_dim = 512
    config.proposal_world_refiner.rollout_steps = 1
    config.proposal_world_refiner.proposal_chunk_size = 8
    config.proposal_world_refiner.init_refine_gate = 0.0
    config.proposal_world_refiner.init_score_gate = 0.0
    return config


def load_features(args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    feature_path = args.cache_root / args.log_name / args.scene_token / "drivor_feature.gz"
    with gzip.open(feature_path, "rb") as file:
        features = pickle.load(file)

    bev_path = (
        args.bev_root
        / "trainval"
        / args.log_name
        / f"{args.scene_token}_decoder_neck.pt"
    )
    features["bev_feature"] = torch.load(bev_path, map_location="cpu", weights_only=True)
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in features.items()
        if torch.is_tensor(value)
    }


def load_model(checkpoint: Path, device: torch.device) -> DrivoRModel:
    model = DrivoRModel(build_config())
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "agent._drivor_model."
    state_dict = {
        key[len(prefix) :]: value
        for key, value in checkpoint_data["state_dict"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return model.eval().to(device)


def run_inference(model: DrivoRModel, features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    current_tokens: List[torch.Tensor] = []
    future_chunks: List[torch.Tensor] = []

    bev_hook = model.bev_tokenizer.register_forward_hook(
        lambda _module, _inputs, output: current_tokens.append(output.detach().cpu())
    )
    world_hook = model.proposal_world_refiner.world_model.register_forward_hook(
        lambda _module, _inputs, output: future_chunks.append(output.detach().cpu())
    )
    try:
        with torch.inference_mode():
            output = model(features)
    finally:
        bev_hook.remove()
        world_hook.remove()

    if len(current_tokens) != 1 or not future_chunks:
        raise RuntimeError(
            f"Expected one current-token call and rollout chunks, got {len(current_tokens)} and {len(future_chunks)}"
        )

    current = current_tokens[0][0]
    futures = torch.cat(future_chunks, dim=1)[0]
    base_proposals = output["proposal_list"][-2][0].detach().cpu()
    scores = output["pdm_score"][0].detach().cpu()
    return {
        "current_tokens": current.numpy(),
        "future_tokens": futures.numpy(),
        "base_proposals": base_proposals.numpy(),
        "scores": scores.numpy(),
        "refine_gate": np.asarray(float(model.proposal_world_refiner.refine_gate.detach().cpu())),
        "score_gate": np.asarray(float(model.proposal_world_refiner.score_gate.detach().cpu())),
    }


def shared_pca_rgb(current: np.ndarray, futures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([current[None], futures], axis=0)
    flat = stacked.reshape(-1, stacked.shape[-1]).astype(np.float64)
    flat -= flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(flat, full_matrices=False)
    projected = (flat @ vh[:3].T).reshape(stacked.shape[0], 8, 8, 3)
    rgb = np.empty_like(projected, dtype=np.float32)
    for channel in range(3):
        low, high = np.percentile(projected[..., channel], [1.0, 99.0])
        rgb[..., channel] = np.clip((projected[..., channel] - low) / (high - low + 1e-8), 0, 1)
    return rgb[0], rgb[1:]


def select_diverse_candidates(proposals: np.ndarray, scores: np.ndarray) -> List[int]:
    lateral = proposals[:, -1, 1]
    ordered = np.argsort(lateral)
    candidates = [
        int(ordered[0]),
        int(ordered[len(ordered) // 4]),
        int(np.argmax(scores)),
        int(ordered[(3 * len(ordered)) // 4]),
        int(ordered[-1]),
    ]
    selected = []
    for index in candidates + list(np.argsort(scores)[::-1]):
        if index not in selected:
            selected.append(int(index))
        if len(selected) == 5:
            break
    return selected


def plot_selected(data: Dict[str, np.ndarray], output_path: Path) -> List[int]:
    current = data["current_tokens"]
    futures = data["future_tokens"]
    proposals = data["base_proposals"]
    scores = data["scores"]
    selected = select_diverse_candidates(proposals, scores)
    current_rgb, future_rgb = shared_pca_rgb(current, futures)
    delta = (
        np.linalg.norm(futures - current[None], axis=-1) / np.sqrt(current.shape[-1])
    ).reshape(futures.shape[0], 8, 8)
    pairwise_rms = np.linalg.norm(
        futures[:, None] - futures[None, :], axis=-1
    ).mean() / np.sqrt(current.shape[-1])
    vmax = float(np.percentile(delta, 99.0))
    colors = ["#2166AC", "#67A9CF", "#F4A261", "#EF8354", "#B2182B"]

    fig = plt.figure(figsize=(17, 7.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, width_ratios=[1.22, 1, 1, 1, 1, 1])
    trajectory_axis = fig.add_subplot(grid[0, 0])
    for proposal in proposals:
        trajectory_axis.plot(proposal[:, 0], proposal[:, 1], color="#CAD1D8", linewidth=0.7)
    for color, index in zip(colors, selected):
        trajectory_axis.plot(
            proposals[index, :, 0],
            proposals[index, :, 1],
            marker="o",
            markersize=2.6,
            color=color,
            linewidth=2.2,
            label=f"#{index}",
        )
    trajectory_axis.scatter([0], [0], marker="^", s=75, color="#172026", label="ego")
    trajectory_axis.set_title("64 base proposals", fontweight="bold")
    trajectory_axis.set_xlabel("longitudinal")
    trajectory_axis.set_ylabel("lateral")
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.axis("equal")
    trajectory_axis.legend(fontsize=8, ncol=2, loc="best")

    current_axis = fig.add_subplot(grid[1, 0])
    current_axis.imshow(current_rgb, interpolation="nearest")
    current_axis.set_title("Current BEV tokens\nshared PCA → RGB", fontweight="bold")
    current_axis.set_xticks([])
    current_axis.set_yticks([])

    norm = Normalize(vmin=0.0, vmax=vmax)
    image = None
    for column, (color, index) in enumerate(zip(colors, selected), start=1):
        future_axis = fig.add_subplot(grid[0, column])
        future_axis.imshow(future_rgb[index], interpolation="nearest")
        future_axis.set_title(
            f"Proposal #{index}\nscore={scores[index]:.3f}", color=color, fontweight="bold"
        )
        future_axis.set_xticks([])
        future_axis.set_yticks([])

        delta_axis = fig.add_subplot(grid[1, column])
        image = delta_axis.imshow(delta[index], cmap="magma", norm=norm, interpolation="nearest")
        delta_axis.set_title(
            f"||future − current||\nmean={delta[index].mean():.3f}", fontsize=10
        )
        delta_axis.set_xticks([])
        delta_axis.set_yticks([])

    if image is not None:
        fig.colorbar(image, ax=fig.axes[2:], shrink=0.72, label="RMS latent feature change")
    fig.suptitle(
        "Proposal-Conditioned BEV World-Model Rollouts",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        (
            "One latent rollout update; PCA colors are not semantic classes.  "
            f"Mean current→future RMS={delta.mean():.4f}; candidate-pair RMS={pairwise_rms:.4f}."
        ),
        ha="center",
        fontsize=10,
        color="#5E6A72",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return selected


def plot_all_rollouts(data: Dict[str, np.ndarray], output_path: Path) -> None:
    futures = data["future_tokens"]
    scores = data["scores"]
    mean_future = futures.mean(axis=0, keepdims=True)
    proposal_specific = (
        np.linalg.norm(futures - mean_future, axis=-1) / np.sqrt(futures.shape[-1])
    ).reshape(futures.shape[0], 8, 8)
    vmax = float(np.percentile(proposal_specific, 99.0))
    best = int(np.argmax(scores))

    fig, axes = plt.subplots(8, 8, figsize=(13, 13), constrained_layout=True)
    image = None
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(
            proposal_specific[index], cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest"
        )
        axis.set_title(
            f"#{index}  {scores[index]:.2f}",
            fontsize=7.5,
            color="#C95D3A" if index == best else "#172026",
            fontweight="bold" if index == best else "normal",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#C95D3A" if index == best else "#D8DEE3")
            spine.set_linewidth(2.0 if index == best else 0.6)
    if image is not None:
        fig.colorbar(
            image, ax=axes, shrink=0.72, label="RMS deviation from mean future rollout"
        )
    fig.suptitle(
        "All 64 Proposal-Specific Future-BEV Deviations",
        fontsize=20,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)
    features = load_features(args, device)
    data = run_inference(model, features)

    selected_path = args.output_dir / f"{args.scene_token}_selected_rollouts.png"
    all_path = args.output_dir / f"{args.scene_token}_all64_rollouts.png"
    data_path = args.output_dir / f"{args.scene_token}_rollouts.npz"
    selected = plot_selected(data, selected_path)
    plot_all_rollouts(data, all_path)
    np.savez_compressed(data_path, selected=np.asarray(selected), **data)

    delta = np.linalg.norm(
        data["future_tokens"] - data["current_tokens"][None], axis=-1
    ) / np.sqrt(data["current_tokens"].shape[-1])
    pairwise = np.linalg.norm(
        data["future_tokens"][:, None] - data["future_tokens"][None, :], axis=-1
    ).mean(axis=-1) / np.sqrt(data["current_tokens"].shape[-1])
    print(f"scene_token={args.scene_token}")
    print(f"selected_candidates={selected}")
    print(f"refine_gate={float(data['refine_gate']):.8e}")
    print(f"score_gate={float(data['score_gate']):.8e}")
    print(f"mean_future_current_rms={delta.mean():.6f}")
    print(f"mean_candidate_pairwise_rms={pairwise.mean():.6f}")
    print(selected_path.resolve())
    print(all_path.resolve())
    print(data_path.resolve())


if __name__ == "__main__":
    main()
