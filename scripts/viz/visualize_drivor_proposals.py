#!/usr/bin/env python
"""Visualize DrivoR proposal sets with NAVSIM BEV visualization primitives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import default_collate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "nuplan-devkit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", str(REPO_ROOT / "exp"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO_ROOT))
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset")
os.environ.setdefault(
    "BEV_FEATURES_ROOT", "/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained"
)

from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_MODELS = {
    "baseline_pretrained": {
        "checkpoint": REPO_ROOT / "weights/checkpoints/drivor_Nav1_25epochs.pth",
        "overrides": [
            "config.use_bev_feature=false",
            "config.use_bev_in_scorer=false",
            "config.use_bev_in_decoder=false",
        ],
    },
    "bev_scorer_finetune": {
        "checkpoint": REPO_ROOT
        / "exp/ke/golduck-4gpu-16batch-8worker-01gate-16lora/05.28_23.15/checkpoints/last.ckpt",
        "overrides": [
            "config.use_bev_feature=true",
            "config.use_bev_in_scorer=true",
            "config.use_bev_in_decoder=false",
            "config.use_bev_residual_proposal_refiner=false",
            "config.bev_feature_type=decoder_neck",
            "config.bev_channels=256",
            "config.scorer_bev.lora_rank=16",
            "config.bev_data_split=test",
            "config.long_trajectory_additional_poses=2",
        ],
    },
    "bev_decoder_finetune_last": {
        "checkpoint": REPO_ROOT
        / "exp/ke/Jun12-golduck-4gpu-lora16-bev-decoder/06.12_01.12/checkpoints/last.ckpt",
        "overrides": [
            "config.use_bev_feature=true",
            "config.use_bev_in_scorer=false",
            "config.use_bev_in_decoder=true",
            "config.use_bev_residual_proposal_refiner=false",
            "config.bev_feature_type=decoder_neck",
            "config.bev_channels=256",
            "config.decoder_bev.lora_rank=16",
            "config.bev_data_split=test",
            "config.long_trajectory_additional_poses=2",
        ],
    },
}


COMMON_AGENT_OVERRIDES = [
    "config.proposal_num=64",
    "config.refiner_ls_values=0.0",
    "config.image_backbone.focus_front_cam=false",
    "config.one_token_per_traj=true",
    "config.refiner_num_heads=1",
    "config.tf_d_model=256",
    "config.tf_d_ffn=1024",
    "config.area_pred=false",
    "config.agent_pred=false",
    "config.ref_num=4",
    "config.noc=1",
    "config.dac=1",
    "config.ddc=0.0",
    "config.ttc=5",
    "config.ep=5",
    "config.comfort=2",
    "config.use_ray_score=false",
    "lr_args=null",
    "scheduler_args=null",
    "loss=null",
    "batch_size=null",
    "num_gpus=1",
    "progress_bar=false",
]


def _set_default_env() -> None:
    """Retained for readability; defaults are set before NAVSIM imports above."""


def _compose_agent(model_name: str) -> object:
    model = DEFAULT_MODELS[model_name]
    checkpoint = Path(model["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")

    overrides = [
        f"checkpoint_path={checkpoint}",
        f"config.bev_features_root={os.environ['BEV_FEATURES_ROOT']}",
        *COMMON_AGENT_OVERRIDES,
        *model["overrides"],
    ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "navsim/planning/script/config/common/agent"),
    ):
        cfg = compose(config_name="drivoR", overrides=overrides)

    agent = instantiate(cfg)
    agent.initialize()
    agent.eval()
    return agent


def _compose_eval_cfg(split: str, max_scenes: Optional[int]) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides = [f"train_test_split={split}"]
    if max_scenes is not None:
        overrides.append(f"train_test_split.scene_filter.max_scenes={max_scenes}")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "navsim/planning/script/config/pdm_scoring"),
    ):
        return compose(config_name="default_run_create_submission_pickle", overrides=overrides)


def _load_scene_loader(
    agent: object,
    split: str,
    max_scenes: Optional[int],
    tokens: Optional[List[str]] = None,
) -> SceneLoader:
    cfg = _compose_eval_cfg(split, max_scenes)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if tokens:
        scene_filter.tokens = tokens
        scene_filter.max_scenes = len(tokens)
    return SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _to_device(features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in features.items()}


def _run_model_on_tokens(agent: object, loader: SceneLoader, tokens: Iterable[str], device: torch.device) -> Dict[str, Dict]:
    builder = agent.get_feature_builders()[0]
    agent.to(device)
    outputs: Dict[str, Dict] = {}

    with torch.no_grad():
        for token in tqdm(list(tokens), desc="Inference", leave=False):
            scene = loader.get_scene_from_token(token)
            agent_input = loader.get_agent_input_from_token(token)
            metadata = scene.scene_metadata
            features = builder.compute_features(
                agent_input,
                scene_token=metadata.initial_token,
                log_name=metadata.log_name,
            )
            batch = _to_device(default_collate([features]), device)
            pred = agent.forward(batch)
            proposals = pred["proposals"][0].detach().cpu()
            trajectory = pred["trajectory"][0].detach().cpu()
            pdm_score = pred.get("pdm_score")
            selected_index = None
            if pdm_score is not None:
                selected_index = int(torch.argmax(pdm_score[0].detach().cpu()).item())
            outputs[token] = {
                "proposals": proposals,
                "trajectory": trajectory,
                "selected_index": selected_index,
            }
    agent.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def _proposal_config(color, alpha: float = 0.9, width: float = 1.45) -> Dict:
    return {
        "line_color": color,
        "line_color_alpha": alpha,
        "line_width": width,
        "line_style": "-",
        "marker": ".",
        "marker_size": 2.8,
        "marker_edge_color": color,
        "zorder": 3,
    }


def _plot_model_scene(ax: plt.Axes, scene, model_name: str, result: Dict, color: str) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    proposals = result["proposals"]
    cmap = plt.get_cmap("turbo")
    denom = max(len(proposals) - 1, 1)
    for proposal_idx, proposal in enumerate(proposals):
        proposal_cfg = _proposal_config(cmap(proposal_idx / denom))
        add_trajectory_to_bev_ax(ax, Trajectory(proposal.numpy()), proposal_cfg)

    selected_traj = result["trajectory"].numpy()
    endpoint = selected_traj[-1]
    ax.scatter(
        endpoint[1],
        endpoint[0],
        marker="*",
        s=110,
        c="black",
        edgecolors="white",
        linewidths=0.9,
        zorder=9,
    )

    try:
        add_trajectory_to_bev_ax(ax, scene.get_future_trajectory(), TRAJECTORY_CONFIG["human"])
    except Exception:
        pass

    selected = result["selected_index"]
    suffix = f" selected={selected}" if selected is not None else ""
    ax.set_title(f"{model_name}{suffix}", fontsize=9)
    configure_bev_ax(ax)
    configure_ax(ax)


def _save_single(scene, token: str, model_name: str, result: Dict, output_dir: Path, color: str) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    _plot_model_scene(ax, scene, model_name, result, color)
    fig.tight_layout()
    path = output_dir / model_name / f"{token}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_comparison(scene, token: str, results_by_model: Dict[str, Dict], output_dir: Path) -> Path:
    colors = {
        "baseline_pretrained": "#d62728",
        "bev_scorer_finetune": "#1f77b4",
        "bev_decoder_finetune_last": "#2ca02c",
    }
    fig, axes = plt.subplots(1, len(results_by_model), figsize=(5 * len(results_by_model), 5))
    if len(results_by_model) == 1:
        axes = [axes]
    for ax, (model_name, result) in zip(axes, results_by_model.items()):
        _plot_model_scene(ax, scene, model_name, result, colors[model_name])
    fig.suptitle(token, fontsize=11)
    fig.tight_layout()
    path = output_dir / "comparison" / f"{token}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="navtest", help="Hydra train_test_split name.")
    parser.add_argument("--max-scenes", type=int, default=12, help="Number of scenes to visualize.")
    parser.add_argument("--tokens", nargs="*", default=None, help="Optional explicit scene tokens to visualize.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "exp/visualizations/drivor_proposals_navtest")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_default_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    print(f"Saving visualizations to: {args.output_dir}")

    first_agent = _compose_agent("baseline_pretrained")
    loader_max_scenes = None if args.tokens else args.max_scenes
    loader = _load_scene_loader(first_agent, args.split, loader_max_scenes, args.tokens)
    if args.tokens:
        available = set(loader.tokens)
        missing = [token for token in args.tokens if token not in available]
        if missing:
            raise ValueError(f"Requested tokens not found in {args.split}: {missing}")
        tokens = args.tokens
    else:
        tokens = list(loader.tokens)[: args.max_scenes]
    first_agent.cpu()

    all_results: Dict[str, Dict[str, Dict]] = {}
    for model_name in DEFAULT_MODELS:
        print(f"Loading and running {model_name}")
        agent = first_agent if model_name == "baseline_pretrained" else _compose_agent(model_name)
        all_results[model_name] = _run_model_on_tokens(agent, loader, tokens, device)
        if model_name == "baseline_pretrained":
            first_agent = None
        del agent

    colors = {
        "baseline_pretrained": "#d62728",
        "bev_scorer_finetune": "#1f77b4",
        "bev_decoder_finetune_last": "#2ca02c",
    }
    saved = []
    for token in tqdm(tokens, desc="Saving figures"):
        scene = loader.get_scene_from_token(token)
        per_model = {model_name: all_results[model_name][token] for model_name in DEFAULT_MODELS}
        for model_name, result in per_model.items():
            saved.append(str(_save_single(scene, token, model_name, result, args.output_dir, colors[model_name])))
        saved.append(str(_save_comparison(scene, token, per_model, args.output_dir)))

    manifest = {
        "split": args.split,
        "tokens": tokens,
        "models": {
            name: {"checkpoint": str(spec["checkpoint"]), "overrides": spec["overrides"]}
            for name, spec in DEFAULT_MODELS.items()
        },
        "files": saved,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(saved)} PNG files")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
