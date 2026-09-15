#!/usr/bin/env python
"""Render NAVSIM-v2 baseline/proposal-world disagreement cases side by side."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import default_collate


WORKSPACE = Path("/mnt/ws-frb/users/jingyuso/wenzhet")
DEFAULT_NAVSIM_ROOT = Path("/tmp/navsim-v2-proposal-world")
DEFAULT_REFERENCE_ROOT = WORKSPACE / "DrivoR"
DEFAULT_BASELINE_CSV = (
    WORKSPACE
    / "navsim/exp/drivoR_nav2_full/2026.06.03.17.36.37/2026.06.03.19.19.49.csv"
)
DEFAULT_PROPOSAL_CSV = (
    WORKSPACE
    / "navsim/exp/drivoR_nav2-idea0002-01-proposal-world-best-epoch3"
    / "2026.09.11.05.12.46/2026.09.11.08.09.36.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs/figures/idea0002_01/navsim_v2_disagreements"
)
DEFAULT_TOKENS = [
    "fd7743059db2ad01f",  # baseline collision/TTC failure, proposal success
    "cac2753c7c7b93847",  # baseline DAC failure, proposal success
    "e422b7a7bb9c41f95",  # baseline success, proposal collision/TTC failure
    "cf516d2c4d97e69a8",  # baseline success, proposal DAC failure
]


METRICS = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC",
    "traffic_light_compliance": "TLC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "lane_keeping": "LK",
    "history_comfort": "HC",
    "two_frame_extended_comfort": "EC",
}


def _install_navsim_imports(navsim_root: Path, reference_root: Path) -> None:
    for path in (navsim_root, reference_root / "nuplan-devkit"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _set_environment(navsim_root: Path) -> None:
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(WORKSPACE / "navsim_dataset/maps"))
    os.environ.setdefault("NAVSIM_EXP_ROOT", str(WORKSPACE / "navsim/exp"))
    os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(navsim_root))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(WORKSPACE / "navsim_dataset"))
    os.environ.setdefault(
        "BEV_FEATURES_ROOT",
        str(WORKSPACE / "navsim_bev_feature/exports_pretrained_navsim_v2"),
    )


def _load_scores(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="") as file:
        return {
            row["token"]: row
            for row in csv.DictReader(file)
            if not row["token"].startswith("extended_pdm_score_")
        }


def _metric_summary(row: Mapping[str, str]) -> Dict[str, object]:
    stage = next(
        stage
        for stage in ("one", "two")
        if any(row.get(f"{metric}_stage_{stage}", "") for metric in METRICS)
    )
    values = {
        short: float(row[f"{metric}_stage_{stage}"])
        for metric, short in METRICS.items()
    }
    failed = [name for name, value in values.items() if value == 0.0]
    return {
        "stage": 1 if stage == "one" else 2,
        "score": float(row["score"]),
        "failed": failed,
        "values": values,
    }


def _common_agent_overrides(reference_root: Path) -> List[str]:
    weights = reference_root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
    return [
        "config.proposal_num=64",
        "config.refiner_ls_values=0.0",
        f"config.image_backbone.model_weights={weights}",
        "config.image_backbone.focus_front_cam=false",
        f"config.lidar_backbone.model_weights={weights}",
        "config.one_token_per_traj=true",
        "config.refiner_num_heads=1",
        "config.tf_d_model=256",
        "config.tf_d_ffn=1024",
        "config.area_pred=false",
        "config.agent_pred=false",
        "config.ref_num=4",
        "config.noc=10",
        "config.dac=13",
        "config.ddc=6",
        "config.ttc=14",
        "config.ep=15",
        "config.comfort=2",
        "config.use_ray_score=true",
        "config.long_trajectory_additional_poses=2",
        "lr_args=null",
        "scheduler_args=null",
        "loss=null",
        "batch_size=null",
        "num_gpus=1",
        "progress_bar=false",
    ]


def _compose_agent(
    navsim_root: Path,
    reference_root: Path,
    model_name: str,
    checkpoint: Path,
):
    overrides = [
        f"checkpoint_path='{checkpoint}'",
        *_common_agent_overrides(reference_root),
    ]
    if model_name == "baseline":
        overrides.extend(
            [
                "config.use_bev_feature=false",
                "config.use_bev_in_scorer=false",
                "config.use_bev_in_decoder=false",
                "config.use_bev_residual_proposal_refiner=false",
                "config.use_proposal_world_refiner=false",
                "config.use_privileged_future_bev=false",
            ]
        )
    else:
        overrides.extend(
            [
                "config.use_bev_feature=true",
                "config.use_bev_in_scorer=false",
                "config.use_bev_in_decoder=false",
                "config.use_bev_residual_proposal_refiner=false",
                "config.use_proposal_world_refiner=true",
                "config.use_privileged_future_bev=false",
                "config.bev_feature_type=decoder_neck",
                "config.bev_channels=256",
                f"config.bev_features_root={os.environ['BEV_FEATURES_ROOT']}",
                "config.bev_data_split=navhard_two_stage",
                "config.proposal_world_refiner.num_layers=2",
                "config.proposal_world_refiner.num_heads=4",
                "config.proposal_world_refiner.ffn_dim=512",
                "config.proposal_world_refiner.rollout_steps=1",
                "config.proposal_world_refiner.proposal_chunk_size=8",
                "config.proposal_world_refiner.init_refine_gate=0.01",
                "config.proposal_world_refiner.init_score_gate=0.01",
            ]
        )

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(navsim_root / "navsim/planning/script/config/common/agent"),
    ):
        cfg = compose(config_name="drivoR", overrides=overrides)
    agent = instantiate(cfg)
    agent.initialize()
    agent.eval()
    return agent


def _compose_eval_cfg(navsim_root: Path) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(navsim_root / "navsim/planning/script/config/pdm_scoring"),
    ):
        return compose(
            config_name="default_run_pdm_score",
            overrides=["train_test_split=navhard_two_stage"],
        )


def _load_scene_loader(navsim_root: Path, agent, tokens: List[str]):
    from navsim.common.dataloader import SceneLoader

    cfg = _compose_eval_cfg(navsim_root)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    stage_one_tokens = set(scene_filter.tokens or [])
    scene_filter.tokens = [token for token in tokens if token in stage_one_tokens]
    scene_filter.synthetic_scene_tokens = [token for token in tokens if token not in stage_one_tokens]
    scene_filter.max_scenes = None
    return SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _to_device(features: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in features.items()
    }


def _infer(agent, loader, tokens: Iterable[str], device: torch.device) -> Dict[str, Dict[str, object]]:
    builder = agent.get_feature_builders()[0]
    agent.to(device)
    outputs: Dict[str, Dict[str, object]] = {}
    with torch.no_grad():
        for token in tokens:
            scene = loader.get_scene_from_token(token)
            features = builder.compute_features(
                loader.get_agent_input_from_token(token),
                scene_token=scene.scene_metadata.initial_token,
                log_name=scene.scene_metadata.log_name,
            )
            prediction = agent.forward(_to_device(default_collate([features]), device))
            scores = prediction.get("pdm_score")
            outputs[token] = {
                "proposals": prediction["proposals"][0].detach().cpu(),
                "trajectory": prediction["trajectory"][0].detach().cpu(),
                "selected_index": (
                    int(torch.argmax(scores[0]).item()) if scores is not None else None
                ),
            }
    agent.cpu()
    torch.cuda.empty_cache()
    return outputs


def _trajectory_config(color: str, alpha: float, width: float, zorder: int) -> Dict[str, object]:
    return {
        "fill_color": color,
        "fill_color_alpha": alpha,
        "line_color": color,
        "line_color_alpha": alpha,
        "line_width": width,
        "line_style": "-",
        "marker": ".",
        "marker_size": 2.5 if width < 2 else 4.5,
        "marker_edge_color": color,
        "zorder": zorder,
    }


def _draw_panel(ax, scene, prediction, metrics, model_label: str, color: str) -> None:
    from navsim.common.dataclasses import Trajectory
    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.config import TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    for proposal in prediction["proposals"]:
        add_trajectory_to_bev_ax(
            ax,
            Trajectory(proposal.numpy()),
            _trajectory_config("#64748b", 0.17, 0.75, 3),
        )
    add_trajectory_to_bev_ax(
        ax,
        Trajectory(prediction["trajectory"].numpy()),
        _trajectory_config(color, 1.0, 3.0, 8),
    )
    try:
        human_cfg = dict(TRAJECTORY_CONFIG["human"])
        human_cfg.update({"line_style": "--", "line_width": 2.2, "zorder": 7})
        add_trajectory_to_bev_ax(ax, scene.get_future_trajectory(), human_cfg)
    except Exception:
        pass

    failed = metrics["failed"]
    status = "SUCCESS" if not failed else "FAIL"
    failure_text = "none" if not failed else ", ".join(failed)
    selected = prediction["selected_index"]
    selected_text = "?" if selected is None else str(selected)
    ax.set_title(
        f"{model_label}: {status} | EPDMS {metrics['score']:.3f}\n"
        f"failed: {failure_text} | selected proposal: {selected_text}",
        fontsize=11,
        fontweight="bold",
        color="#166534" if status == "SUCCESS" else "#b91c1c",
        pad=9,
    )
    configure_bev_ax(ax)
    configure_ax(ax)


def _render_scene(
    token: str,
    scene,
    predictions: Mapping[str, Mapping[str, object]],
    metrics: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> Path:
    baseline_wins = metrics["baseline"]["score"] > metrics["proposal"]["score"]
    direction = "baseline_success_proposal_fail" if baseline_wins else "baseline_fail_proposal_success"
    stage = metrics["baseline"]["stage"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    _draw_panel(
        axes[0], scene, predictions["baseline"], metrics["baseline"], "Pretrained baseline", "#dc2626"
    )
    _draw_panel(
        axes[1],
        scene,
        predictions["proposal"],
        metrics["proposal"],
        "Proposal-conditioned BEV",
        "#2563eb",
    )
    fig.suptitle(
        f"NAVSIM v2 Stage {stage} disagreement | scene {token}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.13, top=0.82, wspace=0.08)
    fig.text(
        0.5,
        0.025,
        "gray: all 64 proposals   red/blue: selected trajectory   green dashed: logged human future\n"
        "NC: no-at-fault collision   DAC: drivable-area compliance   TTC: time-to-collision",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    target = output_dir / direction / f"{token}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def _render_contact_sheets(paths: List[Path], output_dir: Path, page_size: int = 4) -> List[Path]:
    from PIL import Image, ImageDraw, ImageFont

    targets = []
    pages = [paths[start : start + page_size] for start in range(0, len(paths), page_size)]
    for page_index, page_paths in enumerate(pages, start=1):
        images = [Image.open(path).convert("RGB") for path in page_paths]
        target_width = min(1800, max(image.width for image in images))
        resized = []
        for image in images:
            height = round(image.height * target_width / image.width)
            resized.append(image.resize((target_width, height)))
        gap = 24
        header_height = 76
        canvas = Image.new(
            "RGB",
            (
                target_width,
                header_height + sum(image.height for image in resized) + gap * (len(resized) - 1),
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=26)
        page_suffix = f" | page {page_index}/{len(pages)}" if len(pages) > 1 else ""
        draw.text(
            (target_width // 2, 24),
            "NAVSIM v2 qualitative reversals: baseline vs proposal-conditioned BEV" + page_suffix,
            fill="#0f172a",
            font=font,
            anchor="ma",
        )
        y = header_height
        for image in resized:
            canvas.paste(image, (0, y))
            y += image.height + gap
        filename = (
            f"navsim_v2_disagreement_contact_sheet_page_{page_index}.png"
            if len(pages) > 1
            else "navsim_v2_disagreement_contact_sheet.png"
        )
        target = output_dir / filename
        canvas.save(target)
        targets.append(target)
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-root", type=Path, default=DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--proposal-csv", type=Path, default=DEFAULT_PROPOSAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokens", nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_environment(args.navsim_root)
    _install_navsim_imports(args.navsim_root, args.reference_root)

    baseline_scores = _load_scores(args.baseline_csv)
    proposal_scores = _load_scores(args.proposal_csv)
    missing = [
        token
        for token in args.tokens
        if token not in baseline_scores or token not in proposal_scores
    ]
    if missing:
        raise KeyError(f"Tokens missing from evaluation CSVs: {missing}")

    baseline_checkpoint = args.reference_root / "weights/checkpoints/drivor_Nav2_10epochs.pth"
    proposal_checkpoint = (
        args.reference_root
        / "exp/ke/Aug18-idea0002-01-proposal-world-full-8gpu-gates001"
        / "08.18_8gpu_full_gates001_schedfix_recache/checkpoints/best-epoch=3-step=7844.ckpt"
    )
    baseline_agent = _compose_agent(
        args.navsim_root, args.reference_root, "baseline", baseline_checkpoint
    )
    loader = _load_scene_loader(args.navsim_root, baseline_agent, args.tokens)
    unavailable = [token for token in args.tokens if token not in set(loader.tokens)]
    if unavailable:
        raise KeyError(f"Tokens unavailable from NAVSIM-v2 scene loader: {unavailable}")

    device = torch.device(args.device)
    predictions = {
        "baseline": _infer(baseline_agent, loader, args.tokens, device),
    }
    del baseline_agent
    proposal_agent = _compose_agent(
        args.navsim_root, args.reference_root, "proposal", proposal_checkpoint
    )
    predictions["proposal"] = _infer(proposal_agent, loader, args.tokens, device)
    del proposal_agent

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = {
        "baseline_checkpoint": str(baseline_checkpoint),
        "proposal_checkpoint": str(proposal_checkpoint),
        "baseline_csv": str(args.baseline_csv),
        "proposal_csv": str(args.proposal_csv),
        "scenes": {},
    }
    for token in args.tokens:
        scene_metrics = {
            "baseline": _metric_summary(baseline_scores[token]),
            "proposal": _metric_summary(proposal_scores[token]),
        }
        scene = loader.get_scene_from_token(token)
        path = _render_scene(
            token,
            scene,
            {name: model_predictions[token] for name, model_predictions in predictions.items()},
            scene_metrics,
            args.output_dir,
        )
        paths.append(path)
        manifest["scenes"][token] = {
            "metrics": scene_metrics,
            "selected_proposals": {
                name: predictions[name][token]["selected_index"] for name in predictions
            },
            "figure": str(path),
        }

    contact_sheets = _render_contact_sheets(paths, args.output_dir)
    manifest["contact_sheets"] = [str(path) for path in contact_sheets]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(paths)} scene comparisons")
    for contact_sheet in contact_sheets:
        print(f"Contact sheet: {contact_sheet}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
