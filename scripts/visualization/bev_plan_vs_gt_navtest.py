#!/usr/bin/env python3
"""
NAVSIM v1: BEV visualization — model plan vs human GT using navsim.visualization.plot_bev_with_agent.

Requires env: OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION (same as PDMS eval).
Run from repo root so relative paths like weights/ resolve.

Example (failures, bottom 20 by score):
  python scripts/visualization/bev_plan_vs_gt_navtest.py \\
    --csv exp/ke/drivoR_nav1_eval/04.01_17.16/2026.04.02.02.35.44.csv \\
    --out_dir exp/viz_navtest_plan_vs_gt/run1 \\
    --checkpoint weights/drivor_Nav1_25epochs.pth \\
    --n_worst 20

Example (success sample):
  python scripts/visualization/bev_plan_vs_gt_navtest.py --mode success --n_success 10 --min_score 0.99 ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_with_agent, plot_cameras_frame_with_annotations

logger = logging.getLogger(__name__)

# Same optional overrides as scripts/evaluation/run_drivor_nav1_pdms.sh (for parity with PDMS eval)
EVAL_LIKE_OVERRIDES: Sequence[str] = (
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
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _compose_navtest_scene_filter() -> Tuple[Any, str]:
    GlobalHydra.instance().clear()
    split_dir = _repo_root() / "navsim/planning/script/config/common/train_test_split"
    with initialize_config_dir(version_base=None, config_dir=str(split_dir)):
        cfg = compose(config_name="navtest")
    data_split = str(cfg.data_split)
    scene_filter = instantiate(cfg.scene_filter)
    return scene_filter, data_split


def _compose_drivor_agent(checkpoint: Path, eval_like: bool) -> Any:
    GlobalHydra.instance().clear()
    agent_dir = _repo_root() / "navsim/planning/script/config/common/agent"
    overrides: List[str] = [
        f"checkpoint_path={checkpoint.resolve()}",
        "scheduler_args.num_epochs=1",
        "scheduler_args.dataset_size=85000",
        "batch_size=64",
    ]
    if eval_like:
        overrides.extend(EVAL_LIKE_OVERRIDES)
    with initialize_config_dir(version_base=None, config_dir=str(agent_dir)):
        cfg = compose(config_name="drivoR", overrides=overrides)
    return instantiate(cfg)


def _parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def select_tokens_failure(
    df: pd.DataFrame,
    n_worst: int,
    bottom_percentile: Optional[float],
    worst_metric: Optional[str],
    worst_metric_k: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (token, meta) for phase-1 failure / weak-score inspection."""
    if "token" not in df.columns or "score" not in df.columns:
        raise ValueError("CSV must contain columns: token, score")

    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "valid" in df.columns:
        df["_valid"] = _parse_bool_series(df["valid"])
    else:
        df["_valid"] = True

    out: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()

    invalid = df[~df["_valid"]]
    for _, row in invalid.iterrows():
        t = str(row["token"])
        if t in seen:
            continue
        seen.add(t)
        out.append(
            (
                t,
                {
                    "selection_reason": "invalid_valid_false",
                    "score": float(row["score"]) if pd.notna(row["score"]) else None,
                    "valid": False,
                },
            )
        )

    df_valid = df[df["_valid"]].dropna(subset=["score"]).sort_values("score", ascending=True)
    if bottom_percentile is not None:
        thr = df_valid["score"].quantile(bottom_percentile / 100.0)
        sub = df_valid[df_valid["score"] <= thr]
    else:
        sub = df_valid

    n_take = max(0, n_worst - len(out))
    for _, row in sub.head(n_take).iterrows():
        t = str(row["token"])
        if t in seen:
            continue
        seen.add(t)
        reason = f"bottom{bottom_percentile}pct_score" if bottom_percentile is not None else "bottom_n_by_score"
        meta = {"selection_reason": reason, "score": float(row["score"]), "valid": True}
        for c in df.columns:
            if c not in ("token",) and c not in meta:
                meta[c] = row[c]
        out.append((t, meta))

    if worst_metric and worst_metric in df.columns and worst_metric_k > 0:
        sub_m = df[df["_valid"]].sort_values(worst_metric, ascending=True).head(worst_metric_k)
        for _, row in sub_m.iterrows():
            t = str(row["token"])
            if t in seen:
                continue
            seen.add(t)
            meta = {
                "selection_reason": f"worst_{worst_metric}",
                "score": float(row["score"]) if pd.notna(row["score"]) else None,
                "valid": bool(row["_valid"]),
                worst_metric: float(row[worst_metric]) if pd.notna(row[worst_metric]) else None,
            }
            out.append((t, meta))

    return out


def select_tokens_success(
    df: pd.DataFrame, n_success: int, min_score: float, seed: int
) -> List[Tuple[str, Dict[str, Any]]]:
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    high = df[df["score"] >= min_score].dropna(subset=["token", "score"])
    tokens = high["token"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(tokens)
    pick = tokens[:n_success]
    out: List[Tuple[str, Dict[str, Any]]] = []
    score_map = dict(zip(df["token"].astype(str), df["score"]))
    for t in pick:
        out.append(
            (
                t,
                {
                    "selection_reason": "success_random_over_min_score",
                    "score": float(score_map.get(t, 0.0)),
                    "min_score_threshold": min_score,
                },
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NAVSIM v1 BEV: plan vs GT (plot_bev_with_agent)")
    parser.add_argument("--csv", type=Path, required=True, help="PDMS output CSV (navtest evaluation)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="DrivoR Nav1 checkpoint (.pth)")
    parser.add_argument(
        "--mode",
        choices=("failure", "success"),
        default="failure",
        help="failure: weak/low score tokens first; success: random sample above --min_score",
    )
    parser.add_argument("--n_worst", type=int, default=20, help="Max failure tokens by low score (after invalid)")
    parser.add_argument(
        "--bottom_percentile",
        type=float,
        default=None,
        help="If set, take tokens with score <= this percentile (e.g. 5 for worst 5%%)",
    )
    parser.add_argument("--worst_metric", type=str, default=None, help="Extra: take K worst by this column name")
    parser.add_argument("--worst_metric_k", type=int, default=0, help="How many extra for --worst_metric")
    parser.add_argument("--n_success", type=int, default=10, help="For --mode success: sample size")
    parser.add_argument("--min_score", type=float, default=0.99, help="For --mode success: minimum score")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for success sampling")
    parser.add_argument("--eval_like", action="store_true", help="Apply same agent config overrides as run_drivor_nav1_pdms.sh")
    parser.add_argument("--also_cameras", action="store_true", help="Also save plot_cameras_frame_with_annotations PNGs")
    parser.add_argument("--dry_run", action="store_true", help="Only write manifest, skip SceneLoader and rendering")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of tokens to render (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.environ.get("OPENSCENE_DATA_ROOT"):
        logger.error("OPENSCENE_DATA_ROOT is not set.")
        sys.exit(1)

    repo = _repo_root()
    os.chdir(repo)
    checkpoint = args.checkpoint or (repo / "weights/drivor_Nav1_25epochs.pth")
    if not checkpoint.is_file():
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.mode == "failure":
        selected = select_tokens_failure(
            df,
            n_worst=args.n_worst,
            bottom_percentile=args.bottom_percentile,
            worst_metric=args.worst_metric,
            worst_metric_k=args.worst_metric_k,
        )
    else:
        selected = select_tokens_success(df, n_success=args.n_success, min_score=args.min_score, seed=args.seed)

    if args.limit is not None:
        selected = selected[: args.limit]

    subdir = "failures" if args.mode == "failure" else "success_sample"
    out_root = args.out_dir.resolve() / subdir
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    manifest_rows = []
    for token, meta in selected:
        row = {"token": token, **{k: v for k, v in meta.items() if not k.startswith("_")}}
        manifest_rows.append(row)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)
    logger.info("Wrote manifest with %d entries: %s", len(manifest_rows), manifest_path)

    if args.dry_run:
        return

    scene_filter, data_split = _compose_navtest_scene_filter()
    tokens_wanted = [t for t, _ in selected]
    scene_filter.tokens = tokens_wanted

    agent = _compose_drivor_agent(checkpoint, eval_like=args.eval_like)
    agent.initialize()
    sensor_config = agent.get_sensor_config()

    data_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / data_split
    sensor_blobs_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "sensor_blobs" / data_split

    scene_loader = SceneLoader(
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    loadable = set(scene_loader.tokens)
    skipped = [t for t in tokens_wanted if t not in loadable]
    if skipped:
        logger.warning("Tokens not in SceneLoader (skipped): %s", skipped[:20] + (["..."] if len(skipped) > 20 else []))

    meta_by_token = {t: m for t, m in selected}
    n_ok = 0
    for token in tokens_wanted:
        if token not in loadable:
            continue
        meta = meta_by_token.get(token, {})
        score = meta.get("score", "")
        try:
            scene = scene_loader.get_scene_from_token(token)
            fig, ax = plot_bev_with_agent(scene, agent)
            title = f"token={token} score={score}"
            ax.set_title(title, fontsize=8)
            score_tag = f"{float(score):.4f}" if isinstance(score, (int, float)) else str(score).replace("/", "_")
            fname = f"bev_{token}_score_{score_tag}.png".replace("/", "_")
            fig.savefig(out_root / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if args.also_cameras:
                frame_idx = scene.scene_metadata.num_history_frames - 1
                fig_c, _ = plot_cameras_frame_with_annotations(scene, frame_idx)
                fig_c.savefig(
                    out_root / f"cameras_{token}.png".replace("/", "_"),
                    dpi=120,
                    bbox_inches="tight",
                )
                plt.close(fig_c)
            n_ok += 1
        except Exception as e:
            logger.exception("Failed token %s: %s", token, e)

    logger.info("Rendered %d BEV images under %s", n_ok, out_root)


if __name__ == "__main__":
    main()
