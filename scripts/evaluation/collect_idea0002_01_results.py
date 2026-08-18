#!/usr/bin/env python3
"""Collect paired W&B runs and render idea_0002_01 training curves."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import wandb


METRICS = [
    "train/loss_epoch",
    "train/trajectory_loss",
    "train/proposal_world_refine_gate",
    "train/proposal_world_score_gate",
    "train/residual_alpha",
    "val/score_epoch",
    "val/l2",
    "val/score_hit_rate",
    "val/top_5_score_hit_rate",
]


def latest_matching_run(api, project_path: str, prefix: str):
    runs = list(
        api.runs(
            project_path,
            filters={"display_name": {"$regex": f"^{prefix}/"}},
            order="-created_at",
        )
    )
    if not runs:
        raise RuntimeError(f"No W&B run found with display-name prefix {prefix!r}")
    return runs[0]


def aggregate_history(rows: Iterable[Dict]) -> List[Dict]:
    by_epoch: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        epoch_value = row.get("epoch")
        if epoch_value is None:
            continue
        epoch = int(epoch_value)
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                by_epoch[epoch][metric].append(float(value))

    result = []
    for epoch, values in sorted(by_epoch.items()):
        record = {"epoch": epoch}
        for metric, samples in values.items():
            if metric.startswith("train/"):
                record[metric] = sum(samples) / len(samples)
            else:
                record[metric] = samples[-1]
        result.append(record)
    return result


def summarize_run(run, label: str) -> Dict:
    rows = aggregate_history(run.scan_history(page_size=1000))
    val_rows = [row for row in rows if "val/score_epoch" in row]
    best_row = max(val_rows, key=lambda row: row["val/score_epoch"]) if val_rows else {}
    return {
        "label": label,
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "epochs": rows,
        "best_epoch": best_row.get("epoch"),
        "best_val_score": best_row.get("val/score_epoch"),
    }


def write_epoch_csv(path: Path, records: Dict[str, Dict]) -> None:
    fieldnames = ["variant", "epoch", *METRICS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label, record in records.items():
            for row in record["epochs"]:
                writer.writerow({"variant": label, **row})


def plot_curves(path: Path, records: Dict[str, Dict]) -> None:
    colors = {"static_bev_refiner": "#3B82A0", "proposal_world": "#C65D3A"}
    panels = [
        ("val/score_epoch", "Validation score", "Higher is better"),
        ("val/l2", "Validation L2", "Lower is better"),
        ("train/trajectory_loss", "Trajectory loss", "Epoch mean"),
        ("train/proposal_world_refine_gate", "Residual gate", "Gate evolution"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (metric, title, subtitle) in zip(axes.flat, panels):
        for label, record in records.items():
            points = [row for row in record["epochs"] if metric in row]
            if not points and metric == "train/proposal_world_refine_gate":
                fallback = "train/residual_alpha"
                points = [row for row in record["epochs"] if fallback in row]
                values = [row[fallback] for row in points]
            else:
                values = [row[metric] for row in points]
            if points:
                axis.plot(
                    [row["epoch"] for row in points],
                    values,
                    marker="o",
                    linewidth=2,
                    label=label.replace("_", " "),
                    color=colors.get(label),
                )
        axis.set_title(title)
        axis.text(0.0, 1.01, subtitle, transform=axis.transAxes, fontsize=8, color="#666666")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("idea_0002_01 Matched Small-Data Validation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity")
    parser.add_argument("--project", default="drivor-world-model-fast-validation")
    parser.add_argument("--static-run")
    parser.add_argument("--world-run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/idea0002_01_results"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/idea0002_01/training_curves.png"),
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        raise RuntimeError("W&B entity is unavailable; pass --entity")
    project_path = f"{entity}/{args.project}"

    static_run = (
        api.run(f"{project_path}/{args.static_run}")
        if args.static_run
        else latest_matching_run(api, project_path, "Aug18-idea0002-01-static-bev-refiner-fast")
    )
    world_run = (
        api.run(f"{project_path}/{args.world_run}")
        if args.world_run
        else latest_matching_run(api, project_path, "Aug18-idea0002-01-proposal-world-fast")
    )
    records = {
        "static_bev_refiner": summarize_run(static_run, "static_bev_refiner"),
        "proposal_world": summarize_run(world_run, "proposal_world"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paired_results.json"
    csv_path = args.output_dir / "epoch_metrics.csv"
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True, default=str) + "\n")
    write_epoch_csv(csv_path, records)
    plot_curves(args.figure, records)
    print(json.dumps({label: {"run": item["name"], "state": item["state"], "best_epoch": item["best_epoch"], "best_val_score": item["best_val_score"]} for label, item in records.items()}, indent=2))


if __name__ == "__main__":
    main()
