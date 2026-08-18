#!/usr/bin/env python3
"""Plot structural and training evidence for the idea_0002_01 screen."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("docs/experiments/idea0002_01_structural_benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/idea0002_01/structural_tradeoff.png"),
    )
    args = parser.parse_args()

    data = json.loads(args.benchmark.read_text())
    variants = data["variants"]
    labels = ["Static BEV\nrefiner", "Proposal-conditioned\nworld refiner"]
    keys = ["static_bev_refiner", "proposal_world"]
    colors = ["#3B82A0", "#C65D3A"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    parameters = [variants[key]["parameters"] / 1e6 for key in keys]
    latency = [variants[key]["mean_latency_ms"] for key in keys]

    bars = axes[0].bar(labels, parameters, color=colors, width=0.62)
    axes[0].set_ylabel("Module parameters (M)")
    axes[0].set_title("Matched Capacity")
    axes[0].set_ylim(0, max(parameters) * 1.25)
    axes[0].bar_label(bars, fmt="%.3fM", padding=3)

    bars = axes[1].bar(labels, latency, color=colors, width=0.62)
    axes[1].set_ylabel("CPU latency (ms), batch 1")
    axes[1].set_title("Screening-Time Compute Cost")
    axes[1].set_ylim(0, max(latency) * 1.18)
    axes[1].bar_label(bars, fmt="%.2f ms", padding=3)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)

    fig.suptitle("idea_0002_01 Structural Trade-off", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Module-only CPU benchmark; A100 latency and planning quality pending paired training.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
