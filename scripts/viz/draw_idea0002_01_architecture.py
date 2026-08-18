#!/usr/bin/env python3
"""Render the proposal-conditioned BEV world-refinement architecture."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path("docs/figures/idea0002_01/proposal_world_architecture.png")


def box(axis, xy, width, height, text, face, edge="#222222", dashed=False):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
        linestyle="--" if dashed else "-",
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)
    return patch


def arrow(axis, start, end, color="#333333", style="-|>", curve=0.0, width=1.4):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def main() -> None:
    fig, axis = plt.subplots(figsize=(13.2, 5.4))
    axis.set_xlim(0, 13.2)
    axis.set_ylim(0, 5.4)
    axis.axis("off")

    frozen = "#E8EDF1"
    current = "#B8D8E8"
    future = "#F4B69C"
    trainable = "#F8D97B"

    box(axis, (0.3, 3.4), 1.6, 0.8, "Frozen scene\nencoder", frozen)
    box(axis, (2.3, 3.4), 1.8, 0.8, "Frozen trajectory\ndecoder", frozen)
    box(axis, (4.5, 3.4), 1.5, 0.8, "64 base\nproposals", frozen)
    arrow(axis, (1.9, 3.8), (2.3, 3.8))
    arrow(axis, (4.1, 3.8), (4.5, 3.8))

    box(axis, (0.3, 1.25), 1.6, 0.8, "Current BEV\nfeature map", current)
    box(axis, (2.3, 1.25), 1.8, 0.8, "Shared BEV\ntokenizer", trainable)
    box(axis, (4.5, 1.25), 1.5, 0.8, "64 current\nBEV tokens", current)
    arrow(axis, (1.9, 1.65), (2.3, 1.65))
    arrow(axis, (4.1, 1.65), (4.5, 1.65))

    box(axis, (6.45, 2.25), 2.1, 1.25, "Proposal-conditioned\nBEV Transformer\n(chunked 8 x 8)", trainable)
    arrow(axis, (6.0, 3.8), (6.45, 3.1), color="#C65D3A")
    arrow(axis, (6.0, 1.65), (6.45, 2.55), color="#3B82A0")

    box(axis, (9.0, 2.25), 1.8, 1.25, "64 candidate-\nspecific future\nBEV states", future)
    arrow(axis, (8.55, 2.87), (9.0, 2.87), color="#C65D3A")

    box(axis, (11.25, 3.5), 1.55, 0.85, "Gated delta\ntrajectory head", trainable)
    box(axis, (11.25, 1.3), 1.55, 0.85, "Gated scorer\ncontext adapter", trainable)
    arrow(axis, (10.8, 3.05), (11.25, 3.75), color="#C65D3A")
    arrow(axis, (10.8, 2.65), (11.25, 1.75), color="#C65D3A")
    arrow(axis, (6.0, 3.7), (11.25, 4.0), color="#777777", curve=-0.15)

    axis.text(12.03, 4.72, r"$\tau' = \tau + \alpha\,\Delta\tau$", ha="center", fontsize=11)
    axis.text(12.03, 0.82, "future-aware proposal scores", ha="center", fontsize=10)
    axis.text(7.5, 1.0, "Each proposal defines a different action token and therefore a different imagined future.", ha="center", fontsize=9, color="#555555")

    axis.text(0.3, 5.02, "idea_0002_01: Future-Aware Proposal Refinement", fontsize=15, weight="bold")
    axis.text(0.3, 4.68, "WoTE-style action-conditioned rollout + ResWorld-style future-guided refinement", fontsize=10, color="#555555")

    legend_x = 7.8
    for idx, (label, color) in enumerate(
        [("Frozen pretrained", frozen), ("Current state", current), ("Predicted future", future), ("Trainable new module", trainable)]
    ):
        x = legend_x + (idx % 2) * 2.4
        y = 0.35 - (idx // 2) * 0.35
        axis.add_patch(FancyBboxPatch((x, y), 0.22, 0.16, boxstyle="round,pad=0.01", facecolor=color, edgecolor="#555555"))
        axis.text(x + 0.3, y + 0.08, label, va="center", fontsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
