#!/usr/bin/env python3
"""Render a one-page group-meeting summary for idea_0002_01."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = ROOT / "docs/figures/idea0002_01/proposal_world_architecture.png"
OUTPUT = ROOT / "docs/slides/idea0002_01_group_meeting_summary.png"

BG = "#F7F8FA"
INK = "#172026"
MUTED = "#5E6A72"
LINE = "#D8DEE3"
TEAL = "#2E7D8F"
ORANGE = "#C95D3A"
YELLOW = "#F2CC60"
GREEN = "#3B7D5A"
PALE_TEAL = "#E4F0F3"
PALE_ORANGE = "#F8E9E3"
PALE_YELLOW = "#FBF3D7"


def add_box(ax, x, y, w, h, facecolor="white", edgecolor=LINE, radius=0.012, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, size=18, color=INK, weight="normal", ha="left", va="top", width=None, linespacing=1.2):
    if width is not None:
        text = "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        family="DejaVu Sans",
    )


def add_section_title(ax, x, y, number, title, color):
    ax.add_patch(Rectangle((x, y - 0.025), 0.006, 0.040, transform=ax.transAxes, color=color, linewidth=0))
    add_text(ax, x + 0.015, y, f"{number}  {title}", size=19, weight="bold", va="center")


def draw_slide():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(BG)

    # Header
    ax.add_patch(Rectangle((0, 0.91), 1, 0.09, transform=ax.transAxes, color=INK, linewidth=0))
    add_text(ax, 0.045, 0.958, "Proposal-Conditioned BEV World Modeling", size=30, color="white", weight="bold", va="center")
    add_text(ax, 0.955, 0.958, "idea_0002_01  |  Fast validation", size=15, color="#C9D2D8", ha="right", va="center")
    add_text(ax, 0.045, 0.905, "Can candidate-specific imagined futures improve frozen trajectory refinement and scoring?", size=15, color=MUTED, va="top")

    # Motivation panel
    add_box(ax, 0.035, 0.585, 0.275, 0.285, facecolor="white")
    add_section_title(ax, 0.052, 0.835, "1", "Motivation", TEAL)
    add_text(
        ax, 0.055, 0.785,
        "Current BEV captures only the observed state; it cannot represent how each trajectory proposal changes future interactions.",
        size=15, width=38, linespacing=1.18,
    )
    add_text(ax, 0.055, 0.665, "HYPOTHESIS", size=11.5, color=ORANGE, weight="bold")
    add_text(
        ax, 0.055, 0.638,
        "Imagine 64 proposal-conditioned futures\nfor joint refinement and scoring.",
        size=14.5, weight="bold", linespacing=1.14,
    )

    # Architecture panel
    add_box(ax, 0.325, 0.585, 0.64, 0.285, facecolor="white")
    add_section_title(ax, 0.342, 0.835, "2", "Implementation: WoTE rollout + ResWorld refinement", ORANGE)
    architecture = Image.open(ARCHITECTURE).convert("RGB")
    width, height = architecture.size
    architecture = architecture.crop((35, 155, width - 30, height - 95))
    image_ax = fig.add_axes([0.345, 0.605, 0.596, 0.205])
    image_ax.imshow(architecture)
    image_ax.set_axis_off()
    add_text(
        ax, 0.645, 0.590,
        "Frozen encoder/decoder/scorer  |  Train BEV tokenizer + Transformer refiner  |  6.09M trainable (12.98%)",
        size=11.5, color=MUTED, ha="center", va="bottom",
    )

    # Experiment panel
    add_box(ax, 0.035, 0.245, 0.275, 0.315, facecolor="white")
    add_section_title(ax, 0.052, 0.525, "3", "Controlled training", GREEN)
    add_text(ax, 0.055, 0.483, "Matched comparison", size=14, color=GREEN, weight="bold")
    rows = [
        ("Subset", "2,048 scenes: 1,722 / 326"),
        ("Schedule", "4 epochs / 64 optimizer steps"),
        ("Compute", "bf16, batch 4, accum. 4"),
        ("Controls", "same tokens, seed, LR, checkpoint"),
        ("Gates", "both 0-init; params within +0.33%"),
    ]
    y = 0.445
    for label, value in rows:
        add_text(ax, 0.055, y, label, size=13, color=MUTED, weight="bold")
        add_text(ax, 0.145, y, value, size=13.5, color=INK)
        y -= 0.042

    # Results panel
    add_box(ax, 0.325, 0.245, 0.385, 0.315, facecolor="white")
    add_section_title(ax, 0.342, 0.525, "4", "Results", ORANGE)
    add_text(ax, 0.525, 0.485, "Static BEV", size=12, color=TEAL, weight="bold", ha="center")
    add_text(ax, 0.635, 0.485, "Proposal world", size=12, color=ORANGE, weight="bold", ha="center")
    metric_rows = [
        ("Val score  ↑", "0.952299", "0.952299"),
        ("Val L2  ↓", "0.435860", "0.435860"),
        ("Top-5 hit rate  ↑", "0.3594", "0.3594"),
        ("Train traj. loss  ↓", "0.620", "0.506"),
    ]
    y = 0.446
    for idx, (label, static, world) in enumerate(metric_rows):
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.342, y - 0.022), 0.349, 0.037, transform=ax.transAxes, color="#F3F5F6", linewidth=0))
        add_text(ax, 0.350, y, label, size=13.5, va="center")
        add_text(ax, 0.525, y, static, size=13.5, color=TEAL, weight="bold", ha="center", va="center")
        add_text(ax, 0.635, y, world, size=13.5, color=ORANGE, weight="bold", ha="center", va="center")
        y -= 0.043

    add_text(ax, 0.350, 0.288, "A100 module latency", size=11, color=MUTED, weight="bold", va="center")
    # Latency bars, scaled to 10 ms.
    ax.add_patch(Rectangle((0.475, 0.272), 0.019, 0.014, transform=ax.transAxes, color=TEAL, linewidth=0))
    ax.add_patch(Rectangle((0.475, 0.250), 0.195, 0.014, transform=ax.transAxes, color=ORANGE, linewidth=0))
    add_text(ax, 0.500, 0.279, "0.97 ms", size=10.2, color=TEAL, weight="bold", va="center")
    add_text(ax, 0.665, 0.257, "9.66 ms (9.95x)", size=9.6, color="white", weight="bold", ha="right", va="center")

    # Conclusion panel
    add_box(ax, 0.725, 0.245, 0.24, 0.315, facecolor=PALE_ORANGE, edgecolor="#E9C1B3")
    add_section_title(ax, 0.742, 0.525, "5", "Conclusion", ORANGE)
    add_text(ax, 0.747, 0.477, "NO VALIDATION GAIN", size=19, color=ORANGE, weight="bold")
    add_text(
        ax, 0.747, 0.425,
        "Gates stayed near 10^-6: future-BEV residual paths remained closed.",
        size=14, width=29, linespacing=1.15,
    )
    add_text(
        ax, 0.747, 0.350,
        "Lower train loss (0.506 vs 0.620) did not transfer to score, L2, or hit rate.",
        size=14, width=29, linespacing=1.15,
    )
    add_text(ax, 0.747, 0.282, "Reject the 0-init setup; the architecture remains untested.", size=11.8, color=INK, weight="bold", width=32, linespacing=1.05)

    # Next-step band
    add_box(ax, 0.035, 0.065, 0.93, 0.145, facecolor=INK, edgecolor=INK)
    add_text(ax, 0.055, 0.178, "NEXT STEP", size=13, color=YELLOW, weight="bold")
    steps = [
        ("1", "Open the gate", "Set both gates to 0.01;\nhold LR and data fixed."),
        ("2", "Test signal", "Require validation separation\nacross two seeds."),
        ("3", "Go / no-go", "If positive, run PDMS/EPDMS;\nelse deprioritize."),
    ]
    x_positions = [0.055, 0.365, 0.675]
    for x, (number, heading, detail) in zip(x_positions, steps):
        ax.add_patch(FancyBboxPatch((x, 0.092), 0.032, 0.050, boxstyle="round,pad=0.004,rounding_size=0.008", transform=ax.transAxes, facecolor=YELLOW, edgecolor="none"))
        add_text(ax, x + 0.016, 0.117, number, size=16, color=INK, weight="bold", ha="center", va="center")
        add_text(ax, x + 0.043, 0.135, heading, size=15, color="white", weight="bold", va="top")
        add_text(ax, x + 0.043, 0.108, detail, size=10.5, color="#CFD8DE", va="top", linespacing=1.12)

    add_text(ax, 0.045, 0.028, "Source-grounded by WoTE (arXiv:2504.01941) + ResWorld official implementation", size=10.5, color=MUTED, va="center")
    add_text(ax, 0.955, 0.028, "NAVSIM small-data screen  |  Aug 18, 2026", size=10.5, color=MUTED, ha="right", va="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=BG)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    draw_slide()
