#!/usr/bin/env python3
"""Plot the verified experiment results recorded in docs/evaluation-ledger.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


COLORS = {
    "blue": "#2F6B9A",
    "green": "#2A9D78",
    "orange": "#E58C47",
    "red": "#C8553D",
    "yellow": "#E9C46A",
    "gray": "#73808C",
    "light_gray": "#D9E0E5",
    "ink": "#1F2933",
}

# Values are transcribed from the verified tables in docs/evaluation-ledger.md.
# Jun29 last/best are identical and are represented once to avoid double-counting.
RUNS = [
    {
        "id": "Jun12",
        "label": "Pretrained + current BEV decoder\nLoRA16",
        "short": "PT decoder L16",
        "v1": 0.932158,
        "v2_s1": 0.841329,
        "v2_s2": 0.584989,
        "v2": 0.496626,
        "val": 0.941252,
        "train_loss": 0.847087,
        "val_best": 0.975063,
        "val_lost": 0.033812,
        "val_hit": 0.117077,
        "val_top5": 0.375880,
        "val_l2": 0.398444,
        "val_collision": 0.997277,
        "val_dac": 0.991637,
        "val_progress": 0.882842,
        "val_ttc": 0.988721,
        "val_comfort": 0.999945,
        "train_h": 1 + 55 / 60 + 42 / 3600,
        "val_h": 11 / 60 + 55 / 3600,
        "v1_sub": [0.991314, 0.990697, 0.881224, 0.971760, 0.999835, 0.977976],
        "v2_s2_sub": [0.883254, 0.872065, 0.934825, 0.993842, 0.754583, 0.859066, 0.518615, 0.981594, 0.733703],
    },
    {
        "id": "Jun22",
        "label": "Pretrained + current BEV residual\nproposal refiner",
        "short": "PT residual refiner",
        "v1": 0.931695,
        "v2_s1": 0.842178,
        "v2_s2": 0.565484,
        "v2": 0.478820,
        "val": 0.941686,
        "train_loss": 0.867485,
        "val_best": 0.975324,
        "val_lost": 0.033638,
        "val_hit": 0.107724,
        "val_top5": 0.354643,
        "val_l2": 0.409636,
        "val_collision": 0.997414,
        "val_dac": 0.991637,
        "val_progress": 0.884166,
        "val_ttc": 0.988446,
        "val_comfort": 0.999780,
        "train_h": 1 + 3 / 60 + 36 / 3600,
        "val_h": 11 / 60 + 54 / 3600,
        "v1_sub": [0.991685, 0.989626, 0.882383, 0.970690, 1.000000, 0.977606],
        "v2_s2_sub": [0.869743, 0.859456, 0.927707, 0.991619, 0.767012, 0.851680, 0.518043, 0.979836, 0.691413],
    },
    {
        "id": "Jun29",
        "label": "Scratch + current BEV decoder\nLoRA16",
        "short": "Scratch decoder L16",
        "v1": 0.932017,
        "v2_s1": 0.836660,
        "v2_s2": 0.536875,
        "v2": 0.447110,
        "val": 0.958703,
        "train_loss": 1.251491,
        "val_best": 0.990338,
        "val_lost": 0.031635,
        "val_hit": 0.062995,
        "val_top5": 0.252421,
        "val_l2": 0.700153,
        "val_collision": 0.997634,
        "val_dac": 0.991637,
        "val_progress": 0.925446,
        "val_ttc": 0.988061,
        "val_comfort": 0.999945,
        "train_h": 2 + 33 / 60 + 54 / 3600,
        "val_h": 24 / 60 + 19 / 3600,
        "v1_sub": [0.986992, 0.985263, 0.906308, 0.955541, 0.999753, 0.971143],
        "v2_s2_sub": [0.872222, 0.844193, 0.908446, 0.989370, 0.766800, 0.854214, 0.496669, 0.973460, 0.620977],
    },
    {
        "id": "Jul05",
        "label": "Scratch + current BEV decoder\nLoRA8",
        "short": "Scratch decoder L8",
        "v1": 0.931384,
        "v2_s1": 0.834874,
        "v2_s2": 0.535531,
        "v2": 0.448434,
        "val": 0.946079,
        "train_loss": 1.354433,
        "val_best": 0.989947,
        "val_lost": 0.043868,
        "val_hit": 0.042749,
        "val_top5": 0.188160,
        "val_l2": 0.622069,
        "val_collision": 0.995213,
        "val_dac": 0.989437,
        "val_progress": 0.905152,
        "val_ttc": 0.982119,
        "val_comfort": 0.999835,
        "train_h": 2 + 31 / 60 + 25 / 3600,
        "val_h": 23 / 60 + 15 / 3600,
        "v1_sub": [0.990120, 0.985427, 0.893936, 0.965091, 0.999753, 0.967438],
        "v2_s2_sub": [0.856191, 0.858177, 0.906976, 0.992740, 0.736758, 0.838047, 0.508690, 0.980724, 0.717520],
    },
    {
        "id": "Jul08",
        "label": "Pretrained + current BEV decoder\n+ scorer LoRA16",
        "short": "PT decoder+scorer L16",
        "v1": 0.936007,
        "v2_s1": 0.847325,
        "v2_s2": 0.547436,
        "v2": 0.467831,
        "val": 0.951959,
        "train_loss": 0.952228,
        "val_best": 0.990435,
        "val_lost": 0.038476,
        "val_hit": 0.050121,
        "val_top5": 0.187280,
        "val_l2": 0.639529,
        "val_collision": 0.997112,
        "val_dac": 0.990042,
        "val_progress": 0.913017,
        "val_ttc": 0.986356,
        "val_comfort": 0.999945,
        "train_h": 53 / 60 + 56 / 3600,
        "val_h": 10 / 60 + 28 / 3600,
        "v1_sub": [0.990120, 0.988144, 0.901130, 0.965091, 1.000000, 0.973078],
        "v2_s2_sub": [0.855054, 0.844222, 0.919130, 0.990897, 0.793148, 0.836443, 0.516290, 0.977976, 0.671874],
    },
    {
        "id": "Jul14 e2",
        "label": "Pretrained + current/future BEV\ndecoder + scorer LoRA16 (epoch 2)",
        "short": "PT future decoder+scorer L16 e2",
        "v1": 0.933425,
        "v2_s1": 0.837930,
        "v2_s2": 0.546367,
        "v2": 0.461443,
        "val": 0.919751,
        "train_loss": 1.464653,
        "val_best": 0.968898,
        "val_lost": 0.049147,
        "val_hit": 0.032088,
        "val_top5": 0.149505,
        "val_l2": 0.620705,
        "val_collision": 0.991616,
        "val_dac": 0.972409,
        "val_progress": 0.877072,
        "val_ttc": 0.970160,
        "val_comfort": 0.999924,
        "train_h": 8 + 42 / 60 + 11 / 3600,
        "val_h": 39 / 60 + 53 / 3600,
        "v1_sub": [0.989997, 0.988556, 0.893291, 0.966656, 0.999835, 0.972254],
        "v2_s2_sub": [0.870881, 0.836706, 0.919632, 0.991429, 0.775803, 0.839679, 0.517425, 0.978009, 0.705899],
    },
    {
        "id": "Jul14 e6",
        "label": "Pretrained + current/future BEV\ndecoder + scorer LoRA16 (epoch 6)",
        "short": "PT future decoder+scorer L16 e6",
        "v1": 0.933163,
        "v2_s1": 0.849266,
        "v2_s2": 0.561461,
        "v2": 0.480632,
        "val": 0.920095,
        "train_loss": 1.446993,
        "val_best": 0.968434,
        "val_lost": 0.048339,
        "val_hit": 0.033155,
        "val_top5": 0.153544,
        "val_l2": 0.604394,
        "val_collision": 0.991521,
        "val_dac": 0.971608,
        "val_progress": 0.879482,
        "val_ttc": 0.969779,
        "val_comfort": 0.999962,
        "train_h": 14 + 31 / 60 + 50 / 3600,
        "val_h": 39 / 60 + 32 / 3600,
        "v1_sub": [0.989709, 0.987650, 0.894209, 0.966079, 0.999835, 0.973325],
        "v2_s2_sub": [0.870729, 0.856781, 0.915275, 0.988053, 0.796639, 0.846325, 0.529390, 0.972375, 0.674714],
    },
]

BASELINE = {
    "label": "Pretrained baseline",
    "v1": 0.936905,
    "v2_s1": 0.809321,
    "v2_s2": 0.594511,
    "v2": 0.483144,
    "v1_sub": [0.990367, 0.989297, 0.899420, 0.967150, 1.000000, 0.972542],
    "v2_s2_sub": [0.902004, 0.883532, 0.918618, 0.986224, 0.697924, 0.879503, 0.500725, 0.985213, 0.762156],
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB4BC",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": "#52606D",
            "ytick.color": "#52606D",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, digits: int = 3) -> None:
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.{digits}f}",
            va="center",
            fontsize=8,
            color=COLORS["ink"],
        )


def plot_overall(output_dir: Path) -> None:
    labels = [run["label"] for run in RUNS]
    y = np.arange(len(RUNS))
    metrics = [
        ("v1", "NAVSIM v1 PDMS", COLORS["blue"], (0.925, 0.940)),
        ("v2_s1", "NAVSIM v2 Stage 1", COLORS["green"], (0.800, 0.855)),
        ("v2", "NAVSIM v2 Combined EPDMS", COLORS["orange"], (0.435, 0.505)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), sharey=True)
    for ax, (key, title, color, limits) in zip(axes, metrics):
        values = [run[key] for run in RUNS]
        bars = ax.barh(y, values, color=color, height=0.62)
        annotate_bars(ax, bars, digits=4)
        ax.set_title(title, loc="left")
        ax.set_xlim(*limits)
        ax.grid(axis="x", color="#E7EBEE", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        baseline_value = BASELINE[key]
        ax.axvline(baseline_value, color=COLORS["red"], linestyle="--", linewidth=1.4)
        ax.set_xlabel(f"Dashed line: pretrained baseline = {baseline_value:.4f}", fontsize=8, color=COLORS["red"])
    axes[0].set_yticks(y, labels)
    fig.suptitle("DrivoR Official Evaluation Results", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.01, "Verified NAVSIM v1/v2 evaluations; Jun29 identical last/best results shown once.", fontsize=9, color="#52606D")
    fig.subplots_adjust(left=0.28, right=0.98, top=0.82, bottom=0.13, wspace=0.28)
    save_figure(fig, output_dir, "01_official_test_results")


def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, rows: list[str], columns: list[str], title: str) -> None:
    cmap = LinearSegmentedColormap.from_list("drivor", ["#F7E8D6", "#F8F8F5", "#A7D9C8", "#2A9D78"])
    image = ax.imshow(matrix, cmap=cmap, vmin=0.49, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title(title, loc="left", pad=12)
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value > 0.965 or value < 0.58 else COLORS["ink"]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7.5, color=text_color)
    return image


def plot_submetrics(output_dir: Path) -> None:
    run_rows = [run["label"] for run in RUNS]
    rows = ["Pretrained baseline"] + run_rows
    v1 = np.array([BASELINE["v1_sub"]] + [run["v1_sub"] for run in RUNS])
    v2_rows = ["Pretrained baseline"] + run_rows
    v2 = np.array([BASELINE["v2_s2_sub"]] + [run["v2_s2_sub"] for run in RUNS])
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.8), gridspec_kw={"height_ratios": [1, 1.15]})
    draw_heatmap(axes[0], v1, rows, ["NC", "DAC", "EP", "TTC", "Comfort", "DDC"], "NAVSIM v1 submetrics")
    image = draw_heatmap(
        axes[1],
        v2,
        v2_rows,
        ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"],
        "NAVSIM v2 Stage 2 submetrics",
    )
    colorbar_axis = fig.add_axes([0.25, 0.055, 0.50, 0.025])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Score")
    fig.suptitle("Where Each Design Gains and Loses", x=0.08, ha="left", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.90, bottom=0.17, hspace=0.38)
    save_figure(fig, output_dir, "02_submetric_heatmaps")


def correlation_text(x: np.ndarray, y: np.ndarray) -> str:
    return f"Pearson r = {np.corrcoef(x, y)[0, 1]:.2f}"


def plot_proxy_alignment(output_dir: Path) -> None:
    val = np.array([run["val"] for run in RUNS])
    targets = [("v1", "Official v1 PDMS", COLORS["blue"]), ("v2", "Official v2 Combined EPDMS", COLORS["orange"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, (key, title, color) in zip(axes, targets):
        test = np.array([run[key] for run in RUNS])
        ax.scatter(val, test, s=75, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        for run, x_value, y_value in zip(RUNS, val, test):
            ax.annotate(run["short"], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
        slope, intercept = np.polyfit(val, test, 1)
        x_line = np.linspace(val.min() - 0.003, val.max() + 0.003, 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLORS["gray"], linestyle="--", linewidth=1)
        ax.text(0.03, 0.95, correlation_text(val, test), transform=ax.transAxes, va="top", fontsize=9, color="#52606D")
        ax.set_title(title, loc="left")
        ax.set_xlabel("val/score_epoch (training proxy)")
        ax.set_ylabel(title)
        ax.grid(color="#E7EBEE", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Validation Proxy Does Not Reliably Rank Official Test Performance", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.01,
        f"Descriptive only (n={len(RUNS)}); validation proxy and official metrics are different evaluators.",
        fontsize=9,
        color="#52606D",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.16, wspace=0.28)
    save_figure(fig, output_dir, "03_validation_proxy_vs_test")


def plot_timing(output_dir: Path) -> None:
    rows = [run["label"] for run in RUNS]
    y = np.arange(len(RUNS))
    train = np.array([run["train_h"] for run in RUNS])
    val = np.array([run["val_h"] for run in RUNS])
    v2 = np.array([run["v2"] for run in RUNS])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].barh(y, train, color=COLORS["blue"], height=0.62, label="Train")
    axes[0].barh(y, val, left=train, color=COLORS["yellow"], height=0.62, label="Validation")
    for index, total in enumerate(train + val):
        axes[0].text(total + 0.08, index, f"{total:.2f} h", va="center", fontsize=8)
    axes[0].set_yticks(y, rows)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Wall-clock hours per recorded epoch")
    axes[0].set_title("Epoch wall time", loc="left")
    axes[0].grid(axis="x", color="#E7EBEE", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    total = train + val
    axes[1].scatter(
        total,
        v2,
        s=90,
        c=[
            COLORS["green"],
            COLORS["green"],
            COLORS["gray"],
            COLORS["gray"],
            COLORS["orange"],
            COLORS["red"],
            COLORS["blue"],
        ],
        edgecolor="white",
        linewidth=0.8,
    )
    for run, x_value, y_value in zip(RUNS, total, v2):
        axes[1].annotate(run["short"], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1].axhline(BASELINE["v2"], color=COLORS["red"], linestyle="--", linewidth=1.2, label="Pretrained baseline")
    axes[1].set_xlabel("Wall-clock hours per recorded epoch")
    axes[1].set_ylabel("Official v2 Combined EPDMS")
    axes[1].set_title("Cost vs. official v2 result", loc="left")
    axes[1].grid(color="#E7EBEE", linewidth=0.8)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_axisbelow(True)
    fig.suptitle("Training-Time Cost and Test-Time Outcome", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.01, "Wall times come from one recorded epoch per evaluated checkpoint; hardware/data settings differ across runs.", fontsize=9, color="#52606D")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.15, wspace=0.28)
    save_figure(fig, output_dir, "04_epoch_time_and_v2_tradeoff")


def grouped_barh(
    ax: plt.Axes,
    y: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    x_label: str,
    x_limits: tuple[float, float] | None = None,
) -> None:
    height = 0.72 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * height
    for offset, (name, values, color) in zip(offsets, series):
        ax.barh(y + offset, values, height=height * 0.88, label=name, color=color)
    ax.set_title(title, loc="left")
    ax.set_xlabel(x_label)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.grid(axis="x", color="#E7EBEE", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_train_val_metrics(output_dir: Path) -> None:
    labels = [run["label"] for run in RUNS]
    y = np.arange(len(RUNS))
    values = lambda key: np.array([run[key] for run in RUNS])

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.5), sharey=True)

    bars = axes[0, 0].barh(y, values("train_loss"), height=0.58, color=COLORS["blue"])
    annotate_bars(axes[0, 0], bars, digits=3)
    axes[0, 0].set_title("Training optimization", loc="left")
    axes[0, 0].set_xlabel("train/loss_epoch (lower is better)")
    axes[0, 0].grid(axis="x", color="#E7EBEE", linewidth=0.8)
    axes[0, 0].set_axisbelow(True)

    grouped_barh(
        axes[0, 1],
        y,
        [
            ("score", values("val"), COLORS["green"]),
            ("best score", values("val_best"), COLORS["orange"]),
            ("progress", values("val_progress"), COLORS["blue"]),
        ],
        "Validation score proxies",
        "Score (higher is better)",
        (0.86, 1.005),
    )

    grouped_barh(
        axes[1, 0],
        y,
        [
            ("collision", values("val_collision"), COLORS["blue"]),
            ("DAC", values("val_dac"), COLORS["green"]),
            ("TTC", values("val_ttc"), COLORS["orange"]),
            ("comfort", values("val_comfort"), COLORS["yellow"]),
        ],
        "Validation driving/compliance",
        "Score (higher is better)",
        (0.96, 1.002),
    )

    grouped_barh(
        axes[1, 1],
        y,
        [
            ("lost score ↓", values("val_lost"), COLORS["red"]),
            ("hit ↑", values("val_hit"), COLORS["green"]),
            ("top-5 hit ↑", values("val_top5"), COLORS["orange"]),
            ("L2 ↓", values("val_l2"), COLORS["gray"]),
        ],
        "Validation proposal diagnostics",
        "Metric value (direction shown in legend)",
        (0.0, 0.75),
    )

    legend_specs = [
        (axes[0, 1], (0.79, 0.505)),
        (axes[1, 0], (0.46, 0.075)),
        (axes[1, 1], (0.80, 0.075)),
    ]
    for ax, anchor in legend_specs:
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=8,
            loc="center",
            bbox_to_anchor=anchor,
            ncol=len(legend_labels),
        )

    for ax in axes[:, 0]:
        ax.set_yticks(y, labels)
    axes[0, 0].invert_yaxis()

    fig.suptitle("Train and Validation Metrics by Model Configuration", x=0.04, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.04,
        0.015,
        "Metrics are aligned to each evaluated checkpoint; values use their native scales and should be compared within a panel.",
        fontsize=9,
        color="#52606D",
    )
    fig.subplots_adjust(left=0.28, right=0.98, top=0.90, bottom=0.16, hspace=0.48, wspace=0.22)
    save_figure(fig, output_dir, "05_train_validation_metric_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/evaluation"),
        help="Directory for generated PNG and PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_overall(args.output_dir)
    plot_submetrics(args.output_dir)
    plot_proxy_alignment(args.output_dir)
    plot_timing(args.output_dir)
    plot_train_val_metrics(args.output_dir)
    print(f"Wrote evaluation figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
