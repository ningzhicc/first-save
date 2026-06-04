#!/usr/bin/env python3
"""Generate thesis figures and tables for the ABR experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "results"
    / "fcc-test_video1"
    / "trace_num_100_fixed_True"
)
LLAMA_ROOT = RESULT_ROOT / "llama_small"
FT_ROOT = PROJECT_ROOT / "data" / "ft_plms" / "llama_small" / "exp_pool_ssna"
OUTPUT_DIR = PROJECT_ROOT / "论文书写" / "论文图片"

VIDEO_BITRATE_KBPS = np.array([300, 750, 1200, 1850, 2850, 4300])


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    table_label: str
    category: str
    dirs: tuple[Path, ...]
    include_main: bool = True
    include_ablation: bool = False
    color: str = "#4C78A8"


def pdirs(pattern: str, exclude: tuple[str, ...] = ()) -> tuple[Path, ...]:
    return tuple(
        sorted(
            Path(p)
            for p in LLAMA_ROOT.glob(pattern)
            if all(token not in p.name for token in exclude)
        )
    )


def merge_dirs(*groups: tuple[Path, ...]) -> tuple[Path, ...]:
    merged = {}
    for group in groups:
        for path in group:
            merged[path.name] = path
    return tuple(sorted(merged.values()))


METHODS = [
    MethodSpec(
        "bba",
        "BBA",
        "BBA rule-based algorithm",
        "Traditional",
        tuple(sorted((RESULT_ROOT / "bba").glob("seed_*"))),
        color="#8E8E8E",
    ),
    MethodSpec(
        "mpc",
        "MPC",
        "MPC predictive control algorithm",
        "Traditional",
        tuple(sorted((RESULT_ROOT / "mpc").glob("seed_*"))),
        color="#5E5E5E",
    ),
    MethodSpec(
        "udr",
        "UDR",
        "UDR reinforcement learning baseline",
        "RL baseline",
        tuple(sorted((RESULT_ROOT / "udr_3").glob("seed_*"))),
        color="#6B8E23",
    ),
    MethodSpec(
        "genet",
        "Genet",
        "Genet reinforcement learning baseline",
        "RL baseline",
        tuple(sorted((RESULT_ROOT / "genet").glob("seed_*"))),
        color="#2A9D8F",
    ),
    MethodSpec(
        "semantic",
        "Semantic\nreprogramming",
        "Semantic reprogramming baseline",
        "This work",
        merge_dirs(
            pdirs("sr_sfd256_h4_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1"),
            pdirs("sr_sfd256_h4_isa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1"),
        ),
        include_ablation=True,
        color="#4C78A8",
    ),
    MethodSpec(
        "context",
        "Context\npre-alignment",
        "Context pre-alignment enhanced model",
        "This work",
        pdirs("sr_sfd256_h4_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1"),
        include_ablation=True,
        color="#F58518",
    ),
    MethodSpec(
        "hmix",
        "Original\nmulti-scale",
        "Original multi-scale history mixer",
        "This work",
        pdirs(
            "sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1",
            exclude=("v2lite",),
        ),
        include_main=False,
        include_ablation=True,
        color="#B279A2",
    ),
    MethodSpec(
        "lightweight",
        "Lightweight\nmulti-scale",
        "Lightweight multi-scale history enhanced model",
        "This work",
        pdirs("sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_v2lite_stop-1_tgt1"),
        include_ablation=True,
        color="#E45756",
    ),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def infer_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_s(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer seed from {path}")


def parse_result_file(path: Path, skip_first_reward: bool = True) -> pd.DataFrame:
    rows = []
    first_valid = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) <= 1:
                continue
            if len(parts) < 8:
                continue
            if first_valid and skip_first_reward:
                first_valid = False
                continue
            first_valid = False
            rows.append(
                {
                    "time": float(parts[0]),
                    "bitrate": float(parts[1]),
                    "buffer": float(parts[2]),
                    "rebuf": float(parts[3]),
                    "chunk_size": float(parts[4]),
                    "download_time": float(parts[5]),
                    "smooth": float(parts[6]),
                    "reward": float(parts[7]),
                }
            )
    return pd.DataFrame(rows)


def parse_result_file_full(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "time": float(parts[0]),
                    "bitrate": float(parts[1]),
                    "buffer": float(parts[2]),
                    "rebuf": float(parts[3]),
                    "chunk_size": float(parts[4]),
                    "download_time": float(parts[5]),
                    "smooth": float(parts[6]),
                    "reward": float(parts[7]),
                }
            )
    return pd.DataFrame(rows)


def collect_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    trace_rows = []

    for spec in METHODS:
        for result_dir in spec.dirs:
            if not result_dir.exists():
                continue
            files = sorted(result_dir.glob("result_sim_abr_*"))
            if not files:
                continue
            seed = infer_seed(result_dir)
            trace_frames = []
            for result_file in files:
                reward_df = parse_result_file(result_file, skip_first_reward=True)
                full_df = parse_result_file_full(result_file)
                if reward_df.empty or full_df.empty:
                    continue
                trace_frames.append(
                    pd.DataFrame(
                        {
                            "reward": reward_df["reward"],
                            "bitrate": full_df["bitrate"],
                            "rebuf": full_df["rebuf"],
                            "smooth": full_df["smooth"],
                        }
                    )
                )
                trace_rows.append(
                    {
                        "method": spec.key,
                        "label": spec.label.replace("\n", " "),
                        "table_label": spec.table_label,
                        "category": spec.category,
                        "seed": seed,
                        "trace": result_file.name,
                        "mean_reward": reward_df["reward"].mean(),
                        "bitrate": full_df["bitrate"].mean(),
                        "rebuf": full_df["rebuf"].mean(),
                        "smooth": full_df["smooth"].mean(),
                    }
                )
            if not trace_frames:
                continue
            seed_df = pd.concat(trace_frames, ignore_index=True)
            seed_rows.append(
                {
                    "method": spec.key,
                    "label": spec.label.replace("\n", " "),
                    "table_label": spec.table_label,
                    "category": spec.category,
                    "seed": seed,
                    "trace_count": len(trace_frames),
                    "sample_count": len(seed_df),
                    "mean_reward": seed_df["reward"].mean(),
                    "bitrate": seed_df["bitrate"].mean(),
                    "rebuf": seed_df["rebuf"].mean(),
                    "smooth": seed_df["smooth"].mean(),
                }
            )

    return pd.DataFrame(seed_rows), pd.DataFrame(trace_rows)


def aggregate_seed_metrics(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METHODS:
        group = seed_df[seed_df["method"] == spec.key].copy()
        if group.empty:
            continue
        row = {
            "method": spec.key,
            "label": spec.label.replace("\n", " "),
            "table_label": spec.table_label,
            "category": spec.category,
            "seed_count": group["seed"].nunique(),
        }
        for metric in ["mean_reward", "bitrate", "rebuf", "smooth"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, values, fmt: str = "{:.3f}", dy: float = 0.01) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * dy
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7.2,
        )


def plot_main_reward(summary: pd.DataFrame) -> None:
    specs = [s for s in METHODS if s.include_main and s.key in set(summary["method"])]
    data = summary.set_index("method").loc[[s.key for s in specs]].reset_index()
    colors = [s.color for s in specs]

    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    x = np.arange(len(data))
    bars = ax.bar(
        x,
        data["mean_reward_mean"],
        yerr=data["mean_reward_std"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in specs], rotation=0)
    ax.set_ylabel("Mean QoE reward")
    ax.set_title("Overall ABR performance on FCC traces")
    ax.grid(axis="y")
    ax.set_ylim(0, max(data["mean_reward_mean"] + data["mean_reward_std"]) * 1.18)
    add_bar_labels(ax, bars, data["mean_reward_mean"], "{:.3f}", dy=0.015)
    for idx, seed_count in enumerate(data["seed_count"]):
        ax.text(idx, 0.025, f"n={int(seed_count)}", ha="center", va="bottom", color="#555555", fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, "fig1_main_qoe_reward")


def plot_components(summary: pd.DataFrame) -> None:
    specs = [s for s in METHODS if s.include_main and s.key in set(summary["method"])]
    data = summary.set_index("method").loc[[s.key for s in specs]].reset_index()
    colors = [s.color for s in specs]
    metrics = [
        ("bitrate", "Average bitrate (Kbps)", "{:.0f}"),
        ("rebuf", "Rebuffering time (s)", "{:.3f}"),
        ("smooth", "Smoothness penalty", "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    x = np.arange(len(data))
    for ax, (metric, ylabel, fmt) in zip(axes, metrics):
        means = data[f"{metric}_mean"]
        stds = data[f"{metric}_std"]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="#222222",
            linewidth=0.5,
            capsize=2.5,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([s.label.split("\n")[0] for s in specs], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_ylim(0, max(means + stds) * 1.18)
        ax.set_title(ylabel.split(" (")[0])
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "fig2_qoe_components")


def plot_ablation(summary: pd.DataFrame) -> None:
    specs = [s for s in METHODS if s.include_ablation and s.key in set(summary["method"])]
    data = summary.set_index("method").loc[[s.key for s in specs]].reset_index()
    colors = [s.color for s in specs]

    fig, ax = plt.subplots(figsize=(5.8, 3.25))
    x = np.arange(len(data))
    bars = ax.bar(
        x,
        data["mean_reward_mean"],
        yerr=data["mean_reward_std"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.plot(x, data["mean_reward_mean"], color="#222222", linewidth=0.8, marker="o", markersize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in specs])
    ax.set_ylabel("Mean QoE reward")
    ax.set_title("Ablation of the proposed model components")
    ax.grid(axis="y")
    lower = max(0.0, data["mean_reward_mean"].min() - 0.08)
    upper = min(1.05, max(data["mean_reward_mean"] + data["mean_reward_std"]) + 0.07)
    ax.set_ylim(lower, upper)
    add_bar_labels(ax, bars, data["mean_reward_mean"], "{:.3f}", dy=0.014)
    for idx, seed_count in enumerate(data["seed_count"]):
        ax.text(idx, lower + (upper - lower) * 0.035, f"n={int(seed_count)}", ha="center", color="#555555", fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, "fig3_ablation_reward")


def plot_seed_stability(seed_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    keys = ["genet", "context", "lightweight"]
    labels = ["Genet", "Context\npre-alignment", "Lightweight\nmulti-scale"]
    colors = {"genet": "#2A9D8F", "context": "#F58518", "lightweight": "#E45756"}

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    rng = np.random.default_rng(7)
    for idx, key in enumerate(keys):
        values = seed_df[seed_df["method"] == key]["mean_reward"].to_numpy()
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=26,
            color=colors[key],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        ax.errorbar(
            idx,
            mean,
            yerr=std,
            fmt="o",
            color="#1A1A1A",
            capsize=4,
            markersize=4,
            linewidth=1.0,
            zorder=4,
        )
        ax.hlines(mean, idx - 0.22, idx + 0.22, color="#1A1A1A", linewidth=1.2)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean QoE reward per seed")
    ax.set_title("Multi-seed stability")
    ax.grid(axis="y")
    ax.set_xlim(-0.55, len(keys) - 0.45)
    fig.tight_layout()
    save_figure(fig, "fig4_seed_stability")


def aggregate_trace_means(trace_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trace_df.groupby(["method", "trace"], as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            bitrate=("bitrate", "mean"),
            rebuf=("rebuf", "mean"),
            smooth=("smooth", "mean"),
            seed_count=("seed", "nunique"),
        )
    )


def plot_cdf(trace_summary: pd.DataFrame) -> None:
    keys = ["genet", "context", "lightweight"]
    labels = {
        "genet": "Genet",
        "context": "Context pre-alignment",
        "lightweight": "Lightweight multi-scale",
    }
    colors = {"genet": "#2A9D8F", "context": "#F58518", "lightweight": "#E45756"}

    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    for key in keys:
        values = np.sort(trace_summary[trace_summary["method"] == key]["mean_reward"].to_numpy())
        if len(values) == 0:
            continue
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, label=labels[key], color=colors[key], linewidth=1.8)
    ax.set_xlabel("Per-trace mean QoE reward")
    ax.set_ylabel("CDF")
    ax.set_title("QoE distribution across FCC traces")
    ax.grid(True)
    ax.text(
        0.25,
        0.19,
        "Better ==>",
        transform=ax.transAxes,
        fontsize=8.8,
        fontstyle="italic",
        color="#333333",
        ha="center",
        va="center",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig5_qoe_cdf")


def plot_per_trace_gain(trace_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = trace_summary.pivot(index="trace", columns="method", values="mean_reward")
    common = pivot.dropna(subset=["genet", "lightweight"]).copy()
    common["reward_gain"] = common["lightweight"] - common["genet"]
    common = common.sort_values("reward_gain").reset_index()
    win_rate = (common["reward_gain"] > 0).mean()
    mean_gain = common["reward_gain"].mean()

    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    x = np.arange(len(common))
    colors = np.where(common["reward_gain"] >= 0, "#E45756", "#7F8C8D")
    ax.bar(x, common["reward_gain"], color=colors, width=0.82, linewidth=0)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("FCC traces sorted by reward gain")
    ax.set_ylabel("Reward gain over Genet")
    ax.set_title("Per-trace gain of the lightweight multi-scale model")
    ax.grid(axis="y")
    ax.text(
        0.02,
        0.94,
        f"Win rate: {win_rate * 100:.1f}%\nMean gain: {mean_gain:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.6},
    )
    fig.tight_layout()
    save_figure(fig, "fig6_per_trace_gain")
    common.to_csv(OUTPUT_DIR / "per_trace_reward_gain_lightweight_vs_genet.csv", index=False)
    return pd.DataFrame(
        [
            {
                "comparison": "Lightweight multi-scale vs Genet",
                "common_trace_count": len(common),
                "win_rate": win_rate,
                "mean_reward_gain": mean_gain,
                "median_reward_gain": common["reward_gain"].median(),
            }
        ]
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def read_loss_file(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def plot_training_loss() -> None:
    loss_paths = sorted(
        FT_ROOT.glob(
            "sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_v2lite/early_stop_-1_checkpoint/train_losses.txt"
        )
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    plotted = 0
    interpolated = []
    for path in loss_paths:
        values = read_loss_file(path)
        if len(values) < 2:
            continue
        seed = infer_seed(path.parents[1])
        window = max(5, len(values) // 80)
        smooth = moving_average(values, window)
        x = np.linspace(0, 1, len(smooth))
        ax.plot(x, smooth, color="#E45756", alpha=0.22, linewidth=0.9)
        interpolated.append(np.interp(np.linspace(0, 1, 250), x, smooth))
        plotted += 1
    if interpolated:
        mean_curve = np.mean(np.vstack(interpolated), axis=0)
        ax.plot(np.linspace(0, 1, 250), mean_curve, color="#B51D2A", linewidth=2.0, label=f"Mean over {plotted} seeds")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No training loss files found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Normalized training progress")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss of the final model")
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, "fig7_training_loss")


def choose_case_trace(trace_summary: pd.DataFrame) -> str | None:
    pivot = trace_summary.pivot(index="trace", columns="method", values="mean_reward")
    common = pivot.dropna(subset=["genet", "lightweight"]).copy()
    if common.empty:
        return None
    common["gain"] = common["lightweight"] - common["genet"]
    return str(common["gain"].idxmax())


def plot_case_trace(trace_summary: pd.DataFrame) -> None:
    trace_name = choose_case_trace(trace_summary)
    if trace_name is None:
        return
    genet_dir = RESULT_ROOT / "genet" / "seed_100001"
    light_dirs = [s for s in METHODS if s.key == "lightweight"][0].dirs
    light_dir = next((d for d in light_dirs if infer_seed(d) == 100001), light_dirs[0])
    genet_file = genet_dir / trace_name
    light_file = light_dir / trace_name
    if not genet_file.exists() or not light_file.exists():
        return

    genet = parse_result_file_full(genet_file)
    light = parse_result_file_full(light_file)
    n = min(len(genet), len(light))
    genet = genet.iloc[:n].copy()
    light = light.iloc[:n].copy()
    x = np.arange(1, n + 1)

    def throughput_mbps(df: pd.DataFrame) -> np.ndarray:
        delay_ms = np.maximum(df["download_time"].to_numpy(), 1e-6)
        return df["chunk_size"].to_numpy() * 8.0 / delay_ms / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.2), sharex=True)
    axes[0].plot(x, throughput_mbps(light), color="#4C78A8", linewidth=1.2)
    axes[0].set_ylabel("Est. throughput\n(Mbps)")
    axes[0].set_title("Case study on a representative trace")

    axes[1].step(x, genet["bitrate"], where="post", label="Genet", color="#2A9D8F", linewidth=1.4)
    axes[1].step(x, light["bitrate"], where="post", label="Lightweight multi-scale", color="#E45756", linewidth=1.4)
    axes[1].set_ylabel("Bitrate\n(Kbps)")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].plot(x, genet["buffer"], label="Genet", color="#2A9D8F", linewidth=1.2)
    axes[2].plot(x, light["buffer"], label="Lightweight multi-scale", color="#E45756", linewidth=1.2)
    axes[2].set_ylabel("Buffer\n(s)")

    width = 0.38
    axes[3].bar(x - width / 2, genet["rebuf"], width=width, color="#2A9D8F", alpha=0.75, label="Genet")
    axes[3].bar(x + width / 2, light["rebuf"], width=width, color="#E45756", alpha=0.75, label="Lightweight")
    axes[3].set_ylabel("Rebuffer\n(s)")
    axes[3].set_xlabel("Video chunk index")

    for ax in axes:
        ax.grid(axis="y")
    short_name = trace_name.replace("result_sim_abr_fcc-test_", "")
    if len(short_name) > 72:
        short_name = short_name[:69] + "..."
    axes[0].text(0.01, 0.88, short_name, transform=axes[0].transAxes, fontsize=7.2, color="#555555")
    fig.tight_layout()
    save_figure(fig, "fig8_case_trace_analysis")


def format_summary_table(summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    data = summary.set_index("method").loc[keys].reset_index()
    return pd.DataFrame(
        {
            "Method": data["table_label"],
            "Category": data["category"],
            "Seeds": data["seed_count"].astype(int),
            "Mean reward": data.apply(lambda r: f"{r['mean_reward_mean']:.4f} +/- {r['mean_reward_std']:.4f}", axis=1),
            "Bitrate": data.apply(lambda r: f"{r['bitrate_mean']:.1f} +/- {r['bitrate_std']:.1f}", axis=1),
            "Rebuffering": data.apply(lambda r: f"{r['rebuf_mean']:.4f} +/- {r['rebuf_std']:.4f}", axis=1),
            "Smoothness": data.apply(lambda r: f"{r['smooth_mean']:.4f} +/- {r['smooth_std']:.4f}", axis=1),
        }
    )


def save_table_image(df: pd.DataFrame, stem: str, title: str, figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, pad=8, weight="bold")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#F1F3F5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    fig.tight_layout()
    save_figure(fig, stem)


def save_tables(seed_df: pd.DataFrame, summary: pd.DataFrame, winrate: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_df.sort_values(["method", "seed"]).to_csv(OUTPUT_DIR / "table_seed_level_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "table_all_method_summary.csv", index=False)

    main_keys = [s.key for s in METHODS if s.include_main and s.key in set(summary["method"])]
    ablation_keys = [s.key for s in METHODS if s.include_ablation and s.key in set(summary["method"])]
    main_table = format_summary_table(summary, main_keys)
    ablation_table = format_summary_table(summary, ablation_keys)

    main_table.to_csv(OUTPUT_DIR / "table_main_results.csv", index=False)
    ablation_table.to_csv(OUTPUT_DIR / "table_ablation_results.csv", index=False)
    main_table.to_markdown(OUTPUT_DIR / "table_main_results.md", index=False)
    ablation_table.to_markdown(OUTPUT_DIR / "table_ablation_results.md", index=False)
    winrate.to_csv(OUTPUT_DIR / "table_per_trace_winrate.csv", index=False)
    winrate.to_markdown(OUTPUT_DIR / "table_per_trace_winrate.md", index=False)

    save_table_image(main_table, "table_main_results", "Main Results", (8.4, 2.8))
    save_table_image(ablation_table, "table_ablation_results", "Ablation Results", (7.6, 2.2))


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df, trace_df = collect_metrics()
    if seed_df.empty:
        raise RuntimeError(f"No result files found under {RESULT_ROOT}")
    summary = aggregate_seed_metrics(seed_df)
    trace_summary = aggregate_trace_means(trace_df)

    plot_main_reward(summary)
    plot_components(summary)
    plot_ablation(summary)
    plot_seed_stability(seed_df, summary)
    plot_cdf(trace_summary)
    winrate = plot_per_trace_gain(trace_summary)
    plot_training_loss()
    plot_case_trace(trace_summary)
    save_tables(seed_df, summary, winrate)

    print(f"Generated figures and tables in: {OUTPUT_DIR}")
    print(summary[["method", "seed_count", "mean_reward_mean", "mean_reward_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
