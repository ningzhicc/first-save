#!/usr/bin/env python3
"""Generate supplementary evidence figures/tables for the ABR thesis.

Outputs:
1. video1/video2 chunk-size distribution figure
2. video2 failure trace case study figure
3. paired bootstrap confidence-interval table for Genet vs final model
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")
RESULT_ROOT = ROOT / "artifacts" / "results"
VIDEO_ROOT = ROOT / "data" / "videos"
TRACE_ROOT = ROOT / "data" / "traces" / "test" / "fcc-test"
OUT_DIR = ROOT / "论文书写" / "论文图片"
FONT_PATH = ROOT / "图片制作" / "fonts" / "NotoSansCJKsc-Regular.otf"

BITRATES = np.array([300, 750, 1200, 1850, 2850, 4300])


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    result_subdir: str
    final_pattern: str


EXPERIMENTS = [
    Experiment(
        key="fcc-test_video1",
        label="FCC-test + video1",
        result_subdir="fcc-test_video1/trace_num_100_fixed_True",
        final_pattern="sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_v2lite_stop-1_tgt1",
    ),
    Experiment(
        key="fcc-valid_video1",
        label="FCC-valid + video1",
        result_subdir="fcc-valid_video1/trace_num_100_fixed_True",
        final_pattern="sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1",
    ),
    Experiment(
        key="fcc-test_video2",
        label="FCC-test + video2",
        result_subdir="fcc-test_video2/trace_num_100_fixed_True",
        final_pattern="sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s*_stop-1_tgt1",
    ),
]


def configure_style() -> FontProperties | None:
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 420,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9DEE8",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def infer_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_s(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer seed from {path}")


def parse_result_file(path: Path, skip_first_reward: bool = False) -> pd.DataFrame:
    rows = []
    first_valid = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 8:
                continue
            if skip_first_reward and first_valid:
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


def mean_reward(path: Path) -> float:
    df = parse_result_file(path, skip_first_reward=True)
    return float(df["reward"].mean())


def trace_key_from_result_name(name: str) -> str:
    for prefix in ["result_sim_abr_fcc-test_", "result_sim_abr_fcc-valid_"]:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name.removeprefix("result_sim_abr_")


def load_video_sizes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for video in ["video1", "video2"]:
        for bitrate_idx, bitrate in enumerate(BITRATES):
            path = VIDEO_ROOT / f"{video}_sizes" / f"video_size_{bitrate_idx}"
            values = np.loadtxt(path, dtype=float)
            for chunk_idx, size_bytes in enumerate(values, start=1):
                rows.append(
                    {
                        "video": video,
                        "bitrate_kbps": int(bitrate),
                        "chunk_index": chunk_idx,
                        "size_bytes": size_bytes,
                        "size_mb": size_bytes / (1024 * 1024),
                        "size_mbit": size_bytes * 8 / 1_000_000,
                    }
                )
            summary_rows.append(
                {
                    "video": video,
                    "bitrate_kbps": int(bitrate),
                    "chunk_count": len(values),
                    "mean_size_mb": values.mean() / (1024 * 1024),
                    "median_size_mb": np.median(values) / (1024 * 1024),
                    "p95_size_mb": np.percentile(values, 95) / (1024 * 1024),
                    "max_size_mb": values.max() / (1024 * 1024),
                }
            )
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    pivot = summary.pivot(index="bitrate_kbps", columns="video", values="mean_size_mb")
    summary["video2_vs_video1_mean_ratio"] = summary.apply(
        lambda r: pivot.loc[r["bitrate_kbps"], "video2"] / pivot.loc[r["bitrate_kbps"], "video1"]
        if r["video"] == "video2"
        else np.nan,
        axis=1,
    )
    return df, summary


def plot_video_chunk_size_distribution(font: FontProperties | None) -> None:
    df, summary = load_video_sizes()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "table_supp_video1_video2_chunk_size_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    colors = {"video1": "#4C78A8", "video2": "#F58518"}

    ax = axes[0]
    positions = np.arange(len(BITRATES))
    offset = 0.18
    for video, shift in [("video1", -offset), ("video2", offset)]:
        data = [
            df[(df["video"] == video) & (df["bitrate_kbps"] == bitrate)]["size_mb"].to_numpy()
            for bitrate in BITRATES
        ]
        bp = ax.boxplot(
            data,
            positions=positions + shift,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1F2937", "linewidth": 1.2},
            whiskerprops={"color": "#4B5563", "linewidth": 0.9},
            capprops={"color": "#4B5563", "linewidth": 0.9},
            boxprops={"edgecolor": "#1F2937", "linewidth": 0.8},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[video])
            patch.set_alpha(0.78)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(b) for b in BITRATES])
    ax.set_xlabel("码率档位（Kbps）", fontproperties=font)
    ax.set_ylabel("单分片大小（MB）", fontproperties=font)
    ax.set_title("不同码率档位的分片大小分布", fontproperties=font)
    ax.grid(axis="y")
    handles = [
        plt.Line2D([0], [0], color=colors["video1"], lw=8, alpha=0.78, label="video1"),
        plt.Line2D([0], [0], color=colors["video2"], lw=8, alpha=0.78, label="video2"),
    ]
    ax.legend(handles=handles, prop=font, frameon=False, loc="upper left")

    ax = axes[1]
    chunk_means = df.groupby(["video", "chunk_index"], as_index=False)["size_mb"].mean()
    for video in ["video1", "video2"]:
        sub = chunk_means[chunk_means["video"] == video]
        ax.plot(
            sub["chunk_index"],
            sub["size_mb"],
            color=colors[video],
            linewidth=1.8,
            label=video,
        )
        ax.scatter(sub["chunk_index"], sub["size_mb"], color=colors[video], s=11, alpha=0.75)
    mean_v1 = chunk_means[chunk_means["video"] == "video1"]["size_mb"].mean()
    mean_v2 = chunk_means[chunk_means["video"] == "video2"]["size_mb"].mean()
    ax.axhline(mean_v1, color=colors["video1"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(mean_v2, color=colors["video2"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        0.02,
        0.95,
        f"平均分片大小：video1={mean_v1:.2f} MB，video2={mean_v2:.2f} MB",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontproperties=font,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D1D5DB", "linewidth": 0.6},
    )
    ax.set_xlabel("视频分片序号", fontproperties=font)
    ax.set_ylabel("跨码率平均分片大小（MB）", fontproperties=font)
    ax.set_title("沿播放时间的分片大小变化", fontproperties=font)
    ax.grid(axis="y")
    ax.legend(prop=font, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig_supp_video1_video2_chunk_size_distribution")


def method_dirs(exp: Experiment, method: str) -> tuple[Path, ...]:
    root = RESULT_ROOT / exp.result_subdir
    if method == "genet":
        return tuple(sorted((root / "genet").glob("seed_*")))
    if method == "final":
        return tuple(sorted((root / "llama_small").glob(exp.final_pattern)))
    raise ValueError(method)


def collect_trace_rewards(exp: Experiment, method: str) -> pd.DataFrame:
    rows = []
    for result_dir in method_dirs(exp, method):
        seed = infer_seed(result_dir)
        for file in sorted(result_dir.glob("result_sim_abr*")):
            try:
                reward = mean_reward(file)
            except Exception:
                continue
            rows.append(
                {
                    "experiment": exp.key,
                    "method": method,
                    "seed": seed,
                    "trace": file.name,
                    "trace_key": trace_key_from_result_name(file.name),
                    "mean_reward": reward,
                }
            )
    return pd.DataFrame(rows)


def collect_all_trace_rewards() -> pd.DataFrame:
    frames = []
    for exp in EXPERIMENTS:
        for method in ["genet", "final"]:
            df = collect_trace_rewards(exp, method)
            if df.empty:
                raise RuntimeError(f"No results for {exp.key} / {method}")
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT_DIR / "table_supp_trace_level_rewards_genet_final.csv", index=False)
    return all_df


def paired_trace_table(trace_rewards: pd.DataFrame, exp: Experiment) -> pd.DataFrame:
    sub = trace_rewards[trace_rewards["experiment"] == exp.key]
    trace_mean = (
        sub.groupby(["method", "trace_key"], as_index=False)["mean_reward"]
        .mean()
        .pivot(index="trace_key", columns="method", values="mean_reward")
        .dropna(subset=["genet", "final"])
    )
    trace_mean["gain"] = trace_mean["final"] - trace_mean["genet"]
    return trace_mean


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 20260516) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = values[rng.integers(0, n, size=n)].mean()
    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        float((boot > 0).mean()),
    )


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    header = "| " + " | ".join(df.columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text(header + sep + "\n".join(rows) + "\n", encoding="utf-8")


def plot_bootstrap_table(font: FontProperties | None, trace_rewards: pd.DataFrame) -> None:
    rows = []
    for idx, exp in enumerate(EXPERIMENTS):
        table = paired_trace_table(trace_rewards, exp)
        gains = table["gain"].to_numpy()
        low, high, p_pos = bootstrap_ci(gains, seed=20260516 + idx)
        rows.append(
            {
                "实验配置": exp.label,
                "公共 trace 数": len(table),
                "Genet 平均奖励": table["genet"].mean(),
                "最终模型平均奖励": table["final"].mean(),
                "平均增益": gains.mean(),
                "95% CI 下界": low,
                "95% CI 上界": high,
                "P(增益>0)": p_pos,
                "逐 trace 胜率": float((gains > 0).mean()),
                "中位数增益": float(np.median(gains)),
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "table_supp_bootstrap_reward_gain_ci.csv", index=False)

    pretty = pd.DataFrame(
        {
            "实验配置": raw["实验配置"],
            "n": raw["公共 trace 数"].astype(int),
            "Genet": raw["Genet 平均奖励"].map(lambda x: f"{x:.4f}"),
            "最终模型": raw["最终模型平均奖励"].map(lambda x: f"{x:.4f}"),
            "平均增益": raw["平均增益"].map(lambda x: f"{x:.4f}"),
            "95% CI": raw.apply(lambda r: f"[{r['95% CI 下界']:.4f}, {r['95% CI 上界']:.4f}]", axis=1),
            "P(增益>0)": raw["P(增益>0)"].map(lambda x: f"{x * 100:.1f}%"),
            "胜率": raw["逐 trace 胜率"].map(lambda x: f"{x * 100:.1f}%"),
            "中位增益": raw["中位数增益"].map(lambda x: f"{x:.4f}"),
        }
    )
    save_markdown(pretty, OUT_DIR / "table_supp_bootstrap_reward_gain_ci.md")

    fig, ax = plt.subplots(figsize=(10.8, 2.25))
    ax.axis("off")
    ax.set_title("Genet 与最终模型的 paired bootstrap 奖励增益置信区间", loc="left", fontproperties=font, pad=8)
    table = ax.table(
        cellText=pretty.values,
        colLabels=pretty.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.055, 0.09, 0.10, 0.095, 0.16, 0.105, 0.08, 0.095],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.7)
    table.scale(1, 1.28)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#EEF2F7")
            cell.set_text_props(weight="bold", fontproperties=font)
        else:
            if row % 2 == 0:
                cell.set_facecolor("#FAFAFA")
            cell.set_text_props(fontproperties=font)
    fig.tight_layout()
    save_figure(fig, "table_supp_bootstrap_reward_gain_ci")


def choose_video2_failure_case() -> tuple[str, pd.DataFrame]:
    exp = next(e for e in EXPERIMENTS if e.key == "fcc-test_video2")
    genet_dir = next(d for d in method_dirs(exp, "genet") if infer_seed(d) == 100005)
    final_dir = next(d for d in method_dirs(exp, "final") if infer_seed(d) == 100005)

    rows = []
    final_files = {p.name: p for p in final_dir.glob("result_sim_abr*")}
    for genet_file in genet_dir.glob("result_sim_abr*"):
        final_file = final_files.get(genet_file.name)
        if final_file is None:
            continue
        g_full = parse_result_file(genet_file, skip_first_reward=False)
        f_full = parse_result_file(final_file, skip_first_reward=False)
        g_reward = mean_reward(genet_file)
        f_reward = mean_reward(final_file)
        rows.append(
            {
                "trace": genet_file.name,
                "trace_key": trace_key_from_result_name(genet_file.name),
                "genet_reward": g_reward,
                "final_reward": f_reward,
                "reward_gain": f_reward - g_reward,
                "genet_bitrate": g_full["bitrate"].mean(),
                "final_bitrate": f_full["bitrate"].mean(),
                "genet_rebuf": g_full["rebuf"].mean(),
                "final_rebuf": f_full["rebuf"].mean(),
                "genet_smooth": g_full["smooth"].mean(),
                "final_smooth": f_full["smooth"].mean(),
            }
        )
    case_df = pd.DataFrame(rows).sort_values("reward_gain")
    if case_df.empty:
        raise RuntimeError("No common video2 seed100005 trace files found")
    case_df.to_csv(OUT_DIR / "table_supp_video2_seed100005_trace_gain_ranking.csv", index=False)
    return str(case_df.iloc[0]["trace"]), case_df


def read_trace_throughput(trace_key: str) -> pd.DataFrame:
    path = TRACE_ROOT / trace_key
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                rows.append({"time_s": float(parts[0]), "throughput_mbps": float(parts[1])})
    return pd.DataFrame(rows)


def plot_video2_failure_case(font: FontProperties | None) -> None:
    trace_name, ranking = choose_video2_failure_case()
    exp = next(e for e in EXPERIMENTS if e.key == "fcc-test_video2")
    genet_dir = next(d for d in method_dirs(exp, "genet") if infer_seed(d) == 100005)
    final_dir = next(d for d in method_dirs(exp, "final") if infer_seed(d) == 100005)
    genet_file = genet_dir / trace_name
    final_file = final_dir / trace_name
    trace_key = trace_key_from_result_name(trace_name)

    g = parse_result_file(genet_file, skip_first_reward=False)
    f = parse_result_file(final_file, skip_first_reward=False)
    n = min(len(g), len(f))
    g = g.iloc[:n].copy()
    f = f.iloc[:n].copy()
    x = np.arange(1, n + 1)

    def observed_throughput_mbps(df: pd.DataFrame) -> np.ndarray:
        delay_ms = np.maximum(df["download_time"].to_numpy(), 1e-9)
        return df["chunk_size"].to_numpy() * 8.0 / delay_ms / 1000.0

    case_row = ranking.iloc[0].copy()
    pd.DataFrame([case_row]).to_csv(OUT_DIR / "table_supp_video2_failure_case_metrics.csv", index=False)

    colors = {"genet": "#4C78A8", "final": "#F58518"}
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 6.8), sharex=True, gridspec_kw={"hspace": 0.13})

    axes[0].plot(x, observed_throughput_mbps(g), color=colors["genet"], linewidth=1.25, label="Genet")
    axes[0].plot(x, observed_throughput_mbps(f), color=colors["final"], linewidth=1.25, label="最终模型")
    axes[0].set_ylabel("观测吞吐\n(Mbps)", fontproperties=font)
    axes[0].set_title("video2 失败案例：最终模型在 seed 100005 下的激进码率选择", fontproperties=font)
    trace_parts = trace_key.split("_")
    short_trace = f"{trace_parts[0]}_trace_{trace_parts[4]}_{trace_parts[-2]}_{trace_parts[-1]}" if len(trace_parts) > 6 else trace_key
    axes[0].text(
        0.01,
        0.88,
        f"trace: {short_trace}",
        transform=axes[0].transAxes,
        fontsize=8.2,
        color="#4B5563",
        fontproperties=font,
    )
    axes[0].legend(prop=font, frameon=False, ncol=2, loc="upper right")

    axes[1].step(x, g["bitrate"], where="post", color=colors["genet"], linewidth=1.5, label="Genet")
    axes[1].step(x, f["bitrate"], where="post", color=colors["final"], linewidth=1.5, label="最终模型")
    axes[1].set_ylabel("码率\n(Kbps)", fontproperties=font)
    axes[1].legend(prop=font, frameon=False, ncol=2, loc="upper right")

    axes[2].plot(x, g["buffer"], color=colors["genet"], linewidth=1.4, label="Genet")
    axes[2].plot(x, f["buffer"], color=colors["final"], linewidth=1.4, label="最终模型")
    axes[2].fill_between(x, 0, 2, color="#FEE2E2", alpha=0.35, linewidth=0)
    axes[2].set_ylabel("缓冲区\n(s)", fontproperties=font)

    width = 0.38
    axes[3].bar(x - width / 2, g["rebuf"], width=width, color=colors["genet"], alpha=0.72, label="Genet")
    axes[3].bar(x + width / 2, f["rebuf"], width=width, color=colors["final"], alpha=0.72, label="最终模型")
    axes[3].set_ylabel("重缓冲\n(s)", fontproperties=font)
    axes[3].set_xlabel("视频分片序号", fontproperties=font)

    summary_text = (
        f"平均奖励差值 {case_row['reward_gain']:.3f}; "
        f"码率 {case_row['genet_bitrate']:.0f}->{case_row['final_bitrate']:.0f} Kbps; "
        f"重缓冲 {case_row['genet_rebuf']:.3f}->{case_row['final_rebuf']:.3f}"
    )
    axes[3].text(
        0.01,
        0.92,
        summary_text,
        transform=axes[3].transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontproperties=font,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "linewidth": 0.6},
    )

    for ax in axes:
        ax.grid(axis="y")
    axes[1].set_xlim(1, n)
    fig.tight_layout()
    save_figure(fig, "fig_supp_video2_failure_trace_case_seed100005")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = configure_style()
    plot_video_chunk_size_distribution(font)
    trace_rewards = collect_all_trace_rewards()
    plot_video2_failure_case(font)
    plot_bootstrap_table(font, trace_rewards)
    print(f"Generated supplementary outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
