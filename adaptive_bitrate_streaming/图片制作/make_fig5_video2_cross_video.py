from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")
FIG_DIR = ROOT / "论文书写" / "论文图片"
FONT_PATH = ROOT / "图片制作" / "fonts" / "NotoSansCJKsc-Regular.otf"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FIG_DIR / "table_video1_video2_seed_metrics.csv")
    summary = pd.read_csv(FIG_DIR / "table_video1_video2_summary.csv")
    comp = pd.read_csv(FIG_DIR / "table_video1_video2_per_trace_comparison.csv")

    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    colors = {"genet": "#4C78A8", "final": "#F58518"}
    labels = {"genet": "Genet", "final": "最终模型"}
    videos = ["video1", "video2"]
    video_labels = {"video1": "video1 主测试", "video2": "video2 跨视频"}

    fig = plt.figure(figsize=(9.2, 4.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    x = range(len(videos))
    width = 0.32
    for offset, method in [(-width / 2, "genet"), (width / 2, "final")]:
        rows = summary.set_index(["video", "method"]).loc[(slice(None), method), :]
        means = [rows.loc[(v, method), "mean_reward_mean"] for v in videos]
        stds = [rows.loc[(v, method), "mean_reward_std"] for v in videos]
        ax0.bar(
            [i + offset for i in x],
            means,
            width,
            yerr=stds,
            capsize=4,
            color=colors[method],
            edgecolor="#263238",
            linewidth=0.6,
            label=labels[method],
            alpha=0.92,
        )
    ax0.set_xticks(list(x), [video_labels[v] for v in videos], fontproperties=font)
    ax0.set_ylabel("平均奖励", fontproperties=font)
    ax0.set_ylim(0, 1.03)
    ax0.grid(axis="y", color="#D9DEE8", linewidth=0.8, alpha=0.9)
    ax0.set_axisbelow(True)
    ax0.legend(prop=font, frameon=False, loc="upper right")
    ax0.set_title("不同视频配置下的平均 QoE", fontproperties=font, fontsize=12)

    ax1 = fig.add_subplot(gs[0, 1])
    v2 = df[df["video"] == "video2"].pivot(index="seed", columns="method", values="mean_reward").sort_index()
    seeds = [str(s) for s in v2.index]
    ax1.plot(seeds, v2["genet"], marker="o", color=colors["genet"], linewidth=2.0, label="Genet")
    ax1.plot(seeds, v2["final"], marker="o", color=colors["final"], linewidth=2.0, label="最终模型")
    for seed, row in v2.iterrows():
        delta = row["final"] - row["genet"]
        ax1.text(
            str(seed),
            max(row["final"], row["genet"]) + 0.025,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2F3A45",
        )
    ax1.set_ylabel("平均奖励", fontproperties=font)
    ax1.set_xlabel("随机种子", fontproperties=font, labelpad=8)
    ax1.set_ylim(0.30, 0.84)
    ax1.grid(axis="y", color="#D9DEE8", linewidth=0.8, alpha=0.9)
    ax1.set_axisbelow(True)
    ax1.legend(prop=font, frameon=False, loc="lower left")
    ax1.set_title("video2 上的 seed 级差异", fontproperties=font, fontsize=12)

    note = comp[comp["video"] == "video2"].iloc[0]
    fig.text(
        0.50,
        0.018,
        f"video2: 逐轨迹胜率 {note['win_rate'] * 100:.1f}%，平均增益 {note['mean_gain']:.4f}，中位数增益 {note['median_gain']:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontproperties=font,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    png_path = FIG_DIR / "fig5_7_video2_cross_video_comparison.png"
    pdf_path = FIG_DIR / "fig5_7_video2_cross_video_comparison.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
