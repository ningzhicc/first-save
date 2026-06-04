from pathlib import Path

from PIL import Image, ImageDraw

from make_chapter3_method_figures import (
    BG,
    COLORS,
    F_BODY,
    F_HEAD,
    F_SMALL,
    F_TINY,
    F_TITLE,
    INK,
    MUTED,
    WHITE,
    arrow,
    rounded_box,
    save,
    text_size,
)


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")


def centered(draw: ImageDraw.ImageDraw, box, text, font, fill=INK):
    x1, y1, x2, y2 = box
    tw, th = text_size(draw, text, font)
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, fill=fill, font=font)


def token_chip(draw: ImageDraw.ImageDraw, x, y, w, h, text, fill, outline, text_fill=INK):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=fill, outline=outline, width=3)
    tw, th = text_size(draw, text, F_SMALL)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2), text, fill=text_fill, font=F_SMALL)


def draw_mini_series(draw: ImageDraw.ImageDraw, box, values, color="#16A34A", show_labels=True):
    x1, y1, x2, y2 = box
    pad_x = 42
    pad_y = 32
    chart_x1 = x1 + pad_x
    chart_y1 = y1 + pad_y
    chart_x2 = x2 - pad_x
    chart_y2 = y2 - pad_y
    n = len(values)
    max_v = max(values)
    min_v = min(values)
    rng = max(max_v - min_v, 1e-6)

    draw.line((chart_x1, chart_y2, chart_x2, chart_y2), fill="#CBD5E1", width=3)
    draw.line((chart_x1, chart_y1, chart_x1, chart_y2), fill="#CBD5E1", width=3)

    points = []
    for idx, value in enumerate(values):
        x = chart_x1 + idx * (chart_x2 - chart_x1) / max(n - 1, 1)
        y = chart_y2 - (value - min_v) * (chart_y2 - chart_y1) / rng
        points.append((x, y))

    for idx in range(len(points) - 1):
        draw.line((points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1]), fill=color, width=7)

    for idx, (x, y) in enumerate(points):
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE, outline=color, width=4)
        if show_labels:
            label = f"t-{n - 1 - idx}" if idx < n - 1 else "t"
            tw, _ = text_size(draw, label, F_SMALL)
            draw.text((x - tw / 2, chart_y2 + 12), label, fill=MUTED, font=F_SMALL)


def draw_grouped_windows(draw: ImageDraw.ImageDraw, box, values, groups, color="#7C3AED"):
    x1, y1, x2, y2 = box
    inner_pad = 12
    total_w = x2 - x1 - inner_pad * 2
    group_gap = 18
    chip_w = (total_w - group_gap * (len(groups) - 1)) / len(groups)
    chip_h = 156
    top_y = y1 + 8
    avg_y = y1 + 188

    for g_idx, group in enumerate(groups):
        gx = x1 + inner_pad + g_idx * (chip_w + group_gap)
        draw.rounded_rectangle((gx, top_y, gx + chip_w, top_y + chip_h), radius=24, fill="#F5F3FF", outline=color, width=3)
        labels = "  ".join(str(values[i]) for i in group)
        centered(draw, (gx + 10, top_y + 16, gx + chip_w - 10, top_y + 72), labels, F_SMALL, fill=INK)
        centered(draw, (gx + 12, top_y + 82, gx + chip_w - 12, top_y + 136), "局部窗口", F_TINY, fill=MUTED)
        avg = sum(values[i] for i in group) / len(group)
        avg_box = (gx + 10, avg_y, gx + chip_w - 10, avg_y + 90)
        draw.rounded_rectangle(avg_box, radius=20, fill=WHITE, outline="#C4B5FD", width=3)
        centered(draw, avg_box, f"均值 {avg:.1f}", F_SMALL, fill="#6D28D9")


def main():
    w, h = 3800, 2200
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((80, 90, w - 80, h - 90), radius=42, fill=WHITE, outline="#E2E8F0", width=4)

    title = "轻量化历史多尺度建模"
    tw, _ = text_size(draw, title, F_TITLE)
    draw.text(((w - tw) / 2, 92), title, fill=INK, font=F_TITLE)
    subtitle = "仅对吞吐历史做双尺度建模，提取短时波动与局部趋势"
    stw, _ = text_size(draw, subtitle, F_SMALL)
    draw.text(((w - stw) / 2, 175), subtitle, fill=MUTED, font=F_SMALL)

    rounded_box(
        draw,
        (140, 330, 760, 1750),
        "历史吞吐量序列",
        "输入为最近 6 步吞吐历史\n仅处理吞吐通道",
        "green",
        max_units=14,
        title_align="center",
        body_align="center",
    )
    draw_mini_series(draw, (200, 620, 700, 1010), [3.8, 2.9, 3.1, 2.2, 2.6, 1.9], color="#16A34A")
    draw.rounded_rectangle((205, 1105, 655, 1228), radius=22, fill="#F0FDF4", outline="#22C55E", width=3)
    centered(draw, (230, 1120, 630, 1172), "短时起伏明显", F_SMALL, fill="#166534")
    centered(draw, (230, 1170, 630, 1220), "整体趋势下降", F_SMALL, fill="#166534")
    draw.rounded_rectangle((190, 1290, 710, 1595), radius=24, fill="#F8FAFC", outline="#D8E1EF", width=3)
    centered(draw, (225, 1322, 675, 1382), "输入长度：6", F_SMALL)
    centered(draw, (225, 1410, 675, 1470), "只保留 2 个尺度：6 → 3", F_SMALL)
    centered(draw, (225, 1498, 675, 1558), "不直接预测未来带宽", F_SMALL, fill=MUTED)

    rounded_box(
        draw,
        (930, 330, 1650, 980),
        "细粒度尺度",
        "保留原始 6 步序列\n刻画相邻分片的快速波动",
        "cyan",
        max_units=15,
        title_align="center",
        body_align="center",
    )
    draw_mini_series(draw, (1000, 610, 1585, 900), [3.8, 2.9, 3.1, 2.2, 2.6, 1.9], color="#0891B2", show_labels=False)

    rounded_box(
        draw,
        (930, 1110, 1650, 1750),
        "粗粒度尺度",
        "把 6 步聚合为 3 个局部窗口\n提取短窗口趋势信息",
        "purple",
        max_units=16,
        title_align="center",
        body_align="center",
    )
    draw_grouped_windows(draw, (960, 1392, 1625, 1708), [3.8, 2.9, 3.1, 2.2, 2.6, 1.9], [(0, 1), (2, 3), (4, 5)], color="#7C3AED")

    rounded_box(
        draw,
        (1830, 520, 2540, 990),
        "多尺度特征融合",
        "",
        "orange",
        max_units=13,
        title_align="center",
        body_align="center",
    )
    centered(draw, (1880, 660, 2490, 715), "融合细粒度波动特征", F_BODY, fill=INK)
    centered(draw, (1880, 735, 2490, 790), "与短窗口趋势特征", F_BODY, fill=INK)
    token_chip(draw, 1885, 860, 260, 72, "细粒度波动", "#ECFEFF", "#0891B2", text_fill="#0F766E")
    token_chip(draw, 2210, 860, 260, 72, "短窗口趋势", "#F5F3FF", "#7C3AED", text_fill="#6D28D9")

    rounded_box(
        draw,
        (1830, 1170, 2540, 1750),
        "门控注入",
        "",
        "green",
        max_units=13,
        title_align="center",
        body_align="center",
    )
    centered(draw, (1880, 1360, 2490, 1415), "以保守门控方式", F_BODY, fill=INK)
    centered(draw, (1880, 1435, 2490, 1490), "把多尺度信息注入", F_BODY, fill=INK)
    centered(draw, (1880, 1510, 2490, 1565), "原始吞吐表示", F_BODY, fill=INK)
    token_chip(draw, 1895, 1625, 255, 72, "原始吞吐表示", "#ECFDF3", "#16A34A", text_fill="#166534")
    centered(draw, (2165, 1620, 2225, 1700), "+", F_HEAD, fill=INK)
    token_chip(draw, 2240, 1625, 265, 72, "门控残差信息", "#F0FDF4", "#22C55E", text_fill="#15803D")

    rounded_box(
        draw,
        (2720, 520, 3600, 980),
        "增强后的吞吐表示",
        "",
        "red",
        max_units=13,
        title_align="center",
        body_align="center",
    )
    centered(draw, (2780, 655, 3540, 710), "保留原始吞吐语义", F_BODY, fill=INK)
    centered(draw, (2780, 730, 3540, 785), "补充多尺度变化信息", F_BODY, fill=INK)
    token_chip(draw, 2825, 855, 670, 78, "送入状态表示与后续码率决策", "#FFF1F2", "#E11D48", text_fill="#BE123C")

    rounded_box(
        draw,
        (2720, 1180, 3600, 1750),
        "作用",
        "",
        "yellow",
        max_units=13,
        title_align="center",
        body_align="center",
    )
    centered(draw, (2780, 1370, 3540, 1425), "同时感知短时带宽波动", F_BODY, fill=INK)
    centered(draw, (2780, 1445, 3540, 1500), "与短窗口变化趋势", F_BODY, fill=INK)
    centered(draw, (2780, 1520, 3540, 1575), "提升码率决策稳定性", F_BODY, fill=INK)

    arrow(draw, (760, 950), (930, 665), "#16A34A", width=6)
    arrow(draw, (760, 1110), (930, 1430), "#16A34A", width=6)
    arrow(draw, (1650, 700), (1830, 720), "#0891B2", width=6)
    arrow(draw, (1650, 1430), (1830, 805), "#7C3AED", width=6)
    arrow(draw, (2190, 990), (2190, 1170), "#F97316", width=6)
    arrow(draw, (2540, 1460), (2720, 760), "#16A34A", width=6)
    arrow(draw, (3160, 980), (3160, 1180), "#E11D48", width=6)

    draw.rounded_rectangle((180, 1875, 3620, 2040), radius=28, fill="#F8FAFC", outline="#D8E1EF", width=3)
    draw.text((245, 1926), "实现要点：只对吞吐历史建模，采用轻量化双尺度结构，并通过门控方式注入原始表示，而不是单独做未来带宽预测。", fill=INK, font=F_BODY)

    save(img, "fig3_4_lightweight_multiscale_history_cn")


if __name__ == "__main__":
    main()
