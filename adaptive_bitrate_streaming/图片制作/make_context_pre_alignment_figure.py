from pathlib import Path

from PIL import Image, ImageDraw

from make_chapter3_method_figures import (
    BG,
    COLORS,
    F_BODY,
    F_HEAD,
    F_SMALL,
    F_TITLE,
    INK,
    LINE,
    MUTED,
    WHITE,
    arrow,
    rounded_box,
    save,
    text_size,
)


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")


def dashed_line(draw: ImageDraw.ImageDraw, start, end, color, width=5, dash=22, gap=14):
    x1, y1 = start
    x2, y2 = end
    total = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if total == 0:
        return
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    dist = 0.0
    while dist < total:
        seg_start = dist
        seg_end = min(dist + dash, total)
        sx = x1 + dx * seg_start
        sy = y1 + dy * seg_start
        ex = x1 + dx * seg_end
        ey = y1 + dy * seg_end
        draw.line((sx, sy, ex, ey), fill=color, width=width)
        dist += dash + gap


def token_chip(draw: ImageDraw.ImageDraw, x, y, w, h, text, fill, outline, text_fill=INK):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=fill, outline=outline, width=3)
    tw, th = text_size(draw, text, F_SMALL)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2), text, fill=text_fill, font=F_SMALL)


def token_stack(draw: ImageDraw.ImageDraw, x, y, labels, fill, outline):
    chip_w, chip_h = 188, 76
    gap_x, gap_y = 22, 20
    for idx, label in enumerate(labels):
        row = idx // 2
        col = idx % 2
        cx = x + col * (chip_w + gap_x)
        cy = y + row * (chip_h + gap_y)
        token_chip(draw, cx, cy, chip_w, chip_h, label, fill, outline)


def centered(draw: ImageDraw.ImageDraw, box, text, font, fill=INK):
    x1, y1, x2, y2 = box
    tw, th = text_size(draw, text, font)
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, fill=fill, font=font)


def main():
    w, h = 3600, 2050
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((80, 90, w - 80, h - 90), radius=42, fill=WHITE, outline="#E2E8F0", width=4)

    title = "上下文预对齐机制"
    tw, _ = text_size(draw, title, F_TITLE)
    draw.text(((w - tw) / 2, 92), title, fill=INK, font=F_TITLE)
    subtitle = "先利用历史反馈调整当前状态表示，再送入语义对齐与 PLM"
    stw, _ = text_size(draw, subtitle, F_SMALL)
    draw.text(((w - stw) / 2, 175), subtitle, fill=MUTED, font=F_SMALL)

    rounded_box(
        draw,
        (150, 310, 980, 950),
        "当前 ABR 状态",
        "6×6 数值状态先编码为 6 个状态 token",
        "blue",
        max_units=20,
        title_align="center",
        body_align="center",
    )
    token_stack(
        draw,
        280,
        520,
        ["上一码率", "缓冲区", "吞吐历史", "时延历史", "分片大小", "剩余分片"],
        "#EFF6FF",
        "#2563EB",
    )

    rounded_box(
        draw,
        (150, 1090, 500, 1450),
        "历史奖励",
        "上一决策的\nQoE 反馈",
        "yellow",
        max_units=9,
        title_align="center",
        body_align="center",
        body_valign="middle",
        body_font=F_HEAD,
    )

    rounded_box(
        draw,
        (620, 1090, 970, 1450),
        "历史动作",
        "上一分片的\n动作记录",
        "orange",
        max_units=9,
        title_align="center",
        body_align="center",
        body_valign="middle",
        body_font=F_HEAD,
    )

    rounded_box(
        draw,
        (1130, 350, 2470, 1510),
        "受限上下文预对齐",
        "把 6 个状态 token 与 2 个上下文 token 放在同一步内做注意力交互",
        "orange",
        max_units=26,
        title_align="center",
        body_align="center",
    )

    token_stack(
        draw,
        1290,
        600,
        ["状态1", "状态2", "状态3", "状态4", "状态5", "状态6"],
        "#EFF6FF",
        "#2563EB",
    )
    token_chip(draw, 2010, 690, 250, 84, "历史奖励", "#FEFCE8", "#CA8A04", text_fill="#A16207")
    token_chip(draw, 2010, 820, 250, 84, "历史动作", "#FFF7ED", "#F97316", text_fill="#C2410C")

    mask_box = (1300, 1060, 2290, 1380)
    draw.rounded_rectangle(mask_box, radius=24, fill="#FFF7ED", outline="#FDBA74", width=3)
    centered(draw, (1360, 1090, 2230, 1145), "奖励主导的受限读取", F_HEAD, fill="#C2410C")
    draw.text((1395, 1182), "• 状态 token 重点读取历史奖励", fill=INK, font=F_BODY)
    draw.text((1395, 1260), "• 历史动作在此步不直接更新状态", fill=INK, font=F_BODY)

    arrow(draw, (980, 630), (1130, 760), "#2563EB", width=6)
    arrow(draw, (500, 1270), (1130, 940), "#CA8A04", width=6)
    dashed_line(draw, (970, 1270), (1130, 1060), "#F97316", width=6)
    draw.line((1065, 1090, 1110, 1135), fill="#DC2626", width=8)
    draw.line((1110, 1090, 1065, 1135), fill="#DC2626", width=8)
    centered(draw, (970, 1140, 1205, 1195), "受限连接", F_SMALL, fill="#DC2626")

    draw.text((1805, 695), "重点读取", fill="#A16207", font=F_SMALL)
    draw.text((1800, 826), "不直接读取", fill="#C2410C", font=F_SMALL)

    rounded_box(
        draw,
        (2660, 350, 3450, 920),
        "输出状态 token",
        "",
        "green",
        max_units=18,
        title_align="center",
        body_align="center",
        body_valign="middle",
    )
    centered(draw, (2710, 455, 3400, 520), "只保留更新后的 6 个状态 token", F_SMALL)
    token_stack(
        draw,
        2815,
        585,
        ["状态1'", "状态2'", "状态3'", "状态4'", "状态5'", "状态6'"],
        "#ECFDF3",
        "#16A34A",
    )

    rounded_box(
        draw,
        (2660, 1070, 3450, 1450),
        "后续处理",
        "语义对齐\n→ PLM 主体\n→ 下一分片码率预测",
        "purple",
        max_units=12,
        title_align="center",
        body_align="center",
        body_valign="middle",
        body_font=F_HEAD,
    )

    arrow(draw, (2470, 790), (2660, 635), "#16A34A", width=6)
    arrow(draw, (3055, 920), (3055, 1070), "#7C3AED", width=6)

    draw.rounded_rectangle((180, 1640, 3420, 1865), radius=28, fill="#F8FAFC", outline="#D8E1EF", width=3)
    draw.text((245, 1695), "主线解释：预对齐阶段重点利用上一决策的奖励反馈调整当前状态表示；", fill=INK, font=F_BODY)
    draw.text((245, 1768), "历史动作不直接更新状态 token，而是继续保留在主序列中作为动作历史条件。", fill=INK, font=F_BODY)

    # Show action history still goes to the main sequence branch.
    dashed_line(draw, (970, 1365), (2700, 1540), "#F97316", width=5, dash=26, gap=18)
    draw.rounded_rectangle((2620, 1490, 3400, 1595), radius=22, fill=WHITE, outline="#FDBA74", width=3)
    centered(draw, (2645, 1510, 3375, 1575), "历史动作仍保留在主序列中", F_SMALL, fill="#C2410C")

    save(img, "fig3_3a_context_pre_alignment_maskprevreward_cn")


if __name__ == "__main__":
    main()
