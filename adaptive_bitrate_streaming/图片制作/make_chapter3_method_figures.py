from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")
OUT_DIR = ROOT / "论文书写" / "论文图片"
FONT_PATH = ROOT / "图片制作" / "fonts" / "NotoSansCJKsc-Regular.otf"

BG = "#F7F9FC"
INK = "#172033"
MUTED = "#5B677A"
LINE = "#7B8CA6"
WHITE = "#FFFFFF"

COLORS = {
    "blue": ("#EAF3FF", "#2563EB"),
    "green": ("#ECFDF3", "#16A34A"),
    "orange": ("#FFF4E6", "#F97316"),
    "purple": ("#F4F0FF", "#7C3AED"),
    "red": ("#FFF1F2", "#E11D48"),
    "cyan": ("#ECFEFF", "#0891B2"),
    "gray": ("#F3F6FA", "#64748B"),
    "yellow": ("#FEFCE8", "#CA8A04"),
}


def font(size: int):
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_PATH)
    return ImageFont.truetype(str(FONT_PATH), size=size)


F_TITLE = font(64)
F_HEAD = font(58)
F_BODY = font(48)
F_SMALL = font(42)
F_TINY = font(34)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(text: str, max_units: float) -> list[str]:
    lines = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        buf = ""
        width = 0.0
        for ch in raw:
            inc = 1.0 if ord(ch) > 127 else 0.58
            if buf and width + inc > max_units:
                lines.append(buf)
                buf = ch
                width = inc
            else:
                buf += ch
                width += inc
        if buf:
            lines.append(buf)
    return lines


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    color: str,
    max_units: float = 13,
    header=True,
    title_align: str = "left",
    body_align: str = "left",
    body_valign: str = "top",
    body_font: ImageFont.ImageFont = F_BODY,
):
    x1, y1, x2, y2 = xy
    fill, accent = COLORS[color]
    draw.rounded_rectangle((x1 + 8, y1 + 10, x2 + 8, y2 + 10), radius=26, fill="#DDE5F0")
    draw.rounded_rectangle(xy, radius=26, fill=WHITE, outline="#D4DDEB", width=3)
    if header:
        draw.rounded_rectangle((x1, y1, x2, y1 + 100), radius=26, fill=fill)
        draw.rectangle((x1, y1 + 68, x2, y1 + 100), fill=fill)
        draw.rounded_rectangle((x1 + 28, y1 + 30, x1 + 66, y1 + 68), radius=11, fill=accent)
        title_x = x1 + 88
        if title_align == "center":
            tw, _ = text_size(draw, title, F_HEAD)
            title_x = x1 + (x2 - x1 - tw) / 2
        draw.text((title_x, y1 + 16), title, fill=INK, font=F_HEAD)
        body_y = y1 + 128
        body_top = y1 + 100
    else:
        tw, th = text_size(draw, title, F_HEAD)
        draw.text((x1 + (x2 - x1 - tw) / 2, y1 + 24), title, fill=accent, font=F_HEAD)
        body_y = y1 + 110
        body_top = y1 + 88
    lines = wrap_text(body, max_units)
    line_step = 62
    if body_valign == "middle" and body:
        block_height = line_step * len(lines)
        body_y = body_top + (y2 - body_top - block_height) / 2
    for line in lines:
        tw, th = text_size(draw, line, body_font)
        text_x = x1 + 32
        if body_align == "center":
            text_x = x1 + (x2 - x1 - tw) / 2
        draw.text((text_x, body_y), line, fill=MUTED if line.startswith(("•", "→")) else INK, font=body_font)
        body_y += line_step


def chip(draw: ImageDraw.ImageDraw, xy, text, color, fnt=F_SMALL):
    x1, y1, x2, y2 = xy
    _, accent = COLORS[color]
    draw.rounded_rectangle(xy, radius=24, fill=WHITE, outline=accent, width=3)
    tw, th = text_size(draw, text, fnt)
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, fill=accent, font=fnt)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=LINE, width=7, elbow=None):
    x1, y1 = start
    x2, y2 = end
    if elbow is None:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
    else:
        xm, ym = elbow
        draw.line((x1, y1, xm, y1, xm, ym, x2, ym, x2, y2), fill=color, width=width, joint="curve")
        angle = math.atan2(y2 - ym, x2 - x2 if y2 != ym else x2 - xm)
        if y2 != ym:
            angle = math.atan2(y2 - ym, 0)
        else:
            angle = math.atan2(0, x2 - xm)
    size = 24
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - math.pi / 7), y2 - size * math.sin(angle - math.pi / 7))
    p3 = (x2 - size * math.cos(angle + math.pi / 7), y2 - size * math.sin(angle + math.pi / 7))
    draw.polygon([p1, p2, p3], fill=color)


def centered_title(draw, w, text):
    tw, th = text_size(draw, text, F_TITLE)
    draw.text(((w - tw) / 2, 60), text, fill=INK, font=F_TITLE)


def save(img: Image.Image, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    img.save(png, dpi=(300, 300))
    img.save(pdf, "PDF", resolution=300)
    print(png)
    print(pdf)


def fig31_overall_policy():
    w, h = 3600, 2050
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((80, 90, w - 80, h - 90), radius=42, fill=WHITE, outline="#E2E8F0", width=4)

    rounded_box(draw, (150, 245, 790, 780), "ABR 输入状态", "上一分片码率\n缓冲区长度\n吞吐与时延历史\n下一分片大小\n剩余分片数", "blue", 12)
    rounded_box(draw, (150, 940, 790, 1320), "历史反馈", "上一动作\n上一奖励\n回报条件", "orange", 12)

    rounded_box(draw, (940, 220, 1580, 600), "数值状态 token", "按状态变量组织\n保留特征边界\n形成分片级决策输入", "gray", 13)
    rounded_box(draw, (940, 820, 1580, 1200), "上下文 token", "动作反馈\n奖励反馈\n回报条件", "yellow", 13)

    rounded_box(draw, (1740, 180, 2390, 560), "语义重编程", "数值状态映射到\n任务语义表征空间", "purple", 14)
    rounded_box(draw, (1740, 690, 2390, 1070), "上下文预对齐", "当前状态读取历史反馈\n历史奖励作为条件信号", "orange", 14)
    rounded_box(draw, (1680, 1200, 2480, 1580), "轻量化多尺度历史建模", "提取吞吐历史中的\n局部波动与短窗口趋势", "green", 14)

    rounded_box(draw, (2580, 370, 3260, 790), "大语言模型主体", "基座模型 + LoRA\n上下文依赖建模\n生成决策表示", "cyan", 14)
    rounded_box(draw, (2580, 1040, 3260, 1415), "策略输出头", "动作概率分布\n选择下一分片码率档位", "red", 14)

    chip(draw, (1160, 1700, 1620, 1790), "离线经验池训练", "purple")
    chip(draw, (1710, 1700, 2170, 1790), "验证回报选模", "green")
    chip(draw, (2260, 1700, 2720, 1790), "在线仿真测试", "cyan")

    arrow(draw, (790, 515), (940, 410), "#2563EB")
    arrow(draw, (790, 1130), (940, 1010), "#F97316")
    arrow(draw, (1580, 410), (1740, 370), "#7C3AED")
    arrow(draw, (1580, 1010), (1740, 880), "#F97316")
    arrow(draw, (1580, 520), (1680, 1390), "#16A34A")
    arrow(draw, (2390, 370), (2580, 560), "#7C3AED")
    arrow(draw, (2390, 880), (2580, 620), "#F97316")
    arrow(draw, (2480, 1390), (2580, 700), "#16A34A")
    arrow(draw, (2920, 790), (2920, 1040), "#0891B2")
    save(img, "fig3_1_llm_abr_policy_overview_cn")


def fig32_semantic_reprogramming():
    w, h = 3600, 2050
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((80, 90, w - 80, h - 90), radius=42, fill=WHITE, outline="#E2E8F0", width=4)

    left_x, mid_x, right_x = 180, 1280, 2520
    y0 = 220
    features = [
        ("上一分片码率", "平滑性约束", "blue"),
        ("缓冲区长度", "卡顿风险", "red"),
        ("历史吞吐量", "网络传输能力", "green"),
        ("下载时延历史", "下载压力", "orange"),
        ("下一分片大小", "动作下载成本", "purple"),
        ("剩余分片数", "播放阶段信息", "cyan"),
    ]
    for i, (name, meaning, color) in enumerate(features):
        y = y0 + i * 235
        rounded_box(draw, (left_x, y, left_x + 820, y + 180), name, "", color, 12, header=False)
        chip(draw, (mid_x, y + 38, mid_x + 760, y + 132), f"语义锚点：{meaning}", color, F_SMALL)
        arrow(draw, (left_x + 820, y + 90), (mid_x, y + 85), COLORS[color][1], width=5)

    rounded_box(draw, (2280, 420, 3200, 850), "注意力式语义对齐", "数值 token 读取任务语义锚点\n形成具有 ABR 含义的状态表示", "purple", 18)
    rounded_box(draw, (2280, 1090, 3200, 1480), "大模型表征空间", "输入不再是裸数值\n而是携带状态角色与风险含义的嵌入", "cyan", 18)
    for i in range(len(features)):
        y = y0 + i * 235 + 85
        arrow(draw, (mid_x + 760, y), (2280, 635), "#7C3AED", width=4)
    arrow(draw, (2740, 850), (2740, 1090), "#0891B2")

    draw.rounded_rectangle((330, 1710, 3270, 1845), radius=30, fill="#F8FAFC", outline="#D8E1EF", width=3)
    draw.text((400, 1745), "作用：让模型区分带宽下降、缓冲不足、分片变大等不同风险来源。", fill=INK, font=F_SMALL)
    save(img, "fig3_2_semantic_reprogramming_encoder_cn")


def fig33_context_multiscale():
    w, h = 3800, 2300
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((80, 90, w - 80, h - 90), radius=42, fill=WHITE, outline="#E2E8F0", width=4)

    # Left: pre-alignment.
    rounded_box(draw, (170, 260, 810, 600), "当前状态 token", "缓冲区、码率\n分片大小、吞吐等", "blue", 12)
    rounded_box(draw, (170, 760, 810, 1085), "历史动作 token", "上一分片码率选择\n反映策略行为", "orange", 12)
    rounded_box(draw, (170, 1235, 810, 1560), "历史奖励 token", "上一决策 QoE 反馈\n反映动作效果", "yellow", 12)
    rounded_box(
        draw,
        (1010, 620, 1690, 1235),
        "受限上下文预对齐",
        "当前状态读取历史奖励\n历史动作保留为反馈记录\n避免反馈语义被反向污染",
        "orange",
        15,
        title_align="center",
        body_align="center",
        body_valign="middle",
        body_font=F_HEAD,
    )
    arrow(draw, (810, 430), (1010, 820), "#2563EB")
    arrow(draw, (810, 922), (1010, 930), "#F97316")
    arrow(draw, (810, 1398), (1010, 1040), "#CA8A04")

    # Right: multi-scale history modeling.
    rounded_box(draw, (1930, 260, 2600, 600), "吞吐历史序列", "最近若干分片的\n观测吞吐变化", "green", 13)
    rounded_box(draw, (1930, 760, 2600, 1085), "细粒度尺度", "相邻分片快速波动\n局部突发变化", "cyan", 13)
    rounded_box(draw, (1930, 1235, 2600, 1560), "粗粒度尺度", "短窗口整体趋势\n持续升降变化", "purple", 13)
    rounded_box(
        draw,
        (2820, 620, 3500, 1235),
        "趋势与扰动融合",
        "扰动信息自底向上传播\n趋势信息自顶向下传播\n形成多尺度历史表示",
        "green",
        15,
        title_align="center",
        body_align="center",
        body_valign="middle",
        body_font=F_HEAD,
    )
    arrow(draw, (2600, 430), (2820, 805), "#16A34A")
    arrow(draw, (2600, 922), (2820, 940), "#0891B2")
    arrow(draw, (2600, 1398), (2820, 1075), "#7C3AED")

    # Fusion.
    rounded_box(draw, (1300, 1660, 2520, 2050), "融合后的增强状态表示", "保留原始状态语义\n注入反馈条件与吞吐趋势\n供大语言模型和策略头使用", "red", 24)
    arrow(draw, (1350, 1235), (1660, 1660), "#F97316", elbow=(1350, 1490))
    arrow(draw, (3160, 1235), (2200, 1660), "#16A34A", elbow=(3160, 1605))
    save(img, "fig3_3_context_alignment_multiscale_history_cn")


def main():
    fig31_overall_policy()
    fig32_semantic_reprogramming()
    fig33_context_multiscale()


if __name__ == "__main__":
    main()
