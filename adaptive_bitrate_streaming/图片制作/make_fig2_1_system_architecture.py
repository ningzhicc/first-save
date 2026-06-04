from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")
OUT_DIR = ROOT / "论文书写" / "论文图片"
FONT_PATH = ROOT / "图片制作" / "fonts" / "NotoSansCJKsc-Regular.otf"


W, H = 3800, 2350
BG = "#F6F8FB"
INK = "#132238"
MUTED = "#52627A"
LINE = "#6E7F99"
WHITE = "#FFFFFF"


PALETTE = {
    "input": ("#EAF4FF", "#3B82F6"),
    "env": ("#ECFDF5", "#10B981"),
    "baseline": ("#FFF7ED", "#F97316"),
    "pool": ("#F5F3FF", "#8B5CF6"),
    "plm": ("#EEF2FF", "#4F46E5"),
    "result": ("#F0FDFA", "#14B8A6"),
    "paper": ("#FEF2F2", "#EF4444"),
    "note": ("#F8FAFC", "#64748B"),
}


def font(size, bold=False):
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


F_TITLE = font(64)
F_SUBTITLE = font(50)
F_HEAD = font(45)
F_BODY = font(42)
F_SMALL = font(29)
F_TINY = font(22)
F_CHIP = font(45)




def wrap_mixed_text(text, max_chars):
    lines = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        # Chinese text wraps reasonably by character count; keep ASCII path snippets readable.
        buf = ""
        width = 0
        for ch in raw:
            inc = 1.0 if ord(ch) > 127 else 0.62
            if width + inc > max_chars and buf:
                lines.append(buf)
                buf = ch
                width = inc
            else:
                buf += ch
                width += inc
        if buf:
            lines.append(buf)
    return lines


def draw_round_box(draw, xy, title, body, kind, title_suffix=None, max_chars=18):
    x1, y1, x2, y2 = xy
    fill, accent = PALETTE[kind]
    # soft shadow
    draw.rounded_rectangle((x1 + 8, y1 + 12, x2 + 8, y2 + 12), radius=24, fill="#DDE4EF")
    draw.rounded_rectangle(xy, radius=24, fill=WHITE, outline="#D7DFEA", width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 78), radius=24, fill=fill, outline="#D7DFEA", width=0)
    draw.rectangle((x1, y1 + 52, x2, y1 + 78), fill=fill)
    draw.rounded_rectangle((x1 + 24, y1 + 21, x1 + 56, y1 + 53), radius=10, fill=accent)
    draw.text((x1 + 74, y1 + 10), title, fill=INK, font=F_HEAD)
    if title_suffix:
        draw.text((x2 - 22 - draw.textlength(title_suffix, font=F_SMALL), y1 + 21), title_suffix, fill=accent, font=F_SMALL)
    body_y = y1 + 100
    for line in wrap_mixed_text(body, max_chars):
        draw.text((x1 + 28, body_y), line, fill=MUTED if line.startswith(("•", "→")) else INK, font=F_BODY)
        body_y += 48


def draw_chip(draw, xy, text, color):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=24, fill="#FFFFFF", outline=color, width=3)
    bbox = draw.textbbox((0, 0), text, font=F_CHIP)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) / 2 - bbox[0]
    ty = y1 + (y2 - y1 - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill=color, font=F_CHIP)


def arrow(draw, start, end, color=LINE, width=6, curve=0):
    x1, y1 = start
    x2, y2 = end
    if curve == 0:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
    else:
        # orthogonal elbow connector with arrow on final segment
        midx = x1 + curve
        draw.line((x1, y1, midx, y1, midx, y2, x2, y2), fill=color, width=width, joint="curve")
        angle = math.atan2(0, x2 - midx)
    size = 22
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - math.pi / 7), y2 - size * math.sin(angle - math.pi / 7))
    p3 = (x2 - size * math.cos(angle + math.pi / 7), y2 - size * math.sin(angle + math.pi / 7))
    draw.polygon([p1, p2, p3], fill=color)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Background panel. The figure title is supplied by the thesis caption, so
    # the image itself only keeps the architecture content.
    draw.rounded_rectangle((80, 80, W - 80, H - 90), radius=40, fill="#FFFFFF", outline="#E2E8F0", width=4)

    # Compact boxes with larger text.
    input_box = (150, 275, 760, 720)
    env_box = (930, 275, 1560, 720)
    baseline_box = (930, 845, 1560, 1190)
    pool_box = (930, 1310, 1560, 1655)
    data_box = (1760, 275, 2390, 615)
    encoder_box = (1760, 740, 2390, 1250)
    policy_box = (1760, 1375, 2390, 1760)
    test_box = (2780, 345, 3405, 760)
    result_box = (2780, 930, 3405, 1275)
    paper_box = (2780, 1445, 3405, 1760)

    draw_round_box(draw, input_box, "输入数据与实验配置", "FCC 网络轨迹\n视频分片配置\n随机种子与测试协议\n基座模型与训练参数", "input", "输入", 16)
    draw_round_box(draw, env_box, "ABR 仿真环境", "接收码率动作\n更新下载时延、缓冲区\n计算重缓冲与 QoE 奖励", "env", "环境", 16)
    draw_round_box(draw, baseline_box, "基线方法评测", "BBA、MPC\nUDR、Genet\n统一协议下形成对照结果", "baseline", "对照", 16)
    draw_round_box(draw, pool_box, "经验池构造", "记录状态、动作、奖励、回报\n生成离线训练样本\n支撑后续大模型训练", "pool", "数据", 16)
    draw_round_box(draw, data_box, "训练样本组织", "按时间窗口切分轨迹\n组织状态序列与决策标签\n保持训练/验证/测试划分", "pool", "样本", 16)
    draw_round_box(draw, encoder_box, "状态编码与语义适配", "直接数值接入对照\n语义重编程\n上下文预对齐\n轻量化吞吐历史多尺度建模", "plm", "核心", 16)
    draw_round_box(draw, policy_box, "大模型策略学习", "Llama-3.2-1B + LoRA\n策略头输出码率动作\n按验证回报选择模型", "plm", "训练", 16)
    draw_round_box(draw, test_box, "在线验证与测试", "主测试与补充实验\n逐分片选择码率\n与仿真环境闭环交互", "result", "测试", 16)
    draw_round_box(draw, result_box, "结果统计", "平均奖励、平均码率\n重缓冲时间、平滑性惩罚\n多 seed 均值与方差\nBootstrap 与胜率", "result", "指标", 16)
    draw_round_box(draw, paper_box, "论文图表输出", "主结果与消融表\n稳定性和分布分析\n同源验证与跨视频分析", "paper", "输出", 16)

    # Main arrows, close to first version but cleaner.
    arrow(draw, (760, 495), (930, 495), "#3B82F6", 7)
    arrow(draw, (1245, 720), (1245, 845), "#10B981", 7)
    arrow(draw, (1245, 1190), (1245, 1310), "#8B5CF6", 7)
    arrow(draw, (1560, 1483), (1760, 445), "#8B5CF6", 6, curve=90)
    arrow(draw, (2075, 615), (2075, 740), "#4F46E5", 7)
    arrow(draw, (2075, 1250), (2075, 1375), "#4F46E5", 7)
    arrow(draw, (2390, 1565), (2780, 555), "#4F46E5", 7, curve=145)
    arrow(draw, (3092, 760), (3092, 930), "#14B8A6", 7)
    arrow(draw, (3092, 1275), (3092, 1445), "#EF4444", 7)
    arrow(draw, (1560, 1103), (2780, 1103), "#F97316", 6, curve=230)

    # Testing loop back to the same environment.
    draw.line((2780, 475, 2600, 475, 2600, 235, 1245, 235, 1245, 275), fill="#94A3B8", width=5)
    draw.polygon([(1245, 275), (1228, 245), (1262, 245)], fill="#94A3B8")
    draw.text((1560, 195), "主测试与补充实验复用统一仿真环境，保证不同方法公平比较", fill="#64748B", font=F_SMALL)

    # Bottom principles.
    chip_y = 1900
    chips = [
        ("统一测试协议", "#2563EB"),
        ("固定轨迹顺序", "#0891B2"),
        ("五随机种子", "#7C3AED"),
        ("日志复算统计", "#DC2626"),
        ("补充泛化验证", "#059669"),
    ]
    x = 345
    for text, color in chips:
        draw_chip(draw, (x, chip_y, x + 500, chip_y + 70), text, color)
        x += 610


    png_path = OUT_DIR / "fig2_1_system_architecture_cn.png"
    pdf_path = OUT_DIR / "fig2_1_system_architecture_cn.pdf"
    img.save(png_path, dpi=(300, 300))
    img.save(pdf_path, "PDF", resolution=300)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
