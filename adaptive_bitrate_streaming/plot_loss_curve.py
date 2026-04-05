#!/usr/bin/env python3

import argparse
import math
from pathlib import Path


def read_losses(path: Path) -> list[float]:
    with path.open() as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    result: list[float] = []
    running = 0.0
    start = 0
    for idx, value in enumerate(values):
        running += value
        if idx - start + 1 > window:
            running -= values[start]
            start += 1
        result.append(running / (idx - start + 1))
    return result


def downsample(values: list[float], target_points: int) -> list[tuple[int, float]]:
    if len(values) <= target_points:
        return list(enumerate(values))

    points: list[tuple[int, float]] = []
    bucket = len(values) / target_points
    for index in range(target_points):
        start = int(index * bucket)
        end = int((index + 1) * bucket)
        if end <= start:
            end = start + 1
        chunk = values[start:end]
        center = (start + end - 1) // 2
        points.append((center, sum(chunk) / len(chunk)))
    return points


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg(
    losses: list[float],
    output_path: Path,
    title: str,
    smooth_window: int,
    epoch_size: int | None,
) -> None:
    width = 1400
    height = 840
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    smooth_values = moving_average(losses, smooth_window)
    raw_points = downsample(losses, min(plot_width, 1800))
    smooth_points = downsample(smooth_values, min(plot_width, 1800))

    y_min = min(losses)
    y_max = max(losses)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_lo = max(0.0, y_min - y_pad)
    y_hi = y_max + y_pad
    x_max = max(1, len(losses) - 1)

    def map_x(index: int) -> float:
        return margin_left + (index / x_max) * plot_width

    def map_y(value: float) -> float:
        return margin_top + (1 - (value - y_lo) / (y_hi - y_lo)) * plot_height

    raw_svg_points = [(map_x(i), map_y(v)) for i, v in raw_points]
    smooth_svg_points = [(map_x(i), map_y(v)) for i, v in smooth_points]

    y_ticks = 6
    x_ticks = 8
    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".bg { fill: #ffffff; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #6b7280; stroke-width: 1.5; }",
        ".raw { fill: none; stroke: #93c5fd; stroke-width: 1.2; opacity: 0.55; }",
        ".smooth { fill: none; stroke: #dc2626; stroke-width: 2.6; }",
        ".epoch { stroke: #cbd5e1; stroke-width: 1; stroke-dasharray: 4 4; }",
        ".label { font-size: 13px; }",
        ".title { font-size: 26px; font-weight: bold; }",
        ".subtitle { font-size: 14px; fill: #4b5563; }",
        ".legend { font-size: 13px; }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="{margin_left}" y="34">{svg_escape(title)}</text>',
        (
            f'<text class="subtitle" x="{margin_left}" y="56">'
            f"steps={len(losses)}, min={min(losses):.4f}, max={max(losses):.4f}, "
            f"final={losses[-1]:.4f}, smooth_window={smooth_window}"
            "</text>"
        ),
    ]

    for tick in range(y_ticks + 1):
        value = y_lo + (y_hi - y_lo) * tick / y_ticks
        y = margin_top + plot_height - plot_height * tick / y_ticks
        svg_lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        svg_lines.append(f'<text class="label" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>')

    for tick in range(x_ticks + 1):
        index = int(x_max * tick / x_ticks)
        x = map_x(index)
        svg_lines.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}"/>')
        svg_lines.append(f'<text class="label" x="{x:.2f}" y="{height - 26}" text-anchor="middle">{index}</text>')

    if epoch_size and epoch_size > 0:
        epoch_count = math.ceil(len(losses) / epoch_size)
        for epoch in range(1, epoch_count):
            x = map_x(min(epoch * epoch_size, x_max))
            svg_lines.append(f'<line class="epoch" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}"/>')
        for epoch in range(epoch_count):
            mid_index = min(epoch * epoch_size + epoch_size // 2, x_max)
            x = map_x(mid_index)
            svg_lines.append(f'<text class="label" x="{x:.2f}" y="{height - 8}" text-anchor="middle">epoch {epoch}</text>')

    svg_lines.extend(
        [
            f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>',
            f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>',
            f'<polyline class="raw" points="{polyline(raw_svg_points)}"/>',
            f'<polyline class="smooth" points="{polyline(smooth_svg_points)}"/>',
            f'<text class="label" x="{width / 2:.2f}" y="{height - 48}" text-anchor="middle">training step</text>',
            f'<text class="label" x="22" y="{height / 2:.2f}" transform="rotate(-90 22 {height / 2:.2f})" text-anchor="middle">cross-entropy loss</text>',
            f'<line x1="{width - 250}" y1="34" x2="{width - 214}" y2="34" class="raw"/>',
            f'<text class="legend" x="{width - 205}" y="38">downsampled raw loss</text>',
            f'<line x1="{width - 250}" y1="58" x2="{width - 214}" y2="58" class="smooth"/>',
            f'<text class="legend" x="{width - 205}" y="62">moving average</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(svg_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a training loss curve to SVG.")
    parser.add_argument("input", type=Path, help="Path to train_losses.txt")
    parser.add_argument("output", type=Path, help="Path to output SVG")
    parser.add_argument("--title", default="Training Loss Curve")
    parser.add_argument("--smooth-window", type=int, default=500)
    parser.add_argument("--epoch-size", type=int, default=996)
    args = parser.parse_args()

    losses = read_losses(args.input)
    if not losses:
        raise SystemExit("No loss values found in input file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_svg(losses, args.output, args.title, args.smooth_window, args.epoch_size)


if __name__ == "__main__":
    main()
