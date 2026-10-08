# ── SYNO · syno/tools/plot.py ───────────────────────────
# Draws recorded data as an SVG line chart
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import html

# Picture size in pixels, and the empty border around the plotted line.
WIDTH = 600
HEIGHT = 300
MARGIN = 40

def line_chart(values: list[float], path: str, title: str) -> None:
    """
    Draws values as a line chart and saves it as an SVG file, which any web
    browser can display.

    Values are spaced evenly from left to right. The lowest value is drawn at
    the bottom and the highest at the top, and both are labeled on the left.

    :param values: The values to plot, in order. At least two are required.
    :param path: Where to write the SVG file. An existing file is replaced.
    :param title: The title shown above the chart.
    :raises ValueError: If fewer than two values are given.
    """
    if len(values) < 2:
        raise ValueError("A line chart requires at least two points.")

    low = min(values)
    high = max(values)
    # When every value is equal, high - low would be zero, and the y formula
    # below would divide by zero. Widening the range draws a flat line.
    if low == high:
        high = low + 1

    # Convert each value to a pixel position. SVG's y-axis points down, so
    # the y formula flips it: larger values get smaller y numbers.
    points = []
    for i, value in enumerate(values):
        x = MARGIN + i * (WIDTH - 2 * MARGIN) / (len(values) - 1)
        y = HEIGHT - MARGIN - (value - low) / (high - low) * (HEIGHT - 2 * MARGIN)
        points.append(f"{x:.1f},{y:.1f}")

    points_str = " ".join(points)

    # The xmlns string must match exactly, or browsers do not treat the file
    # as SVG and show only its text.
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">',
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'  <text x="{MARGIN}" y="25" font-family="sans-serif" font-size="16">{html.escape(title)}</text>',
        f'  <text x="5" y="{MARGIN + 5}" font-family="sans-serif" font-size="12">{high:.2f}</text>',
        f'  <text x="5" y="{HEIGHT - MARGIN + 5}" font-family="sans-serif" font-size="12">{low:.2f}</text>',
        f'  <polyline points="{points_str}" fill="none" stroke="steelblue" stroke-width="2"/>',
        "</svg>"
    ]
    svg_text = "\n".join(svg_lines)

    with open(path, "w") as file:
        file.write(svg_text)
