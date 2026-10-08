# ── SYNO · syno/tools/stats.py ──────────────────────────
# Simple statistics for analyzing run data
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

def moving_average(values: list[float], window: int) -> list[float]:
    """
    Smooths values by replacing each one with the average of the window of
    values that ends at it.

    The result has len(values) - window + 1 values, because the first full
    window ends at position window - 1.

    :param values: The values to smooth, in order.
    :param window: How many values each average covers, from 1 to len(values).
    :return: The averages, in order.
    :raises ValueError: If window is less than 1 or greater than len(values).
    """
    if window < 1 or window > len(values):
        raise ValueError("Window must be at least 1 and at most len(values).")
    return [sum(values[i - window + 1 : i + 1]) / window for i in range(window - 1, len(values))]
