# ── SYNO · syno/brain/loss.py ───────────────────────────
# Measures how wrong the network's outputs are
# Laksheth Vijayakumar · 2026-10-06 · GPL-3.0
# ────────────────────────────────────────────────────────

def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    """
    Calculates the mean squared error between predictions and targets.

    :param predictions: The network's output values (floats).
    :param targets: The correct values (floats), one per prediction.
    :return: The average of the squared differences. 0.0 means a perfect match.
    :raises ValueError: If the lists are empty or have different lengths.
    """
    # Guard clause: reject empty input before any math, so the average below
    # can never divide by zero.
    if not predictions:
        raise ValueError("Predictions and targets must not be empty.")

    # Pair each prediction with its target, then square each difference.
    # Squaring removes the sign and punishes large errors more than small ones.
    # strict=True makes zip raise a ValueError when the lengths differ.
    try:
        squared_errors = [ (p - t)**2 for p, t in zip(predictions, targets, strict=True) ]
    except ValueError as e:
        raise ValueError(
            f"Mismatched lengths. Got {len(predictions)} predictions and {len(targets)} targets."
        ) from e

    # The average keeps the loss independent of the number of outputs.
    return sum(squared_errors) / len(squared_errors)
