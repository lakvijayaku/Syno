# ── SYNO · syno/brain/dopamine.py ───────────────────────
# Computes SYNO's reward prediction error (the dopamine signal)
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

def reward_prediction_error(
    reward: float,
    expected: float,
    next_expected: float,
    discount: float,
    done: bool
) -> float:
    """
    Calculates the reward prediction error: how much better or worse a step
    turned out than SYNO expected.

    A positive value means better than expected, a negative value means
    worse, and zero means the outcome was fully predicted.

    :param reward: The reward received during the step.
    :param expected: The value SYNO predicted before the step.
    :param next_expected: The value SYNO predicts after the step.
    :param discount: How much future reward counts, from 0.0 to 1.0.
    :param done: True if the episode ended with this step.
    :return: The reward prediction error.
    :raises ValueError: If discount is outside the range 0.0 to 1.0.
    """
    if not (0.0 <= discount <= 1.0):
        raise ValueError("Discount must be between 0.0 and 1.0 inclusive.")
    # When the episode has ended there is no future, so no future value is
    # added. Otherwise the next prediction is discounted, because reward one
    # step away is worth less than reward now.
    if done:
        future = 0.0
    else:
        future = discount * next_expected
    # What actually happened (reward plus discounted future) minus what was
    # predicted.
    return reward + future - expected
