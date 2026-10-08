# ── SYNO · syno/brain/policy.py ─────────────────────────
# Chooses SYNO's next action
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

def choose_action(values: list[float], epsilon: float) -> int:
    """
    Chooses an action using the epsilon-greedy rule.

    With probability epsilon, a random action is chosen, so SYNO keeps
    exploring. Otherwise, the action with the highest value is chosen. This
    is a temporary, hard-coded exploration rule.

    :param values: How good SYNO currently believes each action is.
    :param epsilon: The probability of choosing a random action, from 0.0 to 1.0.
    :return: The position of the chosen action in values.
    :raises ValueError: If values is empty or epsilon is outside 0.0 to 1.0.
    """
    if not values:
        raise ValueError("The values list must not be empty.")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("Epsilon must be between 0.0 and 1.0 inclusive.")
    # Explore: try a random action, so that wrong beliefs can be discovered.
    if random.random() < epsilon:
        return random.randrange(len(values))
    # Exploit: choose the best-looking action. When values tie, the first
    # one wins.
    return values.index(max(values))
