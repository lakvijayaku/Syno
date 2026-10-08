# ── SYNO · syno/body/hormones.py ────────────────────────
# Slow, global chemical signals that build up over time
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

class Hormone:
    """
    A slow chemical signal whose level drifts gradually toward whatever
    signal it is fed, and lingers after the signal stops.

    Each update closes a fraction (rate) of the gap between the level and the
    signal, so the level is an exponential moving average of recent signals.
    A Hormone does not know what it represents; its meaning comes from the
    signal it is fed.
    """

    def __init__(self, rate: float):
        """
        Initializes a hormone with a level of 0.0.

        :param rate: The fraction of the gap closed per update, greater than 0
            and at most 1. Small rates make slow hormones that average many
            steps; a rate of 1.0 follows the signal instantly.
        :raises ValueError: If rate is not greater than 0 and at most 1.
        """
        if rate <= 0.0 or rate > 1.0:
            raise ValueError("Rate must be greater than 0 and at most 1.")
        self.rate = rate
        self.level = 0.0

    def update(self, signal: float) -> float:
        """
        Moves the level a fraction of the way toward the signal.

        :param signal: The current value feeding the hormone.
        :return: The new level.
        """
        self.level += self.rate * (signal - self.level)
        return self.level
