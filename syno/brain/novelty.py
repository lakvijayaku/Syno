# ── SYNO · syno/brain/novelty.py ────────────────────────
# The Novelty System: curiosity that fades with familiarity
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

class NoveltySystem:
    """
    Gives SYNO a reward for visiting squares it has not seen lately.

    Each square has a familiarity from 0.0 (new) to 1.0 (completely
    familiar). Visiting a square makes it more familiar, so the reward fades
    with repeated visits (habituation). Every tick, all familiarity fades, so
    a square becomes interesting again after time away (recovery).
    """

    def __init__(self, scale: float, habituation: float, recovery: float):
        """
        Initializes the Novelty System.

        :param scale: The reward for visiting a completely new square. Must be at least 0.
        :param habituation: How much more familiar a square becomes per visit, from 0.0 to 1.0.
        :param recovery: How much familiarity remains after each tick, from 0.0 to 1.0.
        :raises ValueError: If any parameter is outside its range.
        """
        if scale < 0.0 or not (0.0 <= habituation <= 1.0) or not (0.0 <= recovery <= 1.0):
            raise ValueError("Invalid configuration parameters for NoveltySystem.")
        self.scale = scale
        self.habituation = habituation
        self.recovery = recovery
        # Maps each visited position to its familiarity. Positions never
        # visited are not stored, and count as completely new.
        self.familiarity = {}

    def signal(self, position: tuple[int, int]) -> float:
        """
        Visits a position: returns its novelty reward, then makes it more familiar.

        :param position: The (x, y) position SYNO is on.
        :return: scale * (1 - familiarity), measured before this visit.
        """
        familiarity = self.familiarity.get(position, 0.0)
        bonus = self.scale * (1.0 - familiarity)
        self.familiarity[position] = familiarity + self.habituation * (1.0 - familiarity)
        return bonus

    def tick(self) -> None:
        """
        Lets every familiarity fade by one step, so curiosity recovers over time.
        """
        for position in self.familiarity:
            self.familiarity[position] *= self.recovery
