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

    def sense(self, position: tuple[int, int], radius: int) -> list[float]:
        """
        Reports how familiar every square in the window around a position is,
        without visiting any of them.

        Values are ordered row by row from the top-left of the window, the
        same order as Habitat.sense, so each familiarity lines up with what
        SYNO sees on that square.

        :param position: The (x, y) position at the center of the window.
        :param radius: How many squares to report in each direction.
        :return: A list of (2 * radius + 1) ** 2 values, from 0.0 (new) to
            1.0 (completely familiar).
        :raises ValueError: If radius is negative.
        """
        if radius < 0:
            raise ValueError("Radius must be at least 0.")

        x, y = position
        senses = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # A square never visited counts as completely new. This also
                # covers squares off the grid, which SYNO's eyes already show
                # as walls. Only reading with get means that looking at a
                # square never counts as visiting it.
                senses.append(self.familiarity.get((x + dx, y + dy), 0.0))
        return senses

    def tick(self) -> None:
        """
        Lets every familiarity fade by one step, so curiosity recovers over time.
        """
        for position in self.familiarity:
            self.familiarity[position] *= self.recovery
