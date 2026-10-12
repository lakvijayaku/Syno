# ── SYNO · syno/brain/working_memory.py ─────────────────
# Working memory, a fading trace of what SYNO has seen
# Laksheth Vijayakumar · 2026-10-11 · GPL-3.0
# ────────────────────────────────────────────────────────

class WorkingMemory:
    """
    A fading trace of recent inputs, such as what SYNO has seen.

    Every update keeps a fraction (decay) of what was remembered and takes in
    the rest from the new values, so something seen once fades away over the
    following steps. It works like a Hormone, but for a list of values. In
    the brain, this resembles persistent activity in prefrontal neurons,
    which keep firing for a while after a stimulus disappears.
    """

    def __init__(self, size: int, decay: float):
        """
        Initializes an empty memory.

        :param size: How many values the memory holds. Must be at least 1.
        :param decay: The fraction of the old memory kept at each update,
            from 0.0 (no memory) up to, but not including, 1.0.
        :raises ValueError: If size is less than 1 or decay is out of range.
        """
        if size < 1:
            raise ValueError("Size must be at least 1.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Decay must be between 0.0 (inclusive) and 1.0 (exclusive).")
        self.size = size
        self.decay = decay
        self.contents = [0.0] * size

    def update(self, values: list[float]) -> list[float]:
        """
        Blends new values into the memory.

        :param values: The newest values, one for each slot.
        :return: A copy of the updated memory, so callers cannot change it.
        :raises ValueError: If the number of values does not match size.
        """
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values, got {len(values)}.")
        self.contents = [
            self.decay * old + (1.0 - self.decay) * new
            for old, new in zip(self.contents, values)
        ]
        return list(self.contents)

    def reset(self) -> None:
        """
        Empties the memory, as at the start of a new life.
        """
        self.contents = [0.0] * self.size
