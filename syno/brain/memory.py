# ── SYNO · syno/brain/memory.py ─────────────────────────
# The Memory Store: past experiences to learn from again
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

class MemoryStore:
    """
    Holds SYNO's most recent experiences so they can be learned from again.

    Each experience is one step of a life: (state, action, reward,
    next_state), exactly what learn needs. When the store is full, the oldest
    experience is forgotten.
    """

    def __init__(self, capacity: int):
        """
        Initializes an empty memory.

        :param capacity: The most experiences the store can hold. Must be at least 1.
        :raises ValueError: If capacity is less than 1.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1.")
        self.capacity = capacity
        self.experiences = []

    def store(self, experience: tuple) -> None:
        """
        Remembers one experience, forgetting the oldest if the store is full.

        :param experience: The experience to remember.
        """
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def sample(self) -> tuple:
        """
        Recalls one experience at random.

        Random recall mixes old lessons in with new ones, so recent
        experiences do not overwrite what was learned earlier.

        :return: A randomly chosen stored experience.
        :raises ValueError: If the store is empty.
        """
        if not self.experiences:
            raise ValueError("Memory store is empty.")
        return random.choice(self.experiences)

    def __len__(self) -> int:
        """
        Lets len(memory) report how many experiences are stored.

        :return: The number of stored experiences.
        """
        return len(self.experiences)
