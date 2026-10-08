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
    experience is forgotten. Each experience also has a priority, so
    surprising experiences can be recalled more often than ordinary ones.
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
        # priorities[i] belongs to experiences[i]. Both lists always have the
        # same length and order.
        self.priorities = []

    def store(self, experience: tuple, priority: float = 1.0) -> None:
        """
        Remembers one experience, forgetting the oldest if the store is full.

        :param experience: The experience to remember.
        :param priority: How strongly to favor this experience in
            sample_by_priority. Must be greater than 0.
        :raises ValueError: If priority is 0 or negative.
        """
        if priority <= 0.0:
            raise ValueError("Priority must be greater than 0.")
        self.experiences.append(experience)
        self.priorities.append(priority)
        if len(self.experiences) > self.capacity:
            # Forget the oldest experience and its priority together, so the
            # two lists stay aligned.
            self.experiences.pop(0)
            self.priorities.pop(0)

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

    def sample_by_priority(self) -> int:
        """
        Recalls the position of one experience, chosen with probability
        proportional to its priority.

        A position is returned rather than the experience, so the caller can
        update that experience's priority after learning from it.

        :return: The position of the chosen experience in experiences.
        :raises ValueError: If the store is empty.
        """
        if not self.experiences:
            raise ValueError("Memory store is empty.")
        return random.choices(range(len(self.experiences)), weights=self.priorities)[0]

    def set_priority(self, index: int, priority: float) -> None:
        """
        Changes how strongly one experience is favored in sample_by_priority.

        :param index: The position of the experience in experiences.
        :param priority: The new priority. Must be greater than 0.
        :raises ValueError: If index is out of range or priority is 0 or negative.
        """
        if not (0 <= index < len(self.experiences)):
            raise ValueError(f"Index {index} is out of bounds.")
        if priority <= 0.0:
            raise ValueError("Priority must be greater than 0.")
        self.priorities[index] = priority
