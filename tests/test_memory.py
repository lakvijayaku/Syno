"""
Unit tests for syno.brain.memory.

These tests are intentionally stricter than the Memory Store's own
safeguards. They verify the step 7a example, that the oldest experience is
forgotten first, that sampling covers every stored experience and only stored
experiences, and that invalid use fails loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import random
import unittest

from syno.brain.memory import MemoryStore


class TestMemoryStore(unittest.TestCase):
    """Verifies storing, forgetting, and recalling experiences."""

    def test_step_7a_example(self):
        memory = MemoryStore(3)
        for item in ["a", "b", "c", "d", "e"]:
            memory.store(item)
        self.assertEqual(memory.experiences, ["c", "d", "e"])
        self.assertEqual(len(memory), 3)
        random.seed(0)
        self.assertEqual([memory.sample() for _ in range(5)], ["d", "d", "c", "d", "e"])

    def test_new_memory_is_empty(self):
        memory = MemoryStore(5)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.experiences, [])

    def test_length_grows_until_capacity_then_stays(self):
        memory = MemoryStore(4)
        lengths = []
        for item in range(10):
            memory.store(item)
            lengths.append(len(memory))
        self.assertEqual(lengths, [1, 2, 3, 4, 4, 4, 4, 4, 4, 4])

    def test_oldest_is_forgotten_first(self):
        memory = MemoryStore(3)
        for item in range(100):
            memory.store(item)
        self.assertEqual(memory.experiences, [97, 98, 99])

    def test_capacity_one_keeps_only_the_latest(self):
        memory = MemoryStore(1)
        memory.store("first")
        memory.store("second")
        self.assertEqual(memory.experiences, ["second"])
        self.assertEqual(memory.sample(), "second")

    def test_stores_real_experiences_unchanged(self):
        experience = ([0.0, 1.0, -1.0], 4, 0.25, [0.0, 0.0, -1.0])
        memory = MemoryStore(10)
        memory.store(experience)
        self.assertEqual(memory.sample(), experience)

    def test_sample_only_returns_stored_experiences(self):
        memory = MemoryStore(3)
        for item in range(6):
            memory.store(item)
        random.seed(1)
        for _ in range(200):
            self.assertIn(memory.sample(), [3, 4, 5])

    def test_sample_reaches_every_stored_experience(self):
        memory = MemoryStore(5)
        for item in range(5):
            memory.store(item)
        random.seed(2)
        self.assertEqual({memory.sample() for _ in range(300)}, {0, 1, 2, 3, 4})

    def test_sample_does_not_remove_the_experience(self):
        memory = MemoryStore(3)
        memory.store("a")
        memory.sample()
        self.assertEqual(len(memory), 1)

    def test_empty_memory_cannot_be_sampled(self):
        with self.assertRaises(ValueError):
            MemoryStore(3).sample()

    def test_capacity_below_one_raises_value_error(self):
        for capacity in (0, -1):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    MemoryStore(capacity)


if __name__ == "__main__":
    unittest.main()
