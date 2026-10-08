"""
Unit tests for syno.brain.memory.

These tests are intentionally stricter than the Memory Store's own
safeguards. They verify the step 7a example, that the oldest experience is
forgotten first, that sampling covers every stored experience and only stored
experiences, that priority-weighted recall favors high priorities in the
right proportion, and that invalid use fails loudly.

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



class TestMemoryPriorities(unittest.TestCase):
    """Verifies priority-weighted recall."""

    def test_default_priority_is_one(self):
        memory = MemoryStore(3)
        memory.store("a")
        self.assertEqual(memory.priorities, [1.0])

    def test_priorities_stay_aligned_with_experiences_when_forgetting(self):
        memory = MemoryStore(3)
        for item in range(6):
            memory.store(item, item + 1.0)
        self.assertEqual(memory.experiences, [3, 4, 5])
        self.assertEqual(memory.priorities, [4.0, 5.0, 6.0])

    def test_step_7c_proportions(self):
        # Priorities 1.0 and 0.01: the surprising memory should be recalled
        # about 1.0 / 1.01, or 99%, of the time.
        memory = MemoryStore(3)
        memory.store("boring", 0.01)
        memory.store("surprising", 1.0)
        random.seed(0)
        picks = [memory.experiences[memory.sample_by_priority()] for _ in range(5000)]
        self.assertAlmostEqual(picks.count("surprising") / 5000, 1.0 / 1.01, delta=0.01)

    def test_equal_priorities_recall_evenly(self):
        memory = MemoryStore(4)
        for item in range(4):
            memory.store(item, 2.0)
        random.seed(1)
        counts = [0, 0, 0, 0]
        for _ in range(4000):
            counts[memory.sample_by_priority()] += 1
        for count in counts:
            self.assertAlmostEqual(count / 4000, 0.25, delta=0.03)

    def test_sample_by_priority_returns_a_valid_position(self):
        memory = MemoryStore(5)
        for item in range(3):
            memory.store(item, 0.5)
        random.seed(2)
        for _ in range(200):
            self.assertIn(memory.sample_by_priority(), range(3))

    def test_set_priority_changes_recall(self):
        memory = MemoryStore(2)
        memory.store("a", 1.0)
        memory.store("b", 1.0)
        memory.set_priority(0, 1000.0)
        random.seed(3)
        picks = [memory.sample_by_priority() for _ in range(500)]
        self.assertGreater(picks.count(0), 480)

    def test_set_priority_changes_only_that_position(self):
        memory = MemoryStore(3)
        for item in range(3):
            memory.store(item, 1.0)
        memory.set_priority(1, 7.0)
        self.assertEqual(memory.priorities, [1.0, 7.0, 1.0])
        self.assertEqual(memory.experiences, [0, 1, 2])

    def test_uniform_sample_still_works_with_priorities(self):
        memory = MemoryStore(3)
        memory.store("a", 0.01)
        random.seed(4)
        self.assertEqual(memory.sample(), "a")

    def test_invalid_priority_raises_value_error(self):
        memory = MemoryStore(3)
        for priority in (0.0, -1.0):
            with self.subTest(priority=priority):
                with self.assertRaises(ValueError):
                    memory.store("a", priority)
        self.assertEqual(len(memory), 0)
        memory.store("a")
        for priority in (0.0, -0.5):
            with self.subTest(priority=priority):
                with self.assertRaises(ValueError):
                    memory.set_priority(0, priority)

    def test_invalid_index_raises_value_error(self):
        memory = MemoryStore(3)
        memory.store("a")
        for index in (-1, 1, 5):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    memory.set_priority(index, 1.0)

    def test_empty_memory_cannot_be_sampled_by_priority(self):
        with self.assertRaises(ValueError):
            MemoryStore(3).sample_by_priority()


if __name__ == "__main__":
    unittest.main()
