"""
Unit tests for syno.brain.working_memory.

These tests are intentionally stricter than WorkingMemory's own safeguards.
They verify the step 14a worked example, the decay arithmetic, how quickly a
memory fades, resetting, that callers cannot change the memory, and that
invalid values fail loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.working_memory import WorkingMemory


class TestWorkingMemoryWorkedExample(unittest.TestCase):
    """Verifies the step 14a worked example."""

    def test_worked_example(self):
        memory = WorkingMemory(3, 0.5)
        self.assertEqual(memory.contents, [0.0, 0.0, 0.0])
        self.assertEqual(memory.update([1.0, 0.0, -1.0]), [0.5, 0.0, -0.5])
        self.assertEqual(memory.update([0.0, 0.0, 0.0]), [0.25, 0.0, -0.25])
        self.assertEqual(memory.update([1.0, 1.0, 1.0]), [0.625, 0.5, 0.375])
        memory.reset()
        self.assertEqual(memory.contents, [0.0, 0.0, 0.0])
        self.assertEqual(WorkingMemory(2, 0.0).update([0.3, -0.7]), [0.3, -0.7])


class TestWorkingMemorySetup(unittest.TestCase):
    """Verifies a new memory and its validation."""

    def test_starts_empty(self):
        self.assertEqual(WorkingMemory(4, 0.3).contents, [0.0] * 4)

    def test_stores_settings(self):
        memory = WorkingMemory(5, 0.25)
        self.assertEqual((memory.size, memory.decay), (5, 0.25))

    def test_size_one_is_allowed(self):
        self.assertEqual(WorkingMemory(1, 0.5).contents, [0.0])

    def test_rejects_size_below_one(self):
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    WorkingMemory(size, 0.5)

    def test_decay_bounds(self):
        WorkingMemory(2, 0.0)
        WorkingMemory(2, 0.999)
        for decay in (-0.1, 1.0, 1.5):
            with self.subTest(decay=decay):
                with self.assertRaises(ValueError):
                    WorkingMemory(2, decay)


class TestWorkingMemoryUpdate(unittest.TestCase):
    """Verifies how new values are blended in."""

    def test_blend_formula(self):
        memory = WorkingMemory(2, 0.75)
        memory.update([1.0, -2.0])
        self.assertEqual(memory.contents, [0.25, -0.5])
        memory.update([0.0, 2.0])
        self.assertAlmostEqual(memory.contents[0], 0.75 * 0.25)
        self.assertAlmostEqual(memory.contents[1], 0.75 * -0.5 + 0.25 * 2.0)

    def test_each_slot_is_independent(self):
        memory = WorkingMemory(3, 0.5)
        memory.update([1.0, 0.0, 0.0])
        self.assertEqual(memory.contents, [0.5, 0.0, 0.0])

    def test_fades_by_half_each_step(self):
        # Seen once, then nothing: half, a quarter, an eighth.
        memory = WorkingMemory(1, 0.5)
        memory.update([1.0])
        strengths = [memory.contents[0]]
        for _ in range(2):
            strengths.append(memory.update([0.0])[0])
        self.assertEqual(strengths, [0.5, 0.25, 0.125])

    def test_repeated_input_approaches_it(self):
        memory = WorkingMemory(1, 0.5)
        for _ in range(30):
            memory.update([1.0])
        self.assertAlmostEqual(memory.contents[0], 1.0, places=6)
        self.assertLess(memory.contents[0], 1.0)

    def test_zero_decay_has_no_memory(self):
        memory = WorkingMemory(2, 0.0)
        memory.update([0.9, 0.9])
        self.assertEqual(memory.update([0.1, -0.1]), [0.1, -0.1])

    def test_higher_decay_remembers_longer(self):
        short = WorkingMemory(1, 0.2)
        long = WorkingMemory(1, 0.8)
        # After one blank step both hold decay * (1 - decay), which is the
        # same for 0.2 and 0.8, so two blank steps are needed to tell them
        # apart: 0.032 against 0.128.
        for memory in (short, long):
            memory.update([1.0])
            memory.update([0.0])
            memory.update([0.0])
        self.assertAlmostEqual(short.contents[0], 0.032)
        self.assertAlmostEqual(long.contents[0], 0.128)

    def test_rejects_wrong_length(self):
        memory = WorkingMemory(3, 0.5)
        for values in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], []):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    memory.update(values)
        self.assertEqual(memory.contents, [0.0, 0.0, 0.0])


class TestWorkingMemoryIsolation(unittest.TestCase):
    """Verifies that callers cannot change the memory by accident."""

    def test_returned_list_is_a_copy(self):
        memory = WorkingMemory(2, 0.5)
        result = memory.update([1.0, 1.0])
        result.append(9.0)
        result[0] = 9.0
        self.assertEqual(memory.contents, [0.5, 0.5])

    def test_input_list_is_not_changed(self):
        values = [1.0, -1.0]
        WorkingMemory(2, 0.5).update(values)
        self.assertEqual(values, [1.0, -1.0])


class TestWorkingMemoryReset(unittest.TestCase):
    """Verifies emptying the memory."""

    def test_reset_empties(self):
        memory = WorkingMemory(3, 0.5)
        memory.update([1.0, 2.0, 3.0])
        memory.reset()
        self.assertEqual(memory.contents, [0.0, 0.0, 0.0])

    def test_reset_keeps_settings(self):
        memory = WorkingMemory(3, 0.4)
        memory.reset()
        self.assertEqual((memory.size, memory.decay), (3, 0.4))
        self.assertEqual(len(memory.contents), 3)

    def test_works_after_reset(self):
        memory = WorkingMemory(1, 0.5)
        memory.update([1.0])
        memory.reset()
        self.assertEqual(memory.update([1.0]), [0.5])


if __name__ == "__main__":
    unittest.main()
