"""
Unit tests for syno.brain.novelty.

These tests are intentionally stricter than the Novelty System's own
safeguards. They verify the step 5e worked example, habituation, recovery,
that each square is remembered separately, and that invalid settings fail
loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.novelty import NoveltySystem


class TestNoveltyWorkedExample(unittest.TestCase):
    """Verifies the exact worked example from step 5e."""

    def test_step_5e_sequence(self):
        novelty = NoveltySystem(0.2, 0.5, 0.95)
        rewards = [novelty.signal((0, 0)) for _ in range(3)]
        for actual, expected in zip(rewards, [0.2, 0.1, 0.05]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(novelty.signal((1, 0)), 0.2)
        for _ in range(50):
            novelty.tick()
        self.assertAlmostEqual(novelty.signal((0, 0)), 0.2 * (1 - 0.875 * 0.95 ** 50))


class TestNoveltyHabituation(unittest.TestCase):
    """Verifies that repeated visits become less rewarding."""

    def test_new_square_gives_the_full_scale(self):
        self.assertEqual(NoveltySystem(0.3, 0.5, 0.9).signal((4, 2)), 0.3)

    def test_each_visit_is_less_rewarding_than_the_last(self):
        novelty = NoveltySystem(0.2, 0.3, 0.9)
        rewards = [novelty.signal((1, 1)) for _ in range(10)]
        for earlier, later in zip(rewards, rewards[1:]):
            self.assertLess(later, earlier)

    def test_reward_never_goes_negative(self):
        novelty = NoveltySystem(0.2, 0.9, 1.0)
        for _ in range(100):
            self.assertGreaterEqual(novelty.signal((0, 0)), 0.0)

    def test_full_habituation_makes_a_second_visit_worthless(self):
        novelty = NoveltySystem(0.2, 1.0, 1.0)
        novelty.signal((0, 0))
        self.assertEqual(novelty.signal((0, 0)), 0.0)

    def test_zero_habituation_never_gets_boring(self):
        novelty = NoveltySystem(0.2, 0.0, 1.0)
        for _ in range(5):
            self.assertEqual(novelty.signal((0, 0)), 0.2)

    def test_squares_are_remembered_separately(self):
        novelty = NoveltySystem(0.2, 0.5, 1.0)
        for _ in range(5):
            novelty.signal((0, 0))
        self.assertEqual(novelty.signal((2, 2)), 0.2)

    def test_familiarity_is_stored_only_for_visited_squares(self):
        novelty = NoveltySystem(0.2, 0.5, 0.9)
        novelty.signal((0, 0))
        novelty.signal((1, 0))
        self.assertEqual(set(novelty.familiarity), {(0, 0), (1, 0)})


class TestNoveltyRecovery(unittest.TestCase):
    """Verifies that curiosity returns with time away."""

    def test_time_away_restores_interest(self):
        novelty = NoveltySystem(0.2, 0.5, 0.9)
        novelty.signal((0, 0))
        bored = NoveltySystem(0.2, 0.5, 0.9)
        bored.signal((0, 0))
        for _ in range(20):
            novelty.tick()
        self.assertGreater(novelty.signal((0, 0)), bored.signal((0, 0)))

    def test_tick_multiplies_every_familiarity_by_recovery(self):
        novelty = NoveltySystem(0.2, 0.5, 0.8)
        novelty.signal((0, 0))
        novelty.signal((1, 1))
        novelty.signal((1, 1))
        novelty.tick()
        self.assertAlmostEqual(novelty.familiarity[(0, 0)], 0.5 * 0.8)
        self.assertAlmostEqual(novelty.familiarity[(1, 1)], 0.75 * 0.8)

    def test_full_recovery_rate_means_no_forgetting(self):
        novelty = NoveltySystem(0.2, 0.5, 1.0)
        novelty.signal((0, 0))
        for _ in range(100):
            novelty.tick()
        self.assertAlmostEqual(novelty.signal((0, 0)), 0.1)

    def test_zero_recovery_forgets_instantly(self):
        novelty = NoveltySystem(0.2, 0.5, 0.0)
        novelty.signal((0, 0))
        novelty.tick()
        self.assertEqual(novelty.signal((0, 0)), 0.2)

    def test_tick_with_no_memories_is_allowed(self):
        NoveltySystem(0.2, 0.5, 0.9).tick()


class TestNoveltyValidation(unittest.TestCase):
    """Verifies that invalid settings fail loudly."""

    def test_invalid_settings_raise_value_error(self):
        for settings in [(-0.1, 0.5, 0.9), (0.2, -0.1, 0.9), (0.2, 1.5, 0.9), (0.2, 0.5, -0.1), (0.2, 0.5, 1.1)]:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    NoveltySystem(*settings)

    def test_boundary_settings_are_allowed(self):
        for settings in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]:
            with self.subTest(settings=settings):
                NoveltySystem(*settings)


if __name__ == "__main__":
    unittest.main()
