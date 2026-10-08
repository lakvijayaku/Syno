"""
Unit tests for syno.brain.policy.

These tests are intentionally stricter than the function's own safeguards.
They verify pure exploitation and pure exploration exactly, check that the
exploration rate matches epsilon over many trials, and confirm that invalid
input fails loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import random
import unittest

from syno.brain.policy import choose_action


class TestChooseActionExploit(unittest.TestCase):
    """Verifies the choice when SYNO never explores."""

    def test_epsilon_zero_always_picks_the_best(self):
        for values, best in [
            ([0.1, 0.7, 0.3], 1),
            ([0.9, 0.2], 0),
            ([-5.0, -1.0, -3.0], 1),
            ([0.0, 0.0, 0.0, 1.0, 0.5], 3),
        ]:
            with self.subTest(values=values):
                for _ in range(50):
                    self.assertEqual(choose_action(values, 0.0), best)

    def test_ties_pick_the_first_best(self):
        self.assertEqual(choose_action([0.2, 0.8, 0.8], 0.0), 1)

    def test_single_action_is_always_chosen(self):
        for epsilon in (0.0, 0.5, 1.0):
            with self.subTest(epsilon=epsilon):
                self.assertEqual(choose_action([0.4], epsilon), 0)

    def test_returns_an_int(self):
        self.assertIsInstance(choose_action([0.1, 0.7], 0.0), int)


class TestChooseActionExplore(unittest.TestCase):
    """Verifies the choice when SYNO explores."""

    def test_epsilon_one_picks_every_action(self):
        random.seed(0)
        picks = {choose_action([0.1, 0.7, 0.3], 1.0) for _ in range(300)}
        self.assertEqual(picks, {0, 1, 2})

    def test_epsilon_one_picks_roughly_evenly(self):
        random.seed(0)
        counts = [0, 0, 0, 0]
        for _ in range(4000):
            counts[choose_action([0.9, 0.1, 0.1, 0.1], 1.0)] += 1
        for count in counts:
            self.assertGreater(count, 850)
            self.assertLess(count, 1150)

    def test_exploration_rate_matches_epsilon(self):
        # With epsilon = 0.3 and 4 actions, the best action is chosen 70% of
        # the time plus a quarter of the 30% random picks: 77.5% in total.
        random.seed(1)
        trials = 10000
        best = sum(choose_action([0.1, 0.9, 0.2, 0.3], 0.3) == 1 for _ in range(trials))
        self.assertAlmostEqual(best / trials, 0.775, delta=0.02)

    def test_always_returns_a_valid_position(self):
        random.seed(2)
        for _ in range(500):
            self.assertIn(choose_action([0.1, 0.2, 0.3, 0.4, 0.5], 0.5), range(5))

    def test_does_not_modify_values(self):
        values = [0.1, 0.7, 0.3]
        random.seed(3)
        for _ in range(20):
            choose_action(values, 0.5)
        self.assertEqual(values, [0.1, 0.7, 0.3])


class TestChooseActionValidation(unittest.TestCase):
    """Verifies that invalid input fails loudly."""

    def test_empty_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            choose_action([], 0.1)

    def test_epsilon_outside_range_raises_value_error(self):
        for epsilon in (-0.1, 1.1, -1.0, 2.0):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    choose_action([0.5, 0.5], epsilon)


if __name__ == "__main__":
    unittest.main()
