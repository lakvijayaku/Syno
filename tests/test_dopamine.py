"""
Unit tests for syno.brain.dopamine.

These tests are intentionally stricter than the function's own safeguards.
Beyond the formula itself, they verify that the RPE reproduces the responses
of real dopamine neurons described in docs/biology.md: a burst for unexpected
reward, no response for predicted reward, a dip for omitted reward, and a
burst at a predictive cue.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.dopamine import reward_prediction_error


class TestDopamineBiology(unittest.TestCase):
    """Verifies the four dopamine responses from the step 4 table."""

    def test_unexpected_reward_causes_a_burst(self):
        self.assertAlmostEqual(reward_prediction_error(1.0, 0.0, 0.0, 0.9, True), 1.0)

    def test_fully_predicted_reward_causes_no_response(self):
        self.assertAlmostEqual(reward_prediction_error(1.0, 1.0, 0.0, 0.9, True), 0.0)

    def test_omitted_reward_causes_a_dip(self):
        self.assertAlmostEqual(reward_prediction_error(0.0, 1.0, 0.0, 0.9, True), -1.0)

    def test_predictive_cue_causes_a_burst_without_reward(self):
        self.assertAlmostEqual(reward_prediction_error(0.0, 0.2, 0.9, 0.9, False), 0.61)


class TestRewardPredictionErrorFormula(unittest.TestCase):
    """Verifies each part of the formula."""

    def test_no_change_in_prediction_costs_the_discount(self):
        # Waiting one step without reward makes the same future one step
        # further away, so the prediction was slightly too high.
        self.assertAlmostEqual(reward_prediction_error(0.0, 0.5, 0.5, 0.9, False), -0.05)

    def test_done_ignores_next_expected(self):
        for next_expected in (0.0, 0.5, 100.0, -3.0):
            with self.subTest(next_expected=next_expected):
                self.assertAlmostEqual(reward_prediction_error(0.3, 0.1, next_expected, 0.9, True), 0.2)

    def test_discount_zero_ignores_the_future(self):
        self.assertAlmostEqual(reward_prediction_error(0.4, 0.1, 5.0, 0.0, False), 0.3)

    def test_discount_one_counts_the_future_fully(self):
        self.assertAlmostEqual(reward_prediction_error(0.0, 0.2, 0.7, 1.0, False), 0.5)

    def test_matches_general_formula(self):
        cases = [
            (0.0, 0.3, 0.6, 0.9),
            (1.0, 0.8, 0.0, 0.5),
            (-1.0, 0.0, 0.4, 0.95),
            (0.25, -0.5, -0.2, 0.7),
        ]
        for reward, expected, next_expected, discount in cases:
            with self.subTest(case=(reward, expected, next_expected, discount)):
                self.assertAlmostEqual(
                    reward_prediction_error(reward, expected, next_expected, discount, False),
                    reward + discount * next_expected - expected,
                )

    def test_larger_reward_gives_larger_rpe(self):
        low = reward_prediction_error(0.2, 0.5, 0.5, 0.9, False)
        high = reward_prediction_error(0.8, 0.5, 0.5, 0.9, False)
        self.assertAlmostEqual(high - low, 0.6)

    def test_higher_expectation_gives_smaller_rpe(self):
        low = reward_prediction_error(1.0, 0.2, 0.0, 0.9, True)
        high = reward_prediction_error(1.0, 0.7, 0.0, 0.9, True)
        self.assertAlmostEqual(low - high, 0.5)

    def test_returns_a_float(self):
        self.assertIsInstance(reward_prediction_error(1.0, 0.0, 0.0, 0.9, True), float)


class TestRewardPredictionErrorValidation(unittest.TestCase):
    """Verifies that an invalid discount fails loudly."""

    def test_discount_outside_range_raises_value_error(self):
        for discount in (-0.1, 1.1, -5.0, 2.0):
            with self.subTest(discount=discount):
                with self.assertRaises(ValueError):
                    reward_prediction_error(0.0, 0.0, 0.0, discount, False)

    def test_boundary_discounts_are_allowed(self):
        for discount in (0.0, 1.0):
            with self.subTest(discount=discount):
                reward_prediction_error(0.0, 0.0, 0.0, discount, False)


if __name__ == "__main__":
    unittest.main()
