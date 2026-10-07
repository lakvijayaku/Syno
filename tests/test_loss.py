"""
Unit tests for syno.brain.loss.

These tests are intentionally stricter than the loss function's own
safeguards. They verify known values, the mathematical properties every mean
squared error must have, and that invalid input fails loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.loss import mean_squared_error


class TestMeanSquaredErrorValues(unittest.TestCase):
    """Verifies that the loss computes the correct values."""

    def test_step_2c_hand_check(self):
        # Errors -0.5 and 0.5, squared to 0.25 each, averaged to 0.25.
        self.assertAlmostEqual(mean_squared_error([0.5, 0.5], [1.0, 0.0]), 0.25)

    def test_worked_example(self):
        # Errors -0.2 and 0.3, squared to 0.04 and 0.09, averaged to 0.065.
        self.assertAlmostEqual(mean_squared_error([0.8, 0.3], [1.0, 0.0]), 0.065)

    def test_perfect_predictions_give_zero(self):
        self.assertEqual(mean_squared_error([1.0], [1.0]), 0.0)
        self.assertEqual(mean_squared_error([0.2, -3.5, 7.0], [0.2, -3.5, 7.0]), 0.0)

    def test_single_pair(self):
        self.assertAlmostEqual(mean_squared_error([3.0], [1.0]), 4.0)

    def test_returns_a_float(self):
        self.assertIsInstance(mean_squared_error([0.5], [1.0]), float)


class TestMeanSquaredErrorProperties(unittest.TestCase):
    """Verifies properties that every mean squared error must have."""

    def test_loss_is_never_negative(self):
        cases = [
            ([0.0], [1.0]),
            ([-5.0, 2.0], [3.0, -1.0]),
            ([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]),
        ]
        for predictions, targets in cases:
            with self.subTest(predictions=predictions, targets=targets):
                self.assertGreaterEqual(mean_squared_error(predictions, targets), 0.0)

    def test_errors_in_opposite_directions_do_not_cancel(self):
        # Without squaring, -1 and +1 would average to 0.
        self.assertAlmostEqual(mean_squared_error([0.0, 2.0], [1.0, 1.0]), 1.0)

    def test_sign_of_the_error_does_not_matter(self):
        too_low = mean_squared_error([0.8], [1.0])
        too_high = mean_squared_error([1.2], [1.0])
        self.assertAlmostEqual(too_low, too_high)

    def test_large_errors_are_punished_more_than_proportionally(self):
        # An error 5 times larger must cost 25 times more.
        small = mean_squared_error([0.9], [1.0])
        large = mean_squared_error([0.5], [1.0])
        self.assertAlmostEqual(large / small, 25.0)

    def test_swapping_predictions_and_targets_gives_same_loss(self):
        self.assertAlmostEqual(
            mean_squared_error([0.3, 0.9], [1.0, 0.0]),
            mean_squared_error([1.0, 0.0], [0.3, 0.9]),
        )

    def test_loss_is_an_average_not_a_total(self):
        # Repeating the same pair must not change the loss.
        one = mean_squared_error([0.5], [1.0])
        many = mean_squared_error([0.5] * 10, [1.0] * 10)
        self.assertAlmostEqual(one, many)

    def test_order_of_pairs_does_not_matter(self):
        self.assertAlmostEqual(
            mean_squared_error([0.1, 0.9, 0.4], [0.0, 1.0, 1.0]),
            mean_squared_error([0.4, 0.1, 0.9], [1.0, 0.0, 1.0]),
        )


class TestMeanSquaredErrorValidation(unittest.TestCase):
    """Verifies that invalid input fails loudly."""

    def test_empty_lists_raise_value_error(self):
        with self.assertRaises(ValueError):
            mean_squared_error([], [])

    def test_empty_predictions_raise_value_error(self):
        with self.assertRaises(ValueError):
            mean_squared_error([], [1.0])

    def test_empty_targets_raise_value_error(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1.0], [])

    def test_more_targets_than_predictions_raise_value_error(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1.0], [1.0, 2.0])

    def test_more_predictions_than_targets_raise_value_error(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1.0, 2.0], [1.0])

    def test_mismatch_message_reports_both_lengths(self):
        with self.assertRaises(ValueError) as context:
            mean_squared_error([1.0, 2.0, 3.0], [1.0])
        self.assertIn("3 predictions and 1 targets", str(context.exception))

    def test_mismatch_preserves_original_cause(self):
        with self.assertRaises(ValueError) as context:
            mean_squared_error([1.0], [1.0, 2.0])
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_inputs_are_not_modified(self):
        predictions = [0.5, 0.5]
        targets = [1.0, 0.0]
        mean_squared_error(predictions, targets)
        self.assertEqual(predictions, [0.5, 0.5])
        self.assertEqual(targets, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
