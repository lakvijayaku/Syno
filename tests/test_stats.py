"""
Unit tests for syno.tools.stats.

These tests are intentionally stricter than moving_average's own safeguards.
They verify the step 6c examples, the length of the result, that every
average uses exactly the right window, and that invalid windows fail loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.tools.stats import moving_average


class TestMovingAverage(unittest.TestCase):
    """Verifies smoothing with a moving average."""

    def test_step_6c_examples(self):
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 2), [1.5, 2.5, 3.5, 4.5])
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 5), [3.0])
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 1), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_result_length(self):
        values = list(range(20))
        for window in (1, 3, 10, 20):
            with self.subTest(window=window):
                self.assertEqual(len(moving_average(values, window)), len(values) - window + 1)

    def test_each_average_uses_the_window_ending_at_it(self):
        values = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0]
        result = moving_average(values, 3)
        for i, average in enumerate(result):
            with self.subTest(i=i):
                self.assertAlmostEqual(average, sum(values[i:i + 3]) / 3)

    def test_constant_values_stay_constant(self):
        self.assertEqual(moving_average([0.7] * 10, 4), [0.7] * 7)

    def test_smoothing_reduces_the_spread_of_noisy_data(self):
        noisy = [0.0, 1.0] * 50
        smoothed = moving_average(noisy, 10)
        self.assertLess(max(smoothed) - min(smoothed), max(noisy) - min(noisy))

    def test_negative_values(self):
        self.assertEqual(moving_average([-1.0, -3.0, 2.0], 2), [-2.0, -0.5])

    def test_does_not_modify_values(self):
        values = [1.0, 2.0, 3.0]
        moving_average(values, 2)
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_invalid_window_raises_value_error(self):
        for window in (0, -1, 6):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    moving_average([1, 2, 3, 4, 5], window)

    def test_empty_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            moving_average([], 1)


if __name__ == "__main__":
    unittest.main()
