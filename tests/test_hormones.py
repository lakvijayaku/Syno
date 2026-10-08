"""
Unit tests for syno.body.hormones.

These tests are intentionally stricter than the Hormone's own safeguards. They
verify the step 9a example, that the level builds up and fades gradually,
that it settles on a steady signal, that slower hormones respond more slowly,
and that invalid rates fail loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.body.hormones import Hormone


class TestHormone(unittest.TestCase):
    """Verifies how a hormone's level follows its signal."""

    def test_step_9a_example(self):
        hormone = Hormone(0.1)
        levels = [hormone.update(1.0) for _ in range(5)]
        for actual, expected in zip(levels, [0.1, 0.19, 0.271, 0.3439, 0.4095]):
            self.assertAlmostEqual(actual, expected, places=4)
        for _ in range(50):
            hormone.update(0.0)
        self.assertAlmostEqual(hormone.level, 0.40951 * 0.9 ** 50, places=6)

    def test_starts_at_zero(self):
        self.assertEqual(Hormone(0.3).level, 0.0)

    def test_update_returns_the_new_level(self):
        hormone = Hormone(0.5)
        returned = hormone.update(2.0)
        self.assertEqual(returned, hormone.level)
        self.assertEqual(returned, 1.0)

    def test_rate_one_follows_the_signal_instantly(self):
        hormone = Hormone(1.0)
        for signal in (0.7, -0.2, 5.0):
            with self.subTest(signal=signal):
                self.assertAlmostEqual(hormone.update(signal), signal)

    def test_builds_up_gradually_and_never_overshoots(self):
        hormone = Hormone(0.2)
        levels = [hormone.update(1.0) for _ in range(30)]
        for earlier, later in zip(levels, levels[1:]):
            self.assertGreater(later, earlier)
        self.assertLess(levels[-1], 1.0)

    def test_settles_on_a_steady_signal(self):
        hormone = Hormone(0.1)
        for _ in range(500):
            hormone.update(0.6)
        self.assertAlmostEqual(hormone.level, 0.6, places=6)

    def test_fades_gradually_after_the_signal_stops(self):
        hormone = Hormone(0.1)
        for _ in range(100):
            hormone.update(1.0)
        levels = [hormone.update(0.0) for _ in range(10)]
        for earlier, later in zip(levels, levels[1:]):
            self.assertLess(later, earlier)
        self.assertGreater(levels[-1], 0.3)

    def test_slower_hormone_responds_more_slowly(self):
        slow, fast = Hormone(0.01), Hormone(0.2)
        for _ in range(10):
            slow.update(1.0)
            fast.update(1.0)
        self.assertLess(slow.level, fast.level)

    def test_single_spike_barely_moves_a_slow_hormone(self):
        hormone = Hormone(0.01)
        hormone.update(10.0)
        self.assertAlmostEqual(hormone.level, 0.1)

    def test_follows_negative_signals(self):
        hormone = Hormone(0.5)
        hormone.update(-1.0)
        self.assertAlmostEqual(hormone.level, -0.5)

    def test_invalid_rate_raises_value_error(self):
        for rate in (0.0, -0.1, 1.5):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    Hormone(rate)


if __name__ == "__main__":
    unittest.main()
