"""
Unit tests for syno.body.homeostasis.

These tests are intentionally stricter than the Homeostatic Core's own
safeguards. They verify the step 5a worked example, every physical rule of
the body (digestion, burning, clamping, and stomach capacity), and that
invalid values fail loudly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.body.homeostasis import (
    BASE_BURN,
    DIGESTION_RATE,
    MOVE_BURN,
    STOMACH_CAPACITY,
    HomeostaticCore,
)


class TestHomeostaticCoreWorkedExample(unittest.TestCase):
    """Verifies the exact worked example from step 5a."""

    def test_step_5a_sequence(self):
        body = HomeostaticCore(0.5, 0.0)
        self.assertAlmostEqual(body.eat(0.3), 0.3)
        self.assertAlmostEqual(body.stomach, 0.3)
        body.tick(True)
        self.assertAlmostEqual(body.energy, 0.53)
        self.assertAlmostEqual(body.stomach, 0.25)
        self.assertAlmostEqual(body.deficit(), 0.47)
        self.assertAlmostEqual(body.eat(0.9), 0.75)
        self.assertAlmostEqual(body.stomach, 1.0)
        body.tick(False)
        self.assertAlmostEqual(body.energy, 0.57)
        self.assertAlmostEqual(body.stomach, 0.95)


class TestHomeostaticCoreEat(unittest.TestCase):
    """Verifies the stomach."""

    def test_food_that_fits_is_all_accepted(self):
        body = HomeostaticCore(0.5, 0.2)
        self.assertAlmostEqual(body.eat(0.5), 0.5)
        self.assertAlmostEqual(body.stomach, 0.7)

    def test_full_stomach_refuses_everything(self):
        body = HomeostaticCore(0.5, STOMACH_CAPACITY)
        self.assertEqual(body.eat(0.4), 0.0)
        self.assertEqual(body.stomach, STOMACH_CAPACITY)

    def test_stomach_never_exceeds_capacity(self):
        body = HomeostaticCore(0.5, 0.0)
        for amount in (0.4, 0.4, 0.4, 0.4):
            body.eat(amount)
            self.assertLessEqual(body.stomach, STOMACH_CAPACITY)

    def test_eating_does_not_change_energy_directly(self):
        # Energy only rises through digestion over later ticks.
        body = HomeostaticCore(0.3, 0.0)
        body.eat(0.5)
        self.assertEqual(body.energy, 0.3)

    def test_eating_zero_is_allowed(self):
        body = HomeostaticCore(0.5, 0.5)
        self.assertEqual(body.eat(0.0), 0.0)
        self.assertEqual(body.stomach, 0.5)

    def test_negative_amount_raises_value_error(self):
        with self.assertRaises(ValueError):
            HomeostaticCore(0.5, 0.5).eat(-0.1)


class TestHomeostaticCoreTick(unittest.TestCase):
    """Verifies digestion, burning, and clamping."""

    def test_resting_with_empty_stomach_burns_base_energy(self):
        body = HomeostaticCore(0.5, 0.0)
        body.tick(False)
        self.assertAlmostEqual(body.energy, 0.5 - BASE_BURN)

    def test_moving_burns_extra_energy(self):
        resting = HomeostaticCore(0.5, 0.0)
        moving = HomeostaticCore(0.5, 0.0)
        resting.tick(False)
        moving.tick(True)
        self.assertAlmostEqual(resting.energy - moving.energy, MOVE_BURN)

    def test_digestion_moves_food_into_energy(self):
        body = HomeostaticCore(0.5, 0.5)
        body.tick(False)
        self.assertAlmostEqual(body.stomach, 0.5 - DIGESTION_RATE)
        self.assertAlmostEqual(body.energy, 0.5 + DIGESTION_RATE - BASE_BURN)

    def test_digestion_never_takes_more_than_the_stomach_holds(self):
        body = HomeostaticCore(0.5, 0.02)
        body.tick(False)
        self.assertEqual(body.stomach, 0.0)
        self.assertAlmostEqual(body.energy, 0.5 + 0.02 - BASE_BURN)

    def test_stomach_never_goes_negative(self):
        body = HomeostaticCore(0.5, 0.12)
        for _ in range(10):
            body.tick(False)
            self.assertGreaterEqual(body.stomach, 0.0)

    def test_energy_never_drops_below_zero(self):
        body = HomeostaticCore(0.005, 0.0)
        body.tick(True)
        self.assertEqual(body.energy, 0.0)

    def test_energy_never_rises_above_set_point(self):
        body = HomeostaticCore(1.0, 1.0)
        body.tick(False)
        self.assertEqual(body.energy, 1.0)
        self.assertAlmostEqual(body.stomach, 1.0 - DIGESTION_RATE)

    def test_a_meal_is_digested_over_several_ticks(self):
        # 0.2 of food at 0.05 per tick takes exactly 4 ticks to digest.
        body = HomeostaticCore(0.5, 0.0)
        body.eat(0.2)
        for _ in range(3):
            body.tick(False)
        self.assertGreater(body.stomach, 0.0)
        body.tick(False)
        self.assertAlmostEqual(body.stomach, 0.0)

    def test_starving_body_stays_at_zero(self):
        body = HomeostaticCore(0.1, 0.0)
        for _ in range(100):
            body.tick(True)
        self.assertEqual(body.energy, 0.0)


class TestHomeostaticCoreDeficit(unittest.TestCase):
    """Verifies the distance from the set-point."""

    def test_deficit_is_distance_from_full_energy(self):
        for energy, deficit in [(1.0, 0.0), (0.0, 1.0), (0.25, 0.75)]:
            with self.subTest(energy=energy):
                self.assertAlmostEqual(HomeostaticCore(energy, 0.0).deficit(), deficit)

    def test_deficit_rises_as_energy_burns(self):
        body = HomeostaticCore(0.8, 0.0)
        before = body.deficit()
        body.tick(True)
        self.assertGreater(body.deficit(), before)


class TestHomeostaticCoreValidation(unittest.TestCase):
    """Verifies that an invalid body fails loudly."""

    def test_out_of_range_values_raise_value_error(self):
        for energy, stomach in [(1.5, 0.0), (-0.1, 0.0), (0.5, 1.1), (0.5, -0.1)]:
            with self.subTest(energy=energy, stomach=stomach):
                with self.assertRaises(ValueError):
                    HomeostaticCore(energy, stomach)

    def test_boundary_values_are_allowed(self):
        for energy, stomach in [(0.0, 0.0), (1.0, 1.0)]:
            with self.subTest(energy=energy, stomach=stomach):
                HomeostaticCore(energy, stomach)


if __name__ == "__main__":
    unittest.main()
