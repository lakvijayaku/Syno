"""
Unit tests for syno.body.homeostasis.

These tests are intentionally stricter than the Homeostatic Core's own
safeguards. They verify the step 5a worked example, every physical rule of
the body (digestion, burning, clamping, and stomach capacity), the drive and
the reward it produces, including alliesthesia, and that invalid values fail
loudly. HydratedCore is checked to add water without changing anything about
energy, the stomach, or HomeostaticCore itself.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.body.homeostasis import (
    BASE_BURN,
    DIGESTION_RATE,
    DRIVE_EXPONENT,
    MOVE_BURN,
    STOMACH_CAPACITY,
    WATER_BURN,
    HomeostaticCore,
    HydratedCore,
    homeostatic_reward,
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



class TestHomeostaticDrive(unittest.TestCase):
    """Verifies the drive and the reward it produces."""

    def test_drive_is_the_squared_deficit(self):
        self.assertEqual(DRIVE_EXPONENT, 2)
        for energy in (0.0, 0.3, 0.75, 1.0):
            with self.subTest(energy=energy):
                body = HomeostaticCore(energy, 0.0)
                self.assertAlmostEqual(body.drive(), (1.0 - energy) ** 2)

    def test_drive_is_zero_at_the_set_point(self):
        self.assertEqual(HomeostaticCore(1.0, 0.0).drive(), 0.0)

    def test_reward_is_the_reduction_in_drive(self):
        self.assertAlmostEqual(homeostatic_reward(0.5, 0.3), 0.2)
        self.assertAlmostEqual(homeostatic_reward(0.3, 0.5), -0.2)
        self.assertEqual(homeostatic_reward(0.4, 0.4), 0.0)

    def test_step_5b_examples(self):
        expected = {0.2: 0.0624, 0.8: 0.0144}
        for energy, reward in expected.items():
            with self.subTest(energy=energy):
                body = HomeostaticCore(energy, 0.5)
                before = body.drive()
                body.tick(False)
                self.assertAlmostEqual(homeostatic_reward(before, body.drive()), reward)

    def test_alliesthesia_same_food_is_worth_more_when_hungry(self):
        # The same digestion gives each body the same energy gain, but the
        # hungrier body must receive a larger reward.
        rewards = []
        for energy in (0.1, 0.3, 0.5, 0.7, 0.9):
            body = HomeostaticCore(energy, 0.5)
            before = body.drive()
            body.tick(False)
            rewards.append(homeostatic_reward(before, body.drive()))
        for hungrier, fuller in zip(rewards, rewards[1:]):
            self.assertGreater(hungrier, fuller)

    def test_hunger_without_food_is_punishing(self):
        body = HomeostaticCore(0.3, 0.0)
        before = body.drive()
        body.tick(True)
        self.assertAlmostEqual(homeostatic_reward(before, body.drive()), -0.0284)

    def test_hunger_hurts_more_when_hungrier(self):
        penalties = []
        for energy in (0.2, 0.5, 0.8):
            body = HomeostaticCore(energy, 0.0)
            before = body.drive()
            body.tick(False)
            penalties.append(homeostatic_reward(before, body.drive()))
        self.assertLess(penalties[0], penalties[1])
        self.assertLess(penalties[1], penalties[2])

    def test_moving_costs_reward(self):
        resting, moving = HomeostaticCore(0.5, 0.0), HomeostaticCore(0.5, 0.0)
        rest_before, move_before = resting.drive(), moving.drive()
        resting.tick(False)
        moving.tick(True)
        self.assertLess(
            homeostatic_reward(move_before, moving.drive()),
            homeostatic_reward(rest_before, resting.drive()),
        )

    def test_full_body_gets_no_reward_from_more_digestion(self):
        # Energy is already at the set-point, so digestion cannot help.
        body = HomeostaticCore(1.0, 1.0)
        before = body.drive()
        body.tick(False)
        self.assertEqual(homeostatic_reward(before, body.drive()), 0.0)


class TestHydratedCoreWorkedExample(unittest.TestCase):
    """Verifies the step 13a worked example."""

    def test_worked_example(self):
        body = HydratedCore(0.5, 0.0, 0.5)
        self.assertEqual(body.drive(), 0.5)
        self.assertEqual(body.drink(0.3), 0.3)
        self.assertEqual(body.water, 0.8)
        self.assertEqual(body.drink(0.3), 0.19999999999999996)
        self.assertEqual(body.water, 1.0)
        body.tick(False)
        self.assertEqual(body.energy, 0.49)
        self.assertEqual(body.water, 0.99)
        self.assertEqual(body.drive(), 0.2602)


class TestHydratedCoreSetup(unittest.TestCase):
    """Verifies that HydratedCore is a HomeostaticCore with water."""

    def test_is_a_homeostatic_core(self):
        self.assertIsInstance(HydratedCore(0.5, 0.0, 0.5), HomeostaticCore)

    def test_water_burn(self):
        self.assertEqual(WATER_BURN, 0.01)

    def test_stores_all_three_values(self):
        body = HydratedCore(0.4, 0.2, 0.7)
        self.assertEqual((body.energy, body.stomach, body.water), (0.4, 0.2, 0.7))

    def test_bounds_are_inclusive(self):
        HydratedCore(0.5, 0.0, 0.0)
        HydratedCore(0.5, 0.0, 1.0)

    def test_rejects_water_out_of_range(self):
        for water in (-0.01, 1.01):
            with self.subTest(water=water):
                with self.assertRaises(ValueError):
                    HydratedCore(0.5, 0.0, water)

    def test_still_checks_energy_and_stomach(self):
        with self.assertRaises(ValueError):
            HydratedCore(1.5, 0.0, 0.5)
        with self.assertRaises(ValueError):
            HydratedCore(0.5, -0.1, 0.5)

    def test_parent_has_no_water(self):
        self.assertFalse(hasattr(HomeostaticCore(0.5, 0.0), "water"))


class TestHydratedCoreDrink(unittest.TestCase):
    """Verifies drinking."""

    def test_drink_adds_water(self):
        body = HydratedCore(0.5, 0.0, 0.2)
        self.assertEqual(body.drink(0.3), 0.3)
        self.assertAlmostEqual(body.water, 0.5)

    def test_drink_stops_at_full(self):
        body = HydratedCore(0.5, 0.0, 0.9)
        self.assertAlmostEqual(body.drink(0.3), 0.1)
        self.assertEqual(body.water, 1.0)

    def test_drink_when_full_accepts_nothing(self):
        body = HydratedCore(0.5, 0.0, 1.0)
        self.assertEqual(body.drink(0.3), 0.0)
        self.assertEqual(body.water, 1.0)

    def test_drink_zero(self):
        body = HydratedCore(0.5, 0.0, 0.5)
        self.assertEqual(body.drink(0.0), 0.0)
        self.assertEqual(body.water, 0.5)

    def test_drink_rejects_negative(self):
        body = HydratedCore(0.5, 0.0, 0.5)
        with self.assertRaises(ValueError):
            body.drink(-0.1)
        self.assertEqual(body.water, 0.5)

    def test_drink_does_not_touch_energy_or_stomach(self):
        body = HydratedCore(0.5, 0.2, 0.5)
        body.drink(0.3)
        self.assertEqual((body.energy, body.stomach), (0.5, 0.2))


class TestHydratedCoreTick(unittest.TestCase):
    """Verifies that a tick loses water and otherwise matches HomeostaticCore."""

    def test_water_drains_each_tick(self):
        body = HydratedCore(0.5, 0.0, 0.5)
        body.tick(False)
        self.assertAlmostEqual(body.water, 0.5 - WATER_BURN)

    def test_moving_costs_no_extra_water(self):
        still = HydratedCore(0.5, 0.0, 0.5)
        moving = HydratedCore(0.5, 0.0, 0.5)
        still.tick(False)
        moving.tick(True)
        self.assertEqual(still.water, moving.water)

    def test_water_never_goes_below_zero(self):
        body = HydratedCore(0.5, 0.0, 0.005)
        body.tick(False)
        self.assertEqual(body.water, 0.0)

    def test_energy_and_stomach_match_homeostatic_core(self):
        for moved in (False, True):
            with self.subTest(moved=moved):
                plain = HomeostaticCore(0.6, 0.3)
                hydrated = HydratedCore(0.6, 0.3, 0.5)
                for _ in range(5):
                    plain.tick(moved)
                    hydrated.tick(moved)
                self.assertEqual(hydrated.energy, plain.energy)
                self.assertEqual(hydrated.stomach, plain.stomach)

    def test_eat_matches_homeostatic_core(self):
        plain = HomeostaticCore(0.5, 0.8)
        hydrated = HydratedCore(0.5, 0.8, 0.5)
        self.assertEqual(hydrated.eat(0.3), plain.eat(0.3))
        self.assertEqual(hydrated.water, 0.5)


class TestHydratedCoreDrive(unittest.TestCase):
    """Verifies the two-need drive."""

    def test_water_deficit(self):
        self.assertAlmostEqual(HydratedCore(0.5, 0.0, 0.3).water_deficit(), 0.7)
        self.assertEqual(HydratedCore(0.5, 0.0, 1.0).water_deficit(), 0.0)

    def test_drive_adds_both_squared_deficits(self):
        body = HydratedCore(0.8, 0.0, 0.2)
        self.assertAlmostEqual(body.drive(), 0.2 ** 2 + 0.8 ** 2)

    def test_drive_is_zero_when_both_needs_are_met(self):
        self.assertEqual(HydratedCore(1.0, 0.0, 1.0).drive(), 0.0)

    def test_drive_is_two_when_both_are_empty(self):
        self.assertEqual(HydratedCore(0.0, 0.0, 0.0).drive(), 2.0)

    def test_larger_need_dominates(self):
        thirsty = HydratedCore(0.8, 0.0, 0.2)
        self.assertGreater(thirsty.water_deficit() ** 2, 10 * thirsty.deficit() ** 2)

    def test_drinking_when_thirsty_is_rewarded_more(self):
        # Alliesthesia for water: the same drink is worth more when thirstier.
        thirsty = HydratedCore(1.0, 0.0, 0.2)
        slightly = HydratedCore(1.0, 0.0, 0.8)
        rewards = []
        for body in (thirsty, slightly):
            before = body.drive()
            body.drink(0.1)
            rewards.append(homeostatic_reward(before, body.drive()))
        self.assertGreater(rewards[0], rewards[1])

    def test_parent_drive_ignores_water(self):
        self.assertAlmostEqual(HomeostaticCore(0.8, 0.0).drive(), 0.2 ** 2)


class TestHydratedCoreDeath(unittest.TestCase):
    """Verifies when SYNO dies."""

    def test_alive_with_both(self):
        self.assertFalse(HydratedCore(0.01, 0.0, 0.01).is_dead())

    def test_dies_without_water(self):
        self.assertTrue(HydratedCore(0.5, 0.0, 0.0).is_dead())

    def test_dies_without_energy(self):
        self.assertTrue(HydratedCore(0.0, 0.0, 0.5).is_dead())

    def test_dies_of_thirst_over_time(self):
        body = HydratedCore(1.0, 1.0, 0.03)
        steps = 0
        while not body.is_dead() and steps < 10:
            body.tick(False)
            steps += 1
        self.assertEqual(steps, 3)


if __name__ == "__main__":
    unittest.main()
