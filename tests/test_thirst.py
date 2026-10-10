"""
Experiment tests for experiments/thirst.py.

The fast tests check the experiment's settings, that new food never lands on
SYNO, food, or the pond, what SYNO senses, that drinking keeps SYNO alive,
and that death ends a life and is learned as a final step. The slow test
trains SYNO for the full 3000 lives (about 85 seconds) and reproduces the
recorded lifespan curve. It is skipped by default and runs only when
SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import random
import unittest
from unittest import mock

import experiments.consolidation as consolidation
import experiments.scarcity as scarcity
import experiments.thirst as thirst
from experiments.xor import make_layer
from syno.body.homeostasis import HydratedCore
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.world.habitat import Habitat

INPUTS = 2 * (2 * thirst.SENSE_RADIUS + 1) ** 2 + 3


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(thirst.HIDDEN_NEURONS, INPUTS),
        make_layer(len(thirst.ACTION_NAMES), thirst.HIDDEN_NEURONS, "linear"),
    ])


def always(action_name: str):
    """Returns a stand-in for choose_action that always picks one action."""
    index = thirst.ACTION_NAMES.index(action_name)
    return lambda values, epsilon: index


class TestThirstSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_actions(self):
        self.assertEqual(thirst.ACTION_NAMES, ["up", "down", "left", "right", "eat", "drink"])

    def test_settings(self):
        self.assertEqual(thirst.HIDDEN_NEURONS, 16)
        self.assertEqual(thirst.DRINK_AMOUNT, 0.3)
        self.assertEqual(thirst.LIVES, 3000)

    def test_shares_the_scarcity_world(self):
        self.assertEqual(thirst.GRID_SIZE, scarcity.GRID_SIZE)
        self.assertEqual(thirst.REGROW_DELAY, scarcity.REGROW_DELAY)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "DISCOUNT", "FOOD_AMOUNT"):
            with self.subTest(name=name):
                self.assertEqual(getattr(thirst, name), getattr(consolidation, name))


class TestFreeSquare(unittest.TestCase):
    """Verifies that new food lands only on an empty square."""

    def test_finds_the_only_free_square(self):
        # SYNO, food, and water fill every square of a 2x2 grid but one.
        habitat = Habitat(2, 2, (0, 0), [(1, 0)], [(0, 1)])
        random.seed(0)
        for _ in range(20):
            self.assertEqual(thirst.free_square(habitat), (1, 1))

    def test_never_picks_the_pond(self):
        habitat = Habitat(2, 1, (0, 0), [], [(1, 0)])
        habitat.agent = (1, 0)
        random.seed(0)
        for _ in range(20):
            self.assertEqual(thirst.free_square(habitat), (0, 0))

    def test_does_not_change_the_habitat(self):
        habitat = Habitat(3, 3, (1, 1), [(0, 0)], [(2, 2)])
        thirst.free_square(habitat)
        self.assertEqual((habitat.food, habitat.water), ([(0, 0)], [(2, 2)]))


class TestSenses(unittest.TestCase):
    """Verifies what SYNO senses."""

    def test_food_window_water_window_then_body(self):
        habitat = Habitat(5, 5, (2, 2), [(2, 1)], [(3, 2)])
        body = HydratedCore(0.7, 0.2, 0.4)
        state = thirst.senses(habitat, body)
        window = (2 * thirst.SENSE_RADIUS + 1) ** 2
        self.assertEqual(len(state), INPUTS)
        self.assertEqual(state[:window], habitat.sense(thirst.SENSE_RADIUS))
        self.assertEqual(state[window:2 * window], habitat.sense_water(thirst.SENSE_RADIUS))
        self.assertEqual(state[-3:], [body.deficit(), 0.2, body.water_deficit()])


class TestThirstLive(unittest.TestCase):
    """Verifies lifespans, drinking, and death."""

    def run_life(self, action_name: str, pond_under_syno: bool, energy: float, water: float):
        """
        Runs one life where SYNO always takes one action, with chosen starting
        energy and water, and the pond either under SYNO or away from it.
        """
        calls = []

        def place(habitat):
            calls.append(None)
            if len(calls) == 1 and pond_under_syno:
                return habitat.agent
            return original(habitat)

        original = thirst.free_square
        memory = MemoryStore(thirst.MEMORY_CAPACITY)
        random.seed(0)
        # Built first, because make_layer also draws from random.uniform.
        network = new_network()
        with mock.patch.object(thirst, "free_square", side_effect=place), \
                mock.patch.object(thirst, "choose_action", side_effect=always(action_name)), \
                mock.patch.object(thirst.random, "uniform", side_effect=[energy, water]):
            steps = thirst.live(network, memory)
        return steps, memory

    def test_dies_of_thirst_without_drinking(self):
        # Water 0.3 drains by 0.01 a step, so SYNO dies after 30 steps.
        steps, _ = self.run_life("eat", False, 0.9, 0.3)
        self.assertEqual(steps, 30)

    def test_drinking_keeps_syno_alive_until_hunger(self):
        # Drinking from the pond every step keeps water up, so SYNO lives
        # until its energy of 0.9 runs out, about 90 steps.
        steps, _ = self.run_life("drink", True, 0.9, 0.3)
        self.assertGreaterEqual(steps, 89)
        self.assertLess(steps, thirst.LIFE_STEPS)

    def test_drinking_away_from_the_pond_does_nothing(self):
        steps, _ = self.run_life("drink", False, 0.9, 0.3)
        self.assertEqual(steps, 30)

    def test_death_is_learned_as_the_final_step(self):
        steps, memory = self.run_life("eat", False, 0.9, 0.3)
        dones = [experience[4] for experience in memory.experiences]
        self.assertEqual(len(dones), steps)
        self.assertTrue(dones[-1])
        self.assertFalse(any(dones[:-1]))

    def test_learn_is_told_about_death(self):
        with mock.patch.object(thirst, "learn", wraps=thirst.learn) as learn:
            self.run_life("eat", False, 0.9, 0.3)
        live_calls = [call for call in learn.call_args_list if call.args[5]]
        self.assertGreaterEqual(len(live_calls), 1)

    def test_replayed_death_is_still_final(self):
        # Memory always recalls the newest experience, so on the last step
        # the death itself is replayed twice. All three updates for it must
        # be told the step was final.
        class NewestFirst(MemoryStore):
            def sample_by_priority(self):
                return len(self.experiences) - 1

        calls = []
        original = thirst.learn

        def recording_learn(*args):
            calls.append(args[5])
            return original(*args)

        random.seed(0)
        network = new_network()
        with mock.patch.object(thirst, "learn", side_effect=recording_learn), \
                mock.patch.object(thirst, "choose_action", side_effect=always("eat")), \
                mock.patch.object(thirst.random, "uniform", side_effect=[0.9, 0.3]):
            thirst.live(network, NewestFirst(thirst.MEMORY_CAPACITY))
        self.assertEqual(calls[-3:], [True, True, True])
        self.assertEqual(sum(calls), 3)

    def test_full_life_when_needs_are_met(self):
        random.seed(0)
        steps = thirst.live(new_network(), MemoryStore(thirst.MEMORY_CAPACITY))
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, thirst.LIFE_STEPS)

    def test_world_has_one_pond_and_one_food(self):
        seen = []
        original = thirst.Habitat

        def recording_habitat(*args, **kwargs):
            habitat = original(*args, **kwargs)
            seen.append(habitat)
            return habitat

        random.seed(0)
        with mock.patch.object(thirst, "Habitat", side_effect=recording_habitat), \
                mock.patch.object(thirst, "choose_action", side_effect=always("eat")):
            thirst.live(new_network(), MemoryStore(thirst.MEMORY_CAPACITY))
        habitat = seen[0]
        self.assertEqual(len(habitat.water), 1)
        self.assertEqual(len(habitat.food), 1)
        self.assertNotIn(habitat.water[0], habitat.food)
        self.assertNotEqual(habitat.water[0], habitat.agent)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestThirstRun(unittest.TestCase):
    """Reproduces the recorded lifespan curve."""

    def test_lifespan_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            thirst.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Lifespan: 45.7 steps",
            "Life 1000 | Avg Lifespan: 56.4 steps",
            "Life 1500 | Avg Lifespan: 76.6 steps",
            "Life 2000 | Avg Lifespan: 85.9 steps",
            "Life 2500 | Avg Lifespan: 89.9 steps",
            "Life 3000 | Avg Lifespan: 87.5 steps",
        ])


if __name__ == "__main__":
    unittest.main()
