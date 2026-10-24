"""
Experiment tests for experiments/habituation.py.

The fast tests check the experiment's settings, what SYNO senses, and that
the familiarity SYNO senses is measured after each visit, so its own square
always feels familiar. The slow test trains SYNO for the full 3000 lives
(about 2 minutes) and reproduces the recorded energy curve. It is skipped by
default and runs only when SYNO_SLOW_TESTS is set:

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
import experiments.curiosity as curiosity
import experiments.habituation as habituation
import experiments.scarcity as scarcity
from experiments.xor import make_layer
from syno.body.homeostasis import HomeostaticCore
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.brain.novelty import NoveltySystem
from syno.world.habitat import Habitat

WINDOW = (2 * habituation.SENSE_RADIUS + 1) ** 2
INPUTS = 2 * WINDOW + 2
# SYNO's own square, in the middle of the familiarity window.
OWN_SQUARE = WINDOW + WINDOW // 2


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(habituation.HIDDEN_NEURONS, INPUTS),
        make_layer(len(habituation.ACTION_NAMES), habituation.HIDDEN_NEURONS, "linear"),
    ])


class TestHabituationSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_settings(self):
        self.assertEqual(habituation.HIDDEN_NEURONS, 16)

    def test_shares_the_scarcity_world(self):
        self.assertEqual(habituation.GRID_SIZE, scarcity.GRID_SIZE)
        self.assertEqual(habituation.REGROW_DELAY, scarcity.REGROW_DELAY)
        self.assertEqual(habituation.LIVES, scarcity.LIVES)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "DISCOUNT", "FOOD_AMOUNT"):
            with self.subTest(name=name):
                self.assertEqual(getattr(habituation, name), getattr(consolidation, name))

    def test_shares_the_novelty_settings(self):
        for name in ("NOVELTY_SCALE", "HABITUATION", "RECOVERY"):
            with self.subTest(name=name):
                self.assertEqual(getattr(habituation, name), getattr(curiosity, name))

    def test_actions_are_unchanged(self):
        self.assertEqual(habituation.ACTION_NAMES, ["up", "down", "left", "right", "eat"])


class TestHabituationSenses(unittest.TestCase):
    """Verifies what SYNO senses."""

    def test_layout(self):
        habitat = Habitat(5, 5, (2, 2), [(3, 2)])
        novelty = NoveltySystem(0.2, 0.5, 0.95)
        novelty.signal((2, 1))
        body = HomeostaticCore(0.7, 0.4)
        state = habituation.senses(habitat, body, novelty)
        self.assertEqual(len(state), INPUTS)
        self.assertEqual(state[:WINDOW], habitat.sense(habituation.SENSE_RADIUS))
        self.assertEqual(state[WINDOW:2 * WINDOW], novelty.sense((2, 2), habituation.SENSE_RADIUS))
        self.assertEqual(state[WINDOW + 7], 0.5)
        self.assertAlmostEqual(state[-2], 0.3)
        self.assertEqual(state[-1], 0.4)

    def test_sensing_does_not_visit(self):
        habitat = Habitat(5, 5, (2, 2), [])
        novelty = NoveltySystem(0.2, 0.5, 0.95)
        habituation.senses(habitat, HomeostaticCore(0.5, 0.0), novelty)
        self.assertEqual(novelty.familiarity, {})


class TestHabituationLive(unittest.TestCase):
    """Verifies when familiarity is sensed during a life."""

    def record_updates(self, action: str) -> list:
        """
        Runs one life choosing the same action on every step, and returns
        the state and next state of every live update.
        """
        updates = []
        learn = habituation.learn

        def recording_learn(network, state, chosen, reward, next_state, *rest):
            """
            Records the state and next state of every update, then learns as
            normal.
            """
            updates.append((state, next_state))
            return learn(network, state, chosen, reward, next_state, *rest)

        random.seed(0)
        with mock.patch.object(habituation, "learn", side_effect=recording_learn), \
                mock.patch.object(habituation, "choose_action",
                                  return_value=habituation.ACTION_NAMES.index(action)):
            habituation.live(new_network(), MemoryStore(habituation.MEMORY_CAPACITY))
        # Each step makes one live update and REPLAYS_PER_STEP replays.
        return updates[::1 + habituation.REPLAYS_PER_STEP]

    def test_starting_square_already_feels_familiar(self):
        state, _ = self.record_updates("eat")[0]
        self.assertEqual(state[OWN_SQUARE], 0.5)

    def test_familiarity_is_sensed_after_each_visit(self):
        # Moving up onto squares not seen for a while, then pressing against
        # the top wall, SYNO's own square always feels at least half familiar.
        updates = self.record_updates("up")
        self.assertEqual(len(updates), habituation.LIFE_STEPS)
        for _, next_state in updates:
            self.assertGreaterEqual(next_state[OWN_SQUARE], 0.5)

    def test_staying_still_grows_familiarity(self):
        updates = self.record_updates("eat")
        own = [next_state[OWN_SQUARE] for _, next_state in updates]
        self.assertGreater(own[-1], 0.9)

    def test_state_layout(self):
        for state, next_state in self.record_updates("up"):
            self.assertEqual(len(state), INPUTS)
            self.assertEqual(len(next_state), INPUTS)

    def test_returns_average_energy_in_range(self):
        random.seed(0)
        energy = habituation.live(new_network(), MemoryStore(habituation.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestHabituationRun(unittest.TestCase):
    """Reproduces the recorded energy curve."""

    def test_energy_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            habituation.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.506",
            "Life 1000 | Avg Energy: 0.664",
            "Life 1500 | Avg Energy: 0.668",
            "Life 2000 | Avg Energy: 0.670",
            "Life 2500 | Avg Energy: 0.682",
            "Life 3000 | Avg Energy: 0.685",
        ])


if __name__ == "__main__":
    unittest.main()
