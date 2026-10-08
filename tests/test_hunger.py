"""
Experiment tests for experiments/hunger.py.

The fast tests check the experiment's building blocks. The slow test trains
SYNO for the full 5000 lives (about 25 seconds), reproduces the recorded
energy curve, and checks that SYNO learned to eat when hungry. It is skipped
by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

from experiments.hunger import (
    ACTION_NAMES,
    GRID_SIZE,
    HIDDEN_NEURONS,
    LIFE_STEPS,
    LIVES,
    SENSE_RADIUS,
    live,
    senses,
    spawn_food,
)
from experiments.xor import make_layer
from syno.body.homeostasis import HomeostaticCore
from syno.brain.network import Network
from syno.world.habitat import Habitat

INPUTS = (2 * SENSE_RADIUS + 1) ** 2 + 2


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(HIDDEN_NEURONS, INPUTS),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear"),
    ])


class TestHungerBuildingBlocks(unittest.TestCase):
    """Verifies the pieces of the experiment."""

    def test_spawn_food_uses_a_free_square(self):
        random.seed(0)
        for _ in range(200):
            habitat = Habitat(GRID_SIZE, GRID_SIZE, (1, 1), [(0, 0)])
            spawn_food(habitat)
            self.assertEqual(len(habitat.food), 2)
            new_food = habitat.food[-1]
            self.assertNotEqual(new_food, (1, 1))
            self.assertNotEqual(new_food, (0, 0))
            self.assertTrue(habitat.in_bounds(new_food))

    def test_senses_append_how_the_body_feels(self):
        habitat = Habitat(GRID_SIZE, GRID_SIZE, (1, 1), [(0, 0)])
        body = HomeostaticCore(0.25, 0.4)
        result = senses(habitat, body)
        self.assertEqual(len(result), INPUTS)
        self.assertEqual(result[:-2], habitat.sense(SENSE_RADIUS))
        self.assertAlmostEqual(result[-2], 0.75)
        self.assertAlmostEqual(result[-1], 0.4)

    def test_make_layer_passes_the_activation(self):
        random.seed(1)
        self.assertTrue(all(n.activation == "linear" for n in make_layer(3, 2, "linear").neurons))
        self.assertTrue(all(n.activation == "sigmoid" for n in make_layer(3, 2).neurons))

    def test_output_layer_is_linear(self):
        # Body rewards can be negative, so values must be able to go below 0.
        random.seed(2)
        network = new_network()
        self.assertTrue(all(n.activation == "linear" for n in network.layers[-1].neurons))
        self.assertTrue(all(n.activation == "sigmoid" for n in network.layers[0].neurons))

    def test_live_returns_an_average_energy(self):
        random.seed(3)
        network = new_network()
        for _ in range(5):
            energy = live(network)
            self.assertGreaterEqual(energy, 0.0)
            self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestHungerLearning(unittest.TestCase):
    """Verifies that SYNO learns to keep itself fed (slow)."""

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.network = new_network()
        cls.energies = [live(cls.network) for _ in range(LIVES)]

    def test_seed_zero_matches_the_recorded_energy_curve(self):
        recorded = [0.231, 0.260, 0.286, 0.332, 0.507, 0.746, 0.740, 0.760, 0.778, 0.823]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                average = sum(self.energies[block * 500:(block + 1) * 500]) / 500
                self.assertAlmostEqual(average, expected, places=3)

    def test_trained_syno_eats_when_hungry(self):
        # On a food square, with an empty stomach and energy at or below 0.7,
        # SYNO's best action must be to eat, from every square.
        for energy in (0.1, 0.3, 0.5, 0.7):
            body = HomeostaticCore(energy, 0.0)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    with self.subTest(energy=energy, square=(x, y)):
                        habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [(x, y)])
                        values = self.network.forward(senses(habitat, body))
                        self.assertEqual(values.index(max(values)), ACTION_NAMES.index("eat"))


if __name__ == "__main__":
    unittest.main()
