"""
Experiment tests for experiments/curiosity.py.

The fast tests check that the curiosity experiment differs from the hunger
experiment only where intended. The slow test trains SYNO for the full 5000
lives (about 25 seconds) and reproduces the recorded energy curve. It is
skipped by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

import experiments.curiosity as curiosity
import experiments.hunger as hunger
from experiments.xor import make_layer
from syno.brain.network import Network


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * hunger.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(hunger.HIDDEN_NEURONS, inputs),
        make_layer(len(hunger.ACTION_NAMES), hunger.HIDDEN_NEURONS, "linear"),
    ])


class TestCuriositySetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_explores_less_randomly_than_hunger(self):
        self.assertLess(curiosity.EPSILON, hunger.EPSILON)
        self.assertGreater(curiosity.EPSILON, 0.0)

    def test_shares_the_hunger_settings(self):
        for name in ("GRID_SIZE", "LEARNING_RATE", "DISCOUNT", "REWARD_SCALE", "LIVES", "LIFE_STEPS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(curiosity, name), getattr(hunger, name))

    def test_live_returns_an_average_energy(self):
        random.seed(3)
        network = new_network()
        for _ in range(5):
            energy = curiosity.live(network)
            self.assertGreaterEqual(energy, 0.0)
            self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestCuriosityLearning(unittest.TestCase):
    """Verifies that curiosity lets SYNO learn with little randomness (slow)."""

    def test_seed_zero_matches_the_recorded_energy_curve(self):
        random.seed(0)
        network = new_network()
        energies = [curiosity.live(network) for _ in range(curiosity.LIVES)]
        recorded = [0.187, 0.196, 0.189, 0.215, 0.411, 0.860, 0.906, 0.897, 0.904, 0.863]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                average = sum(energies[block * 500:(block + 1) * 500]) / 500
                self.assertAlmostEqual(average, expected, places=3)


if __name__ == "__main__":
    unittest.main()
