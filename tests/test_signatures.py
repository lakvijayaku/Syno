"""
Experiment tests for experiments/signatures.py.

The fast tests probe eating_rate with hand-built networks whose preferences
are known exactly, and check that train is reproducible. The slow test trains
SYNO with all 5 seeds (about 2 minutes) and reproduces the recorded E2 table.
It is skipped by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import unittest

import experiments.signatures as signatures
from syno.brain.layer import Layer
from syno.brain.network import Network
from syno.brain.neuron import Neuron

INPUTS = (2 * signatures.SENSE_RADIUS + 1) ** 2 + 2
CENTER = ((2 * signatures.SENSE_RADIUS + 1) ** 2) // 2
DEFICIT = INPUTS - 2
STOMACH = INPUTS - 1
EAT = signatures.ACTION_NAMES.index("eat")


def probe_network(eat_weights: dict[int, float], eat_bias: float) -> Network:
    """
    Builds a one-layer linear network where every action outputs 0.0 except
    eat, whose output is eat_bias plus the given weights on chosen inputs.
    """
    neurons = []
    for action in range(len(signatures.ACTION_NAMES)):
        weights = [0.0] * INPUTS
        bias = 0.0
        if action == EAT:
            for index, weight in eat_weights.items():
                weights[index] = weight
            bias = eat_bias
        neurons.append(Neuron(weights, bias, "linear"))
    return Network([Layer(neurons)])


class TestSignaturesSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_energy_levels_span_empty_to_full(self):
        self.assertEqual(len(signatures.ENERGY_LEVELS), 11)
        self.assertEqual(signatures.ENERGY_LEVELS[0], 0.0)
        self.assertEqual(signatures.ENERGY_LEVELS[-1], 1.0)
        for low, high in zip(signatures.ENERGY_LEVELS, signatures.ENERGY_LEVELS[1:]):
            self.assertAlmostEqual(high - low, 0.1)

    def test_uses_several_seeds(self):
        self.assertGreaterEqual(signatures.SEEDS, 3)

    def test_trains_long_enough(self):
        self.assertGreaterEqual(signatures.TRAINING_LIVES, 1000)


class TestEatingRate(unittest.TestCase):
    """Verifies eating_rate against networks with known preferences."""

    def test_always_eats(self):
        network = probe_network({}, 1.0)
        for energy in (0.0, 0.5, 1.0):
            self.assertEqual(signatures.eating_rate(network, energy), 1.0)

    def test_never_eats(self):
        network = probe_network({}, -1.0)
        for energy in (0.0, 0.5, 1.0):
            self.assertEqual(signatures.eating_rate(network, energy), 0.0)

    def test_eats_only_when_hungry(self):
        # Eat's value is deficit - 0.45, so eating is preferred only when
        # energy is below 0.55.
        network = probe_network({DEFICIT: 1.0}, -0.45)
        for energy in (0.0, 0.2, 0.5):
            self.assertEqual(signatures.eating_rate(network, energy), 1.0)
        for energy in (0.6, 0.8, 1.0):
            self.assertEqual(signatures.eating_rate(network, energy), 0.0)

    def test_food_is_under_syno_in_every_square(self):
        # Eat wins only if the center of the view shows food.
        network = probe_network({CENTER: 1.0}, -0.5)
        self.assertEqual(signatures.eating_rate(network, 0.5), 1.0)

    def test_stomach_is_empty(self):
        # Eat wins only if the stomach input is 0.
        network = probe_network({STOMACH: -1.0}, 0.001)
        self.assertEqual(signatures.eating_rate(network, 0.5), 1.0)

    def test_counts_each_square_separately(self):
        # Eat wins only where a wall is directly above SYNO (the top row), so
        # 3 of the 9 squares eat.
        above = CENTER - (2 * signatures.SENSE_RADIUS + 1)
        network = probe_network({above: -1.0}, -0.5)
        self.assertAlmostEqual(signatures.eating_rate(network, 0.5), 3 / 9)

    def test_does_not_explore(self):
        network = probe_network({}, 0.001)
        for _ in range(20):
            self.assertEqual(signatures.eating_rate(network, 0.5), 1.0)


class TestTrain(unittest.TestCase):
    """Verifies that train builds and trains a reproducible network."""

    def setUp(self):
        self.saved = signatures.TRAINING_LIVES
        signatures.TRAINING_LIVES = 3

    def tearDown(self):
        signatures.TRAINING_LIVES = self.saved

    def test_network_shape(self):
        network = signatures.train(0)
        self.assertEqual(len(network.layers), 2)
        self.assertEqual(len(network.layers[0].neurons), signatures.HIDDEN_NEURONS)
        self.assertEqual(len(network.layers[0].neurons[0].weights), INPUTS)
        self.assertEqual(len(network.layers[1].neurons), len(signatures.ACTION_NAMES))
        self.assertEqual(network.layers[1].neurons[0].activation, "linear")

    def test_same_seed_same_network(self):
        state = [0.0] * INPUTS
        self.assertEqual(signatures.train(4).forward(state), signatures.train(4).forward(state))

    def test_different_seeds_differ(self):
        state = [0.0] * INPUTS
        self.assertNotEqual(signatures.train(0).forward(state), signatures.train(1).forward(state))

    def test_training_changes_the_network(self):
        state = [0.0] * INPUTS
        signatures.TRAINING_LIVES = 0
        untrained = signatures.train(0).forward(state)
        signatures.TRAINING_LIVES = 3
        self.assertNotEqual(untrained, signatures.train(0).forward(state))


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestSignaturesRun(unittest.TestCase):
    """Reproduces the recorded E2 result."""

    def test_e2_table(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            signatures.main()
        expected = ["E2: State-dependent eating (5 seeds)"]
        for level in signatures.ENERGY_LEVELS:
            rate = "0.73" if level == 1.0 else "1.00"
            expected.append(f"Energy {level:.1f} | Eats: {rate}")
        self.assertEqual(output.getvalue().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
