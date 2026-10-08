"""
Experiment tests for experiments/xor.py.

These tests confirm that backpropagation can train a hidden layer to solve
XOR, a problem a single neuron cannot solve. They check every case across
several random seeds, so a lucky starting point cannot hide a broken
training step.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import random
import unittest

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.neuron import Neuron
from syno.brain.training import train_network_step, train_step

XOR_DATA = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def train_xor(seed: int, epochs: int = 5000) -> Network:
    """Trains a 2-3-1 network on XOR with the same setup as the experiment."""
    random.seed(seed)
    network = Network([make_layer(3, 2), make_layer(1, 3)])
    for _ in range(epochs):
        for inputs, targets in XOR_DATA:
            train_network_step(network, inputs, targets, 1.0)
    return network


class TestMakeLayer(unittest.TestCase):
    """Verifies the experiment's random layer builder."""

    def test_layer_has_requested_shape(self):
        layer = make_layer(3, 2)
        self.assertEqual(len(layer.neurons), 3)
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 2)

    def test_values_are_within_range(self):
        random.seed(1)
        layer = make_layer(5, 4)
        for neuron in layer.neurons:
            for value in neuron.weights + [neuron.bias]:
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)

    def test_neurons_start_different(self):
        # Identical neurons would always receive identical updates.
        random.seed(2)
        layer = make_layer(3, 2)
        starts = [(tuple(n.weights), n.bias) for n in layer.neurons]
        self.assertEqual(len(set(starts)), 3)

    def test_same_seed_gives_same_layer(self):
        random.seed(3)
        first = make_layer(2, 2)
        random.seed(3)
        second = make_layer(2, 2)
        for a, b in zip(first.neurons, second.neurons):
            self.assertEqual(a.weights, b.weights)
            self.assertEqual(a.bias, b.bias)


class TestXorExperiment(unittest.TestCase):
    """Verifies that the network learns XOR."""

    def test_seed_zero_matches_the_recorded_result(self):
        network = train_xor(0, epochs=5001)
        outputs = [round(network.forward(inputs)[0], 3) for inputs, _ in XOR_DATA]
        self.assertEqual(outputs, [0.013, 0.989, 0.989, 0.011])

    def test_solves_xor_across_seeds(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                network = train_xor(seed)
                for inputs, targets in XOR_DATA:
                    self.assertAlmostEqual(network.forward(inputs)[0], targets[0], delta=0.1)

    def test_single_neuron_cannot_solve_xor(self):
        # The reason XOR needs a hidden layer: after the same amount of
        # training, a lone neuron still gets at least one case wrong.
        random.seed(0)
        neuron = Neuron([random.uniform(-1.0, 1.0) for _ in range(2)], random.uniform(-1.0, 1.0))
        for _ in range(5000):
            for inputs, targets in XOR_DATA:
                train_step(neuron, inputs, targets[0], 1.0)
        wrong = [
            inputs for inputs, targets in XOR_DATA
            if abs(neuron.forward(inputs) - targets[0]) >= 0.5
        ]
        self.assertGreater(len(wrong), 0)


if __name__ == "__main__":
    unittest.main()
