"""
Unit tests for syno.brain.growth.

These tests are intentionally stricter than grow_neuron's own safeguards.
They verify that growing never changes the network's output, that the new
neuron joins in through training, that a network too small for XOR solves it
after growing, and that only hidden layers can grow.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import random
import unittest

from experiments.xor import make_layer
from syno.brain.growth import grow_neuron
from syno.brain.layer import Layer
from syno.brain.network import Network
from syno.brain.neuron import Neuron
from syno.brain.training import train_network_step

XOR_DATA = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def train_xor(network: Network, epochs: int) -> None:
    """Trains a network on XOR for a number of epochs."""
    for _ in range(epochs):
        for inputs, targets in XOR_DATA:
            train_network_step(network, inputs, targets, 1.0)


class TestGrowNeuronStructure(unittest.TestCase):
    """Verifies the shape of the network after growing."""

    def setUp(self):
        random.seed(0)
        self.network = Network([make_layer(2, 3), make_layer(4, 2), make_layer(1, 4)])

    def test_layer_gains_one_neuron(self):
        grow_neuron(self.network, 0)
        self.assertEqual(len(self.network.layers[0].neurons), 3)

    def test_new_neuron_has_one_weight_per_input(self):
        grow_neuron(self.network, 0)
        self.assertEqual(len(self.network.layers[0].neurons[-1].weights), 3)

    def test_next_layer_gains_one_zero_weight_each(self):
        grow_neuron(self.network, 0)
        for neuron in self.network.layers[1].neurons:
            self.assertEqual(len(neuron.weights), 3)
            self.assertEqual(neuron.weights[-1], 0.0)

    def test_existing_weights_are_unchanged(self):
        before = [[(list(n.weights), n.bias) for n in layer.neurons] for layer in self.network.layers]
        grow_neuron(self.network, 1)
        for layer_index, layer in enumerate(before):
            for neuron_index, (weights, bias) in enumerate(layer):
                neuron = self.network.layers[layer_index].neurons[neuron_index]
                self.assertEqual(neuron.weights[:len(weights)], weights)
                self.assertEqual(neuron.bias, bias)

    def test_layers_further_away_are_unchanged(self):
        output_weights = list(self.network.layers[2].neurons[0].weights)
        grow_neuron(self.network, 0)
        self.assertEqual(self.network.layers[2].neurons[0].weights, output_weights)

    def test_new_neuron_values_are_in_range(self):
        grow_neuron(self.network, 0)
        neuron = self.network.layers[0].neurons[-1]
        for value in neuron.weights + [neuron.bias]:
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_new_neuron_matches_the_layer_activation(self):
        network = Network([
            Layer([Neuron([0.1, 0.2], 0.0, "linear")]),
            Layer([Neuron([0.5], 0.0)]),
        ])
        grow_neuron(network, 0)
        self.assertEqual(network.layers[0].neurons[-1].activation, "linear")

    def test_growing_repeatedly(self):
        for _ in range(5):
            grow_neuron(self.network, 1)
        self.assertEqual(len(self.network.layers[1].neurons), 9)
        self.assertEqual(len(self.network.layers[2].neurons[0].weights), 9)


class TestGrowNeuronPreservesFunction(unittest.TestCase):
    """Verifies that growing never changes what the network outputs."""

    def test_output_is_identical_for_many_inputs(self):
        random.seed(1)
        network = Network([make_layer(3, 2), make_layer(2, 3, "linear")])
        inputs = [[random.uniform(-2, 2), random.uniform(-2, 2)] for _ in range(20)]
        before = [network.forward(x) for x in inputs]
        for _ in range(3):
            grow_neuron(network, 0)
        self.assertEqual([network.forward(x) for x in inputs], before)

    def test_new_neuron_joins_through_training(self):
        # The outgoing weight learns on the first step; the incoming weights
        # follow once that weight is no longer zero.
        random.seed(2)
        network = Network([make_layer(2, 2), make_layer(1, 2)])
        grow_neuron(network, 0)
        new_neuron = network.layers[0].neurons[-1]
        incoming_before = list(new_neuron.weights)
        train_network_step(network, [1.0, 0.0], [1.0], 1.0)
        self.assertNotEqual(network.layers[1].neurons[0].weights[-1], 0.0)
        self.assertEqual(new_neuron.weights, incoming_before)
        train_network_step(network, [1.0, 0.0], [1.0], 1.0)
        self.assertNotEqual(new_neuron.weights, incoming_before)

    def test_step_8a_too_small_network_solves_xor_after_growing(self):
        random.seed(0)
        network = Network([make_layer(1, 2), make_layer(1, 1)])
        train_xor(network, 3000)
        stuck = [round(network.forward(x)[0], 2) for x, _ in XOR_DATA]
        self.assertEqual(stuck, [0.02, 0.65, 0.65, 0.65])
        grow_neuron(network, 0)
        grow_neuron(network, 0)
        train_xor(network, 5000)
        solved = [round(network.forward(x)[0], 2) for x, _ in XOR_DATA]
        self.assertEqual(solved, [0.01, 0.99, 0.99, 0.01])


class TestGrowNeuronValidation(unittest.TestCase):
    """Verifies that only hidden layers can grow."""

    def test_output_and_out_of_range_layers_raise_value_error(self):
        random.seed(3)
        network = Network([make_layer(2, 2), make_layer(1, 2)])
        for layer_index in (1, 2, -1):
            with self.subTest(layer_index=layer_index):
                with self.assertRaises(ValueError):
                    grow_neuron(network, layer_index)

    def test_single_layer_network_cannot_grow(self):
        network = Network([Layer([Neuron([0.1], 0.0)])])
        with self.assertRaises(ValueError):
            grow_neuron(network, 0)

    def test_failed_growth_changes_nothing(self):
        random.seed(4)
        network = Network([make_layer(2, 2), make_layer(1, 2)])
        with self.assertRaises(ValueError):
            grow_neuron(network, 1)
        self.assertEqual(len(network.layers[1].neurons), 1)
        self.assertEqual(len(network.layers[1].neurons[0].weights), 2)


if __name__ == "__main__":
    unittest.main()
