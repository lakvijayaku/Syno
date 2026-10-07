"""
Unit tests for syno.brain.network.

These tests are intentionally stricter than the Network's own safeguards. They
verify that layers are applied in order, that each layer's outputs feed the
next layer, that size mismatches between layers fail loudly, and that the
network does not share or alter outside data.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import math
import unittest

from syno.brain.layer import Layer
from syno.brain.network import Network
from syno.brain.neuron import Neuron


def reference_sigmoid(z: float) -> float:
    """
    Computes the sigmoid of z independently of the Neuron class.

    Only used for moderate values of z, where the standard formula cannot
    overflow.
    """
    return 1 / (1 + math.exp(-z))


def build_example_network() -> Network:
    """
    Builds the 2-2-1 network used in the step 2b hand check.

    Layer 1 outputs [0.5, 0.5] for zero inputs, so Layer 2's weighted sum is
    0.5 + 0.5 = 1.0 and the network outputs sigmoid(1.0).
    """
    hidden = Layer([Neuron([1.0, 1.0], 0.0), Neuron([-1.0, -1.0], 0.0)])
    output = Layer([Neuron([1.0, 1.0], 0.0)])
    return Network([hidden, output])


class TestNetworkOutput(unittest.TestCase):
    """Verifies that the network computes the correct outputs."""

    def test_hand_check_example(self):
        network = build_example_network()
        outputs = network.forward([0.0, 0.0])
        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0], reference_sigmoid(1.0))

    def test_output_length_matches_last_layer(self):
        network = Network([
            Layer([Neuron([1.0], 0.0) for _ in range(4)]),
            Layer([Neuron([1.0] * 4, 0.0) for _ in range(3)]),
        ])
        self.assertEqual(len(network.forward([1.0])), 3)

    def test_single_layer_network_matches_the_layer(self):
        layer = Layer([Neuron([0.5, -1.0], 0.0), Neuron([2.0, 1.0], -1.0)])
        network = Network([layer])
        self.assertEqual(network.forward([2.0, 3.0]), layer.forward([2.0, 3.0]))

    def test_matches_manual_layer_chaining(self):
        # The network must give exactly the same result as feeding each
        # layer's output into the next layer by hand.
        first = Layer([Neuron([0.2, -0.4], 0.1), Neuron([0.7, 0.3], -0.2)])
        second = Layer([Neuron([1.5, -0.5], 0.0), Neuron([-1.0, 2.0], 0.3)])
        third = Layer([Neuron([0.6, 0.9], -0.1)])
        inputs = [1.0, -2.0]
        expected = third.forward(second.forward(first.forward(inputs)))
        network = Network([first, second, third])
        self.assertEqual(network.forward(inputs), expected)

    def test_layer_order_matters(self):
        # Swapping two layers must change the result, which proves the layers
        # are applied in the order given.
        a = Layer([Neuron([2.0], 1.0)])
        b = Layer([Neuron([-3.0], 0.5)])
        self.assertNotEqual(
            Network([a, b]).forward([1.0]),
            Network([b, a]).forward([1.0]),
        )

    def test_returns_only_final_layer_outputs(self):
        # The result must be the last layer's outputs, not a list of every
        # layer's outputs.
        network = build_example_network()
        outputs = network.forward([0.0, 0.0])
        for output in outputs:
            self.assertIsInstance(output, float)

    def test_empty_network_returns_inputs_unchanged(self):
        network = Network([])
        self.assertEqual(network.forward([1.0, 2.0]), [1.0, 2.0])

    def test_forward_is_repeatable(self):
        network = build_example_network()
        self.assertEqual(network.forward([0.3, -0.7]), network.forward([0.3, -0.7]))


class TestNetworkInputValidation(unittest.TestCase):
    """Verifies that size mismatches fail loudly, via Neuron.forward."""

    def test_wrong_number_of_network_inputs_raises_value_error(self):
        network = build_example_network()
        with self.assertRaises(ValueError):
            network.forward([1.0])

    def test_mismatched_layer_sizes_raise_value_error(self):
        # Layer 1 has 2 neurons, but Layer 2's neuron expects 3 inputs.
        network = Network([
            Layer([Neuron([1.0], 0.0), Neuron([1.0], 0.0)]),
            Layer([Neuron([1.0, 1.0, 1.0], 0.0)]),
        ])
        with self.assertRaises(ValueError):
            network.forward([1.0])


class TestNetworkStateIsolation(unittest.TestCase):
    """Verifies that the network does not share or alter outside data."""

    def test_changing_original_list_does_not_affect_network(self):
        layers = [Layer([Neuron([1.0], 0.0)])]
        network = Network(layers)
        layers.append(Layer([Neuron([1.0], 0.0)]))
        self.assertEqual(len(network.layers), 1)

    def test_network_stores_a_different_list_object(self):
        layers = [Layer([Neuron([1.0], 0.0)])]
        network = Network(layers)
        self.assertIsNot(network.layers, layers)

    def test_network_keeps_the_same_layer_objects(self):
        # The copy is shallow on purpose: training must update the very same
        # layers and neurons that the network uses.
        layer = Layer([Neuron([1.0], 0.0)])
        network = Network([layer])
        self.assertIs(network.layers[0], layer)

    def test_forward_does_not_modify_inputs(self):
        network = build_example_network()
        inputs = [0.4, -0.6]
        network.forward(inputs)
        self.assertEqual(inputs, [0.4, -0.6])

    def test_forward_does_not_modify_weights_or_biases(self):
        network = build_example_network()
        before = [
            [(list(n.weights), n.bias) for n in layer.neurons]
            for layer in network.layers
        ]
        network.forward([0.4, -0.6])
        after = [
            [(list(n.weights), n.bias) for n in layer.neurons]
            for layer in network.layers
        ]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
