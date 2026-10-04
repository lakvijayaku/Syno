"""
Unit tests for syno.brain.layer.

These tests are intentionally stricter than the Layer's own safeguards. They
verify that the layer passes the same inputs to every neuron, preserves neuron
order, propagates input errors from Neuron.forward, and does not share or alter
outside data.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import math
import unittest

from syno.brain.layer import Layer
from syno.brain.neuron import Neuron


def reference_sigmoid(z: float) -> float:
    """
    Computes the sigmoid of z independently of the Neuron class.

    Only used for moderate values of z, where the standard formula cannot
    overflow.
    """
    return 1 / (1 + math.exp(-z))


class TestLayerOutput(unittest.TestCase):
    """Verifies that the layer computes the correct outputs."""

    def test_returns_one_output_per_neuron(self):
        layer = Layer([Neuron([1.0, 1.0], 0.0) for _ in range(3)])
        self.assertEqual(len(layer.forward([0.5, 0.5])), 3)

    def test_returns_a_list_of_floats(self):
        layer = Layer([Neuron([1.0], 0.0), Neuron([2.0], 0.0)])
        outputs = layer.forward([1.0])
        self.assertIsInstance(outputs, list)
        for output in outputs:
            self.assertIsInstance(output, float)

    def test_zero_inputs_and_zero_biases_return_halves(self):
        layer = Layer([Neuron([1.0, 1.0], 0.0), Neuron([-1.0, -1.0], 0.0)])
        self.assertEqual(layer.forward([0.0, 0.0]), [0.5, 0.5])

    def test_each_output_matches_its_neuron(self):
        # The layer must produce exactly what each neuron produces on its own.
        neurons = [
            Neuron([0.5, -1.0], 0.0),
            Neuron([2.0, 1.0], -1.0),
            Neuron([-0.3, 0.8], 0.25),
        ]
        inputs = [2.0, 3.0]
        layer = Layer(neurons)
        expected = [neuron.forward(inputs) for neuron in neurons]
        self.assertEqual(layer.forward(inputs), expected)

    def test_known_values(self):
        # Weighted sums: 0.5*2 + (-1)*3 + 0 = -2, and 2*2 + 1*3 - 1 = 6.
        layer = Layer([Neuron([0.5, -1.0], 0.0), Neuron([2.0, 1.0], -1.0)])
        outputs = layer.forward([2.0, 3.0])
        self.assertAlmostEqual(outputs[0], reference_sigmoid(-2.0))
        self.assertAlmostEqual(outputs[1], reference_sigmoid(6.0))

    def test_outputs_keep_neuron_order(self):
        # A strongly negative neuron first and a strongly positive one second
        # must produce outputs in that same order.
        layer = Layer([Neuron([-10.0], 0.0), Neuron([10.0], 0.0)])
        low, high = layer.forward([1.0])
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_single_neuron_layer_returns_single_item_list(self):
        # A one-neuron layer still returns a list, not a bare float.
        layer = Layer([Neuron([1.0], 0.0)])
        self.assertEqual(layer.forward([0.0]), [0.5])

    def test_empty_layer_returns_empty_list(self):
        layer = Layer([])
        self.assertEqual(layer.forward([1.0, 2.0]), [])

    def test_forward_is_repeatable(self):
        layer = Layer([Neuron([0.3, 0.7], -0.1), Neuron([1.0, -1.0], 0.2)])
        self.assertEqual(layer.forward([1.0, 2.0]), layer.forward([1.0, 2.0]))


class TestLayerInputValidation(unittest.TestCase):
    """Verifies that invalid inputs fail loudly, via Neuron.forward."""

    def test_too_few_inputs_raise_value_error(self):
        layer = Layer([Neuron([1.0, 1.0], 0.0)])
        with self.assertRaises(ValueError):
            layer.forward([1.0])

    def test_too_many_inputs_raise_value_error(self):
        layer = Layer([Neuron([1.0], 0.0)])
        with self.assertRaises(ValueError):
            layer.forward([1.0, 2.0])

    def test_one_mismatched_neuron_fails_the_whole_layer(self):
        # If any neuron expects a different number of inputs, the layer must
        # raise rather than return partial results.
        layer = Layer([Neuron([1.0, 1.0], 0.0), Neuron([1.0, 1.0, 1.0], 0.0)])
        with self.assertRaises(ValueError):
            layer.forward([1.0, 2.0])


class TestLayerStateIsolation(unittest.TestCase):
    """Verifies that the layer does not share or alter outside data."""

    def test_changing_original_list_does_not_affect_layer(self):
        neurons = [Neuron([1.0], 0.0)]
        layer = Layer(neurons)
        neurons.append(Neuron([2.0], 0.0))
        self.assertEqual(len(layer.neurons), 1)

    def test_layer_stores_a_different_list_object(self):
        neurons = [Neuron([1.0], 0.0)]
        layer = Layer(neurons)
        self.assertIsNot(layer.neurons, neurons)

    def test_layer_keeps_the_same_neuron_objects(self):
        # The copy is shallow on purpose: the layer must use the very same
        # neurons, so that training a neuron later updates the layer too.
        neuron = Neuron([1.0], 0.0)
        layer = Layer([neuron])
        self.assertIs(layer.neurons[0], neuron)

    def test_forward_does_not_modify_inputs(self):
        layer = Layer([Neuron([0.5, -1.0], 0.0), Neuron([1.0, 1.0], 0.0)])
        inputs = [2.0, 3.0]
        layer.forward(inputs)
        self.assertEqual(inputs, [2.0, 3.0])

    def test_forward_does_not_modify_neurons(self):
        neuron = Neuron([0.5, -1.0], 0.25)
        layer = Layer([neuron])
        layer.forward([2.0, 3.0])
        self.assertEqual(neuron.weights, [0.5, -1.0])
        self.assertEqual(neuron.bias, 0.25)


if __name__ == "__main__":
    unittest.main()
