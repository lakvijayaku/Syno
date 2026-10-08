"""
Unit tests for syno.brain.training.

These tests are intentionally stricter than the training functions' own
safeguards. Beyond checking that the loss goes down, they compare every update
against the exact worked examples from steps 2d and 2e and against slopes
measured numerically, so a step with the wrong size is caught, not just a step
in the wrong direction.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.layer import Layer
from syno.brain.loss import mean_squared_error
from syno.brain.network import Network
from syno.brain.neuron import Neuron
from syno.brain.training import train_network_step, train_step


def numerical_slopes(neuron: Neuron, inputs: list[float], target: float) -> tuple[list[float], float]:
    """
    Measures the slope of the loss for each weight and the bias by nudging
    each one slightly up and down and comparing the losses.

    This is independent of the chain-rule formula in train_step, so it can
    catch a formula that is wrong but still points downhill.
    """
    step = 1e-6

    def loss_with(weights: list[float], bias: float) -> float:
        return mean_squared_error([Neuron(weights, bias).forward(inputs)], [target])

    weight_slopes = []
    for i in range(len(neuron.weights)):
        up = list(neuron.weights)
        down = list(neuron.weights)
        up[i] += step
        down[i] -= step
        weight_slopes.append((loss_with(up, neuron.bias) - loss_with(down, neuron.bias)) / (2 * step))
    bias_slope = (
        loss_with(neuron.weights, neuron.bias + step) - loss_with(neuron.weights, neuron.bias - step)
    ) / (2 * step)
    return weight_slopes, bias_slope


class TestTrainStepWorkedExample(unittest.TestCase):
    """Verifies the exact worked example from step 2d."""

    def setUp(self):
        self.neuron = Neuron([0.0], 0.0)
        self.loss = train_step(self.neuron, [1.0], 1.0, 1.0)

    def test_returns_loss_before_update(self):
        self.assertAlmostEqual(self.loss, 0.25)

    def test_weight_after_update(self):
        self.assertAlmostEqual(self.neuron.weights[0], 0.25)

    def test_bias_after_update(self):
        self.assertAlmostEqual(self.neuron.bias, 0.25)

    def test_output_moves_toward_target(self):
        self.assertGreater(self.neuron.forward([1.0]), 0.5)


class TestTrainStepGradient(unittest.TestCase):
    """Verifies that each update follows the true slope of the loss."""

    def test_updates_match_numerical_slopes(self):
        cases = [
            ([0.3, -0.2], 0.1, [1.0, 0.5], 0.2),
            ([1.5, 0.7, -0.4], -0.3, [0.2, -1.0, 2.0], 0.9),
            ([-0.8], 0.6, [3.0], 0.0),
        ]
        learning_rate = 0.1
        for weights, bias, inputs, target in cases:
            with self.subTest(weights=weights, inputs=inputs, target=target):
                neuron = Neuron(weights, bias)
                weight_slopes, bias_slope = numerical_slopes(neuron, inputs, target)
                train_step(neuron, inputs, target, learning_rate)
                for i, slope in enumerate(weight_slopes):
                    self.assertAlmostEqual(neuron.weights[i], weights[i] - learning_rate * slope, places=6)
                self.assertAlmostEqual(neuron.bias, bias - learning_rate * bias_slope, places=6)

    def test_zero_input_leaves_its_weight_unchanged(self):
        # A weight whose input is 0 had no effect on the output, so it gets
        # no blame and must not move.
        neuron = Neuron([0.5, 0.5], 0.0)
        train_step(neuron, [0.0, 1.0], 1.0, 1.0)
        self.assertEqual(neuron.weights[0], 0.5)
        self.assertNotEqual(neuron.weights[1], 0.5)

    def test_perfect_output_changes_nothing(self):
        # sigmoid(0) = 0.5, so with target 0.5 the loss is 0 and so is delta.
        neuron = Neuron([1.0], 0.0)
        train_step(neuron, [0.0], 0.5, 1.0)
        self.assertEqual(neuron.weights, [1.0])
        self.assertEqual(neuron.bias, 0.0)

    def test_zero_learning_rate_changes_nothing(self):
        neuron = Neuron([0.3, -0.2], 0.1)
        train_step(neuron, [1.0, 0.5], 0.9, 0.0)
        self.assertEqual(neuron.weights, [0.3, -0.2])
        self.assertEqual(neuron.bias, 0.1)

    def test_output_too_low_increases_bias(self):
        neuron = Neuron([0.0], 0.0)
        train_step(neuron, [1.0], 1.0, 0.5)
        self.assertGreater(neuron.bias, 0.0)

    def test_output_too_high_decreases_bias(self):
        neuron = Neuron([0.0], 0.0)
        train_step(neuron, [1.0], 0.0, 0.5)
        self.assertLess(neuron.bias, 0.0)


class TestTrainStepLearning(unittest.TestCase):
    """Verifies that repeated steps actually teach the neuron."""

    def test_loss_decreases_every_step(self):
        neuron = Neuron([0.0], 0.0)
        losses = [train_step(neuron, [1.0], 1.0, 1.0) for _ in range(50)]
        for previous, current in zip(losses, losses[1:]):
            self.assertLess(current, previous)

    def test_learns_a_target_between_zero_and_one(self):
        neuron = Neuron([0.3, -0.2], 0.1)
        for _ in range(5000):
            train_step(neuron, [1.0, 0.5], 0.2, 0.5)
        self.assertAlmostEqual(neuron.forward([1.0, 0.5]), 0.2, places=4)

    def test_learns_two_inputs_at_once(self):
        # One neuron can learn to output high for one input and low for
        # another, as long as a single straight line separates them.
        neuron = Neuron([0.0, 0.0], 0.0)
        for _ in range(5000):
            train_step(neuron, [1.0, 0.0], 1.0, 1.0)
            train_step(neuron, [0.0, 1.0], 0.0, 1.0)
        self.assertGreater(neuron.forward([1.0, 0.0]), 0.9)
        self.assertLess(neuron.forward([0.0, 1.0]), 0.1)


class TestTrainStepSafety(unittest.TestCase):
    """Verifies that invalid input fails loudly and changes nothing."""

    def test_mismatched_inputs_raise_value_error(self):
        neuron = Neuron([1.0, 1.0], 0.0)
        with self.assertRaises(ValueError):
            train_step(neuron, [1.0], 1.0, 0.1)

    def test_failed_step_leaves_neuron_unchanged(self):
        neuron = Neuron([1.0, 1.0], 0.5)
        with self.assertRaises(ValueError):
            train_step(neuron, [1.0, 2.0, 3.0], 1.0, 0.1)
        self.assertEqual(neuron.weights, [1.0, 1.0])
        self.assertEqual(neuron.bias, 0.5)

    def test_inputs_are_not_modified(self):
        inputs = [1.0, 0.5]
        train_step(Neuron([0.3, -0.2], 0.1), inputs, 0.9, 0.5)
        self.assertEqual(inputs, [1.0, 0.5])

    def test_returns_a_float(self):
        self.assertIsInstance(train_step(Neuron([0.0], 0.0), [1.0], 1.0, 1.0), float)



def copy_network(network: Network) -> Network:
    """Builds an independent copy of a network with the same weights and biases."""
    return Network([
        Layer([Neuron(list(n.weights), n.bias) for n in layer.neurons])
        for layer in network.layers
    ])


def network_slopes(network: Network, inputs: list[float], targets: list[float]) -> list[list[tuple[list[float], float]]]:
    """
    Measures the slope of the loss for every weight and bias in a network by
    nudging each one slightly up and down and comparing the losses.

    This is independent of the backpropagation code, so it can catch a
    formula that is wrong but still points downhill.
    """
    step = 1e-6

    def loss_of(net: Network) -> float:
        return mean_squared_error(net.forward(inputs), targets)

    slopes = []
    for layer_index, layer in enumerate(network.layers):
        layer_slopes = []
        for neuron_index, neuron in enumerate(layer.neurons):
            weight_slopes = []
            for i in range(len(neuron.weights)):
                up = copy_network(network)
                down = copy_network(network)
                up.layers[layer_index].neurons[neuron_index].weights[i] += step
                down.layers[layer_index].neurons[neuron_index].weights[i] -= step
                weight_slopes.append((loss_of(up) - loss_of(down)) / (2 * step))
            up = copy_network(network)
            down = copy_network(network)
            up.layers[layer_index].neurons[neuron_index].bias += step
            down.layers[layer_index].neurons[neuron_index].bias -= step
            layer_slopes.append((weight_slopes, (loss_of(up) - loss_of(down)) / (2 * step)))
        slopes.append(layer_slopes)
    return slopes


def snapshot(network: Network) -> list[list[tuple[list[float], float]]]:
    """Records every weight and bias in a network."""
    return [[(list(n.weights), n.bias) for n in layer.neurons] for layer in network.layers]


class TestTrainNetworkStepWorkedExample(unittest.TestCase):
    """Verifies the exact worked example from step 2e."""

    def setUp(self):
        self.hidden = Neuron([0.0], 0.0)
        self.output = Neuron([1.0], 0.0)
        self.network = Network([Layer([self.hidden]), Layer([self.output])])
        self.loss = train_network_step(self.network, [1.0], [1.0], 1.0)

    def test_returns_loss_before_update(self):
        self.assertAlmostEqual(self.loss, 0.1425369565965509)

    def test_output_neuron_update(self):
        self.assertAlmostEqual(self.output.weights[0], 1.0887234586746368)
        self.assertAlmostEqual(self.output.bias, 0.17744691734927373)

    def test_hidden_neuron_update(self):
        # The hidden neuron never touches the loss directly, so this proves
        # the blame was passed back.
        self.assertAlmostEqual(self.hidden.weights[0], 0.04436172933731843)
        self.assertAlmostEqual(self.hidden.bias, 0.04436172933731843)

    def test_output_moves_toward_target(self):
        self.assertAlmostEqual(self.network.forward([1.0])[0], 0.6782937617635624)


class TestTrainNetworkStepGradient(unittest.TestCase):
    """Verifies that every update follows the true slope of the loss."""

    def build_network(self) -> Network:
        # A 2-3-2 network: two hidden neurons would hide index mix-ups between
        # j and k, so the layer sizes are all different from each other.
        return Network([
            Layer([
                Neuron([0.5, -0.3], 0.1),
                Neuron([-0.7, 0.8], -0.2),
                Neuron([0.2, 0.4], 0.05),
            ]),
            Layer([
                Neuron([0.6, -0.4, 0.9], 0.3),
                Neuron([-0.5, 0.7, 0.1], -0.1),
            ]),
        ])

    def test_two_layer_updates_match_numerical_slopes(self):
        network = self.build_network()
        inputs, targets, learning_rate = [1.0, -0.5], [0.9, 0.1], 0.1
        before = snapshot(network)
        slopes = network_slopes(network, inputs, targets)
        train_network_step(network, inputs, targets, learning_rate)
        after = snapshot(network)
        for layer_index in range(len(before)):
            for neuron_index in range(len(before[layer_index])):
                with self.subTest(layer=layer_index, neuron=neuron_index):
                    old_weights, old_bias = before[layer_index][neuron_index]
                    new_weights, new_bias = after[layer_index][neuron_index]
                    weight_slopes, bias_slope = slopes[layer_index][neuron_index]
                    for i in range(len(old_weights)):
                        self.assertAlmostEqual(new_weights[i], old_weights[i] - learning_rate * weight_slopes[i], places=6)
                    self.assertAlmostEqual(new_bias, old_bias - learning_rate * bias_slope, places=6)

    def test_three_layer_updates_match_numerical_slopes(self):
        # A third layer checks that blame is passed back more than once.
        network = Network([
            Layer([Neuron([0.4, -0.6], 0.2), Neuron([0.3, 0.9], -0.1)]),
            Layer([Neuron([0.7, -0.2], 0.0), Neuron([-0.5, 0.6], 0.1), Neuron([0.8, 0.3], -0.3)]),
            Layer([Neuron([0.5, -0.9, 0.4], 0.2)]),
        ])
        inputs, targets, learning_rate = [0.5, 1.5], [0.25], 0.5
        before = snapshot(network)
        slopes = network_slopes(network, inputs, targets)
        train_network_step(network, inputs, targets, learning_rate)
        after = snapshot(network)
        for layer_index in range(len(before)):
            for neuron_index in range(len(before[layer_index])):
                with self.subTest(layer=layer_index, neuron=neuron_index):
                    old_weights, old_bias = before[layer_index][neuron_index]
                    new_weights, new_bias = after[layer_index][neuron_index]
                    weight_slopes, bias_slope = slopes[layer_index][neuron_index]
                    for i in range(len(old_weights)):
                        self.assertAlmostEqual(new_weights[i], old_weights[i] - learning_rate * weight_slopes[i], places=6)
                    self.assertAlmostEqual(new_bias, old_bias - learning_rate * bias_slope, places=6)

    def test_single_layer_network_matches_train_step(self):
        # With one layer and one output, backpropagation must reduce to the
        # single-neuron update from step 2d.
        alone = Neuron([0.3, -0.2], 0.1)
        in_network = Neuron([0.3, -0.2], 0.1)
        train_step(alone, [1.0, 0.5], 0.9, 0.5)
        train_network_step(Network([Layer([in_network])]), [1.0, 0.5], [0.9], 0.5)
        for a, b in zip(alone.weights, in_network.weights):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(alone.bias, in_network.bias)

    def test_zero_learning_rate_changes_nothing(self):
        network = self.build_network()
        before = snapshot(network)
        train_network_step(network, [1.0, -0.5], [0.9, 0.1], 0.0)
        self.assertEqual(snapshot(network), before)


class TestTrainNetworkStepLearning(unittest.TestCase):
    """Verifies that repeated steps actually teach the network."""

    def test_loss_decreases_on_a_single_example(self):
        network = Network([
            Layer([Neuron([0.5, -0.3], 0.1), Neuron([-0.7, 0.8], -0.2)]),
            Layer([Neuron([0.6, -0.4], 0.3)]),
        ])
        first = train_network_step(network, [1.0, 0.0], [1.0], 0.5)
        for _ in range(500):
            last = train_network_step(network, [1.0, 0.0], [1.0], 0.5)
        self.assertLess(last, first)
        self.assertGreater(network.forward([1.0, 0.0])[0], 0.9)

    def test_learns_two_outputs_at_once(self):
        network = Network([
            Layer([Neuron([0.5, -0.3], 0.1), Neuron([-0.7, 0.8], -0.2)]),
            Layer([Neuron([0.6, -0.4], 0.3), Neuron([-0.5, 0.7], -0.1)]),
        ])
        for _ in range(3000):
            train_network_step(network, [1.0, 0.0], [0.9, 0.1], 1.0)
        first, second = network.forward([1.0, 0.0])
        self.assertAlmostEqual(first, 0.9, places=2)
        self.assertAlmostEqual(second, 0.1, places=2)


class TestTrainNetworkStepSafety(unittest.TestCase):
    """Verifies that invalid input fails loudly and changes nothing."""

    def build_network(self) -> Network:
        return Network([
            Layer([Neuron([0.5, -0.3], 0.1), Neuron([-0.7, 0.8], -0.2)]),
            Layer([Neuron([0.6, -0.4], 0.3)]),
        ])

    def test_wrong_number_of_inputs_raises_and_changes_nothing(self):
        network = self.build_network()
        before = snapshot(network)
        with self.assertRaises(ValueError):
            train_network_step(network, [1.0], [1.0], 0.5)
        self.assertEqual(snapshot(network), before)

    def test_wrong_number_of_targets_raises_and_changes_nothing(self):
        network = self.build_network()
        before = snapshot(network)
        with self.assertRaises(ValueError):
            train_network_step(network, [1.0, 0.0], [1.0, 0.0], 0.5)
        self.assertEqual(snapshot(network), before)

    def test_inputs_and_targets_are_not_modified(self):
        inputs, targets = [1.0, 0.0], [1.0]
        train_network_step(self.build_network(), inputs, targets, 0.5)
        self.assertEqual(inputs, [1.0, 0.0])
        self.assertEqual(targets, [1.0])

    def test_returns_a_float(self):
        self.assertIsInstance(train_network_step(self.build_network(), [1.0, 0.0], [1.0], 0.5), float)


if __name__ == "__main__":
    unittest.main()
