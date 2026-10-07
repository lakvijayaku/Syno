"""
Unit tests for syno.brain.training.

These tests are intentionally stricter than train_step's own safeguards. Beyond
checking that the loss goes down, they compare every update against the exact
worked example from step 2d and against a slope measured numerically, so a
step with the wrong size is caught, not just a step in the wrong direction.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.loss import mean_squared_error
from syno.brain.neuron import Neuron
from syno.brain.training import train_step


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


if __name__ == "__main__":
    unittest.main()
