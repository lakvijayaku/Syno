"""
Unit tests for syno.brain.neuron.

These tests are intentionally stricter than the Neuron's own safeguards. They
verify not only that each known bug stays fixed, but also the mathematical
properties of the sigmoid function, so that any future change that breaks the
neuron's behavior is caught immediately.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import math
import unittest

from syno.brain.neuron import Neuron


def reference_sigmoid(z: float) -> float:
    """
    Computes the sigmoid of z independently of the Neuron class.

    The tests compare the Neuron against this reference, so a bug in the
    Neuron's formula cannot also hide in the expected value. Only used for
    moderate values of z, where the standard formula cannot overflow.
    """
    return 1 / (1 + math.exp(-z))


class TestNeuronOutput(unittest.TestCase):
    """Verifies that the neuron computes the correct output values."""

    def test_zero_inputs_and_zero_bias_return_half(self):
        # sigmoid(0) is exactly 0.5, the midpoint of the output range.
        neuron = Neuron([1.0, 2.0, 3.0], 0.0)
        self.assertEqual(neuron.forward([0.0, 0.0, 0.0]), 0.5)

    def test_known_weighted_sum(self):
        # 0.5 * 2.0 + (-1.0) * 3.0 + 0.0 = -2.0, so the output is sigmoid(-2).
        neuron = Neuron([0.5, -1.0], 0.0)
        self.assertAlmostEqual(neuron.forward([2.0, 3.0]), reference_sigmoid(-2.0))

    def test_bias_shifts_output_when_inputs_are_zero(self):
        # With all inputs at 0, the weighted sum equals the bias alone. This
        # proves the bias is actually added to the sum.
        for bias in (-3.0, -0.5, 0.5, 3.0):
            with self.subTest(bias=bias):
                neuron = Neuron([1.0, 1.0], bias)
                self.assertAlmostEqual(neuron.forward([0.0, 0.0]), reference_sigmoid(bias))

    def test_empty_inputs_return_sigmoid_of_bias(self):
        # A neuron with no inputs is valid: its output depends only on the bias.
        neuron = Neuron([], 1.5)
        self.assertAlmostEqual(neuron.forward([]), reference_sigmoid(1.5))

    def test_output_is_a_float(self):
        neuron = Neuron([1.0], 0.0)
        self.assertIsInstance(neuron.forward([1.0]), float)


class TestSigmoidProperties(unittest.TestCase):
    """Verifies mathematical properties that every correct sigmoid must have."""

    def test_output_stays_strictly_between_zero_and_one_for_moderate_inputs(self):
        # For moderate inputs, the output must never touch 0 or 1.
        neuron = Neuron([1.0], 0.0)
        for z in range(-30, 31):
            with self.subTest(z=z):
                output = neuron.forward([float(z)])
                self.assertGreater(output, 0.0)
                self.assertLess(output, 1.0)

    def test_symmetry(self):
        # sigmoid(z) + sigmoid(-z) == 1 for every z. This checks that the
        # positive and negative branches agree with each other.
        neuron = Neuron([1.0], 0.0)
        for z in (0.1, 1.0, 2.5, 10.0, 50.0):
            with self.subTest(z=z):
                total = neuron.forward([z]) + neuron.forward([-z])
                self.assertAlmostEqual(total, 1.0)

    def test_output_increases_as_weighted_sum_increases(self):
        # Sigmoid is monotonically increasing: a larger input must never
        # produce a smaller output.
        neuron = Neuron([1.0], 0.0)
        outputs = [neuron.forward([z / 10]) for z in range(-100, 101)]
        for previous, current in zip(outputs, outputs[1:]):
            self.assertLessEqual(previous, current)

    def test_branches_meet_smoothly_at_zero(self):
        # The two formulas switch at 0. Values just below and just above 0
        # must both be very close to 0.5, with no jump between branches.
        neuron = Neuron([1.0], 0.0)
        self.assertAlmostEqual(neuron.forward([-1e-9]), 0.5)
        self.assertAlmostEqual(neuron.forward([1e-9]), 0.5)

    def test_matches_reference_across_range(self):
        neuron = Neuron([1.0], 0.0)
        for z in (-20.0, -5.0, -1.0, -0.01, 0.01, 1.0, 5.0, 20.0):
            with self.subTest(z=z):
                self.assertAlmostEqual(neuron.forward([z]), reference_sigmoid(z))


class TestNumericalStability(unittest.TestCase):
    """Verifies that extreme inputs never crash the neuron (regression tests)."""

    def test_large_negative_input_does_not_overflow(self):
        neuron = Neuron([1.0], 0.0)
        self.assertAlmostEqual(neuron.forward([-1000.0]), 0.0)

    def test_large_positive_input_does_not_overflow(self):
        neuron = Neuron([1.0], 0.0)
        self.assertAlmostEqual(neuron.forward([1000.0]), 1.0)

    def test_inputs_around_the_overflow_threshold(self):
        # math.exp() overflows for inputs above roughly 709. Both sides of
        # that threshold, in both directions, must be handled safely.
        neuron = Neuron([1.0], 0.0)
        for z in (-710.0, -709.0, 709.0, 710.0):
            with self.subTest(z=z):
                output = neuron.forward([z])
                self.assertGreaterEqual(output, 0.0)
                self.assertLessEqual(output, 1.0)

    def test_large_weights_and_bias_do_not_overflow(self):
        # Overflow can come from the weights or the bias, not only the inputs.
        neuron = Neuron([1e6, 1e6], -1e7)
        self.assertAlmostEqual(neuron.forward([1.0, 1.0]), 0.0)


class TestInputValidation(unittest.TestCase):
    """Verifies that invalid input fails loudly instead of silently."""

    def test_too_few_inputs_raise_value_error(self):
        neuron = Neuron([0.5, 0.5], 0.0)
        with self.assertRaises(ValueError):
            neuron.forward([1.0])

    def test_too_many_inputs_raise_value_error(self):
        neuron = Neuron([0.5], 0.0)
        with self.assertRaises(ValueError):
            neuron.forward([1.0, 2.0])

    def test_error_message_reports_both_lengths(self):
        neuron = Neuron([0.5, 0.5], 0.0)
        with self.assertRaises(ValueError) as context:
            neuron.forward([1.0, 2.0, 3.0])
        self.assertIn("Expected 2 inputs, got 3", str(context.exception))

    def test_error_preserves_original_cause(self):
        # "raise ... from e" must keep the original zip error as the cause.
        neuron = Neuron([0.5, 0.5], 0.0)
        with self.assertRaises(ValueError) as context:
            neuron.forward([1.0])
        self.assertIsInstance(context.exception.__cause__, ValueError)


class TestStateIsolation(unittest.TestCase):
    """Verifies that the neuron does not share or alter outside data."""

    def test_changing_original_weights_does_not_affect_neuron(self):
        weights = [1.0, 2.0]
        neuron = Neuron(weights, 0.0)
        weights[0] = 99.0
        self.assertEqual(neuron.weights, [1.0, 2.0])

    def test_neuron_stores_a_different_list_object(self):
        weights = [1.0, 2.0]
        neuron = Neuron(weights, 0.0)
        self.assertIsNot(neuron.weights, weights)

    def test_forward_does_not_modify_inputs_or_weights(self):
        neuron = Neuron([0.5, -1.0], 0.25)
        inputs = [2.0, 3.0]
        neuron.forward(inputs)
        self.assertEqual(inputs, [2.0, 3.0])
        self.assertEqual(neuron.weights, [0.5, -1.0])
        self.assertEqual(neuron.bias, 0.25)

    def test_forward_is_repeatable(self):
        # Calling forward twice with the same input must give the same result.
        neuron = Neuron([0.3, 0.7], -0.1)
        self.assertEqual(neuron.forward([1.0, 2.0]), neuron.forward([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
