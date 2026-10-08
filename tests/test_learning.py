"""
Unit tests for syno.brain.learning.

These tests are intentionally stricter than the function's own safeguards.
They verify the step 4c worked example, that only the taken action is
updated, how terminal and non-terminal steps differ, and that repeated
learning reproduces two key results of temporal-difference learning: the RPE
for a predicted reward shrinks toward zero, and value flows backward from a
reward to the steps that lead to it.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.brain.layer import Layer
from syno.brain.learning import learn
from syno.brain.network import Network
from syno.brain.neuron import Neuron


def two_action_network() -> tuple[Network, Neuron, Neuron]:
    """Builds a one-layer network with 1 input and 2 actions, all values 0.5."""
    first, second = Neuron([0.0], 0.0), Neuron([0.0], 0.0)
    return Network([Layer([first, second])]), first, second


class TestLearnWorkedExample(unittest.TestCase):
    """Verifies the exact worked example from step 4c."""

    def setUp(self):
        self.network, self.first, self.second = two_action_network()
        self.rpe = learn(self.network, [1.0], 0, 1.0, [1.0], True, 0.9, 1.0)

    def test_returns_the_rpe(self):
        self.assertAlmostEqual(self.rpe, 0.5)

    def test_taken_action_is_updated(self):
        self.assertAlmostEqual(self.first.weights[0], 0.125)
        self.assertAlmostEqual(self.first.bias, 0.125)

    def test_other_action_is_unchanged(self):
        self.assertEqual(self.second.weights, [0.0])
        self.assertEqual(self.second.bias, 0.0)

    def test_value_of_taken_action_rises(self):
        values = self.network.forward([1.0])
        self.assertAlmostEqual(values[0], 0.5621765008857981)
        self.assertAlmostEqual(values[1], 0.5)


class TestLearnTargets(unittest.TestCase):
    """Verifies how each step's target is built."""

    def test_done_ignores_the_next_state(self):
        # When the episode ends, the next state's values must not matter.
        results = []
        for next_state in ([1.0], [-1.0], [0.0]):
            network, _, _ = two_action_network()
            network.layers[0].neurons[1].bias = 3.0
            results.append(learn(network, [1.0], 0, 0.0, next_state, True, 0.9, 1.0))
        self.assertAlmostEqual(results[0], -0.5)
        self.assertAlmostEqual(results[1], results[0])
        self.assertAlmostEqual(results[2], results[0])

    def test_not_done_uses_the_best_next_value(self):
        # Action 1's value is raised, so the best next value is action 1's.
        network, _, second = two_action_network()
        second.bias = 2.0
        best_next = max(network.forward([1.0]))
        rpe = learn(network, [1.0], 0, 0.0, [1.0], False, 0.9, 0.0)
        self.assertAlmostEqual(rpe, 0.9 * best_next - 0.5)

    def test_zero_learning_rate_changes_nothing(self):
        network, first, second = two_action_network()
        learn(network, [1.0], 0, 1.0, [1.0], True, 0.9, 0.0)
        self.assertEqual((first.weights, first.bias), ([0.0], 0.0))
        self.assertEqual((second.weights, second.bias), ([0.0], 0.0))

    def test_better_than_expected_raises_value_and_worse_lowers_it(self):
        network, _, _ = two_action_network()
        learn(network, [1.0], 0, 1.0, [1.0], True, 0.9, 1.0)
        self.assertGreater(network.forward([1.0])[0], 0.5)
        network, _, _ = two_action_network()
        learn(network, [1.0], 0, 0.0, [1.0], True, 0.9, 1.0)
        self.assertLess(network.forward([1.0])[0], 0.5)

    def test_works_with_a_hidden_layer(self):
        network = Network([
            Layer([Neuron([0.5, -0.3], 0.1), Neuron([-0.7, 0.8], -0.2)]),
            Layer([Neuron([0.6, -0.4], 0.3), Neuron([-0.5, 0.7], -0.1), Neuron([0.2, 0.2], 0.0)]),
        ])
        before = network.forward([1.0, 0.0])
        learn(network, [1.0, 0.0], 2, 1.0, [0.0, 1.0], True, 0.9, 0.5)
        after = network.forward([1.0, 0.0])
        self.assertGreater(after[2], before[2])


class TestLearnOverTime(unittest.TestCase):
    """Verifies the behavior of repeated temporal-difference learning."""

    def test_predicted_reward_stops_causing_dopamine(self):
        # The same reward, repeated, becomes predicted, so the RPE shrinks
        # toward zero, just as dopamine neurons stop responding to a reward
        # they have learned to expect.
        network, _, _ = two_action_network()
        first = learn(network, [1.0], 0, 1.0, [1.0], True, 0.9, 1.0)
        for _ in range(2000):
            last = learn(network, [1.0], 0, 1.0, [1.0], True, 0.9, 1.0)
        self.assertGreater(first, 0.4)
        self.assertLess(abs(last), 0.05)

    def test_value_flows_backward_to_earlier_steps(self):
        # A two-step chain: from state A, the action leads to state B with no
        # reward; from state B, the action ends the episode with reward 1.
        # State A never receives reward directly, yet it must learn to expect
        # about discount * 1.
        state_a, state_b = [1.0, 0.0], [0.0, 1.0]
        network = Network([Layer([Neuron([0.0, 0.0], 0.0)])])
        for _ in range(5000):
            learn(network, state_a, 0, 0.0, state_b, False, 0.9, 1.0)
            learn(network, state_b, 0, 1.0, state_b, True, 0.9, 1.0)
        self.assertGreater(network.forward(state_b)[0], 0.9)
        self.assertAlmostEqual(network.forward(state_a)[0], 0.9 * network.forward(state_b)[0], delta=0.03)


class TestLearnSafety(unittest.TestCase):
    """Verifies that invalid input fails loudly."""

    def test_action_out_of_range_raises(self):
        # A negative action is valid Python indexing, so without a guard it
        # would silently train the wrong action.
        for action in (-1, 2):
            with self.subTest(action=action):
                network, _, _ = two_action_network()
                with self.assertRaises((ValueError, IndexError)):
                    learn(network, [1.0], action, 1.0, [1.0], True, 0.9, 1.0)

    def test_failed_step_changes_nothing(self):
        network, first, second = two_action_network()
        with self.assertRaises(ValueError):
            learn(network, [1.0, 2.0], 0, 1.0, [1.0], True, 0.9, 1.0)
        self.assertEqual((first.weights, first.bias), ([0.0], 0.0))
        self.assertEqual((second.weights, second.bias), ([0.0], 0.0))

    def test_inputs_are_not_modified(self):
        state, next_state = [1.0], [1.0]
        network, _, _ = two_action_network()
        learn(network, state, 0, 1.0, next_state, False, 0.9, 1.0)
        self.assertEqual(state, [1.0])
        self.assertEqual(next_state, [1.0])

    def test_returns_a_float(self):
        network, _, _ = two_action_network()
        self.assertIsInstance(learn(network, [1.0], 0, 1.0, [1.0], True, 0.9, 1.0), float)


if __name__ == "__main__":
    unittest.main()
