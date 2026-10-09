"""
Experiment tests for experiments/signatures.py.

The fast tests probe eating_rate, eating_rpe and cue_rpe with hand-built
networks whose predictions are known exactly, and check that new_brain and
train are reproducible. The slow test trains SYNO with all 5 seeds (about 2
minutes) and reproduces the recorded E1, E2, E4 and E5 results. It is skipped by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import unittest
from unittest import mock

import experiments.signatures as signatures
from experiments.hunger import LIFE_STEPS
from syno.brain.layer import Layer
from syno.brain.network import Network
from syno.brain.neuron import Neuron
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward

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

    def test_stomach_levels_span_empty_to_full(self):
        self.assertEqual(signatures.STOMACH_LEVELS[0], 0.0)
        self.assertEqual(signatures.STOMACH_LEVELS[-1], 1.0)
        self.assertEqual(len(signatures.STOMACH_LEVELS), 6)

    def test_satiation_is_tested_half_hungry(self):
        self.assertEqual(signatures.SATIATION_ENERGY, 0.5)

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

    def test_stomach_defaults_to_empty(self):
        # Eat wins only if the stomach input is 0.
        network = probe_network({STOMACH: -1.0}, 0.001)
        self.assertEqual(signatures.eating_rate(network, 0.5), 1.0)
        self.assertEqual(signatures.eating_rate(network, 0.5, 0.0), 1.0)

    def test_eats_only_when_stomach_is_nearly_empty(self):
        # Eat's value is 0.5 - stomach, so eating is preferred below 0.5.
        network = probe_network({STOMACH: -1.0}, 0.5)
        for stomach in (0.0, 0.2, 0.4):
            self.assertEqual(signatures.eating_rate(network, 0.5, stomach), 1.0)
        for stomach in (0.6, 0.8, 1.0):
            self.assertEqual(signatures.eating_rate(network, 0.5, stomach), 0.0)

    def test_stomach_does_not_change_energy(self):
        # Eat's value is deficit - 0.45, so the stomach must not affect it.
        network = probe_network({DEFICIT: 1.0}, -0.45)
        self.assertEqual(signatures.eating_rate(network, 0.2, 1.0), 1.0)
        self.assertEqual(signatures.eating_rate(network, 0.8, 0.0), 0.0)

    def test_does_not_explore(self):
        network = probe_network({}, 0.001)
        for _ in range(20):
            self.assertEqual(signatures.eating_rate(network, 0.5), 1.0)


def step_reward(eats: bool, moved: bool = False) -> float:
    """The scaled reward for one step at energy 0.5, with or without a meal."""
    body = HomeostaticCore(0.5, 0.0)
    before = body.drive()
    if eats:
        body.eat(signatures.FOOD_AMOUNT)
    body.tick(moved)
    return signatures.REWARD_SCALE * homeostatic_reward(before, body.drive())


class TestEatingRpe(unittest.TestCase):
    """Verifies eating_rpe against networks with known predictions."""

    def test_zero_network_rpe_is_the_meal_reward(self):
        # With every prediction 0, the RPE is just the reward.
        network = probe_network({}, 0.0)
        self.assertAlmostEqual(signatures.eating_rpe(network, True), step_reward(True))

    def test_zero_network_rpe_without_food(self):
        network = probe_network({}, 0.0)
        self.assertAlmostEqual(signatures.eating_rpe(network, False), step_reward(False))

    def test_omission_is_worse_than_delivery(self):
        network = probe_network({}, 0.0)
        self.assertLess(signatures.eating_rpe(network, False), signatures.eating_rpe(network, True))

    def test_constant_prediction(self):
        # Eat always predicts 2.0, so the RPE is reward + 0.9 * 2.0 - 2.0.
        network = probe_network({}, 2.0)
        expected = step_reward(False) + signatures.DISCOUNT * 2.0 - 2.0
        self.assertAlmostEqual(signatures.eating_rpe(network, False), expected)

    def test_expectation_is_read_while_food_is_visible(self):
        # Eat predicts 1.0 only while food is under SYNO. If the prediction
        # were read after the food vanished, it would be 0 and the dip lost.
        network = probe_network({CENTER: 1.0}, 0.0)
        self.assertAlmostEqual(signatures.eating_rpe(network, False), step_reward(False) - 1.0)

    def test_new_food_appears_only_after_a_meal(self):
        # As in live, food respawns after every successful meal, and never
        # when the meal is missing.
        network = probe_network({}, 0.0)
        with mock.patch.object(signatures, "spawn_food") as spawn:
            signatures.eating_rpe(network, True)
            self.assertEqual(spawn.call_count, signatures.GRID_SIZE ** 2)
            spawn.reset_mock()
            signatures.eating_rpe(network, False)
            self.assertEqual(spawn.call_count, 0)

    def test_does_not_change_the_network(self):
        network = probe_network({CENTER: 1.0}, 0.5)
        state = [0.0] * INPUTS
        before = network.forward(state)
        signatures.eating_rpe(network, True)
        signatures.eating_rpe(network, False)
        self.assertEqual(network.forward(state), before)


class TestCueRpe(unittest.TestCase):
    """Verifies cue_rpe against networks with known predictions."""

    def test_constant_prediction(self):
        # Eat always predicts 2.0 and is chosen, but there is no food to eat,
        # so the RPE is the reward for an empty step + 0.9 * 2.0 - 2.0.
        network = probe_network({}, 2.0)
        expected = step_reward(False) + signatures.DISCOUNT * 2.0 - 2.0
        self.assertAlmostEqual(signatures.cue_rpe(network), expected)

    def test_food_appears_after_syno_acts(self):
        # New food is placed under SYNO. Eat predicts 0.1 on an empty square
        # and 1.1 once food is under SYNO, so seeing the food is a surprise.
        def food_under_syno(habitat):
            habitat.food.append(habitat.agent)

        network = probe_network({CENTER: 1.0}, 0.1)
        with mock.patch.object(signatures, "spawn_food", side_effect=food_under_syno) as spawn:
            rpe = signatures.cue_rpe(network)
        self.assertEqual(spawn.call_count, signatures.GRID_SIZE ** 2)
        expected = step_reward(False) + signatures.DISCOUNT * 1.1 - 0.1
        self.assertAlmostEqual(rpe, expected)

    def test_moving_costs_extra_energy(self):
        # Up always predicts 2.0 and is chosen. SYNO moves from the 6 squares
        # below the top row, and walks into the wall from the other 3.
        up = signatures.ACTION_NAMES.index("up")
        neurons = [Neuron([0.0] * INPUTS, 2.0 if action == up else 0.0, "linear")
                   for action in range(len(signatures.ACTION_NAMES))]
        network = Network([Layer(neurons)])
        reward = (6 * step_reward(False, True) + 3 * step_reward(False)) / 9
        expected = reward + signatures.DISCOUNT * 2.0 - 2.0
        self.assertAlmostEqual(signatures.cue_rpe(network), expected)

    def test_does_not_change_the_network(self):
        network = probe_network({CENTER: 1.0}, 0.5)
        state = [0.0] * INPUTS
        before = network.forward(state)
        signatures.cue_rpe(network)
        self.assertEqual(network.forward(state), before)


class TestTrain(unittest.TestCase):
    """Verifies that new_brain and train give a reproducible network."""

    STATE = [0.0] * INPUTS

    def trained(self, seed: int, *lives: int) -> list[float]:
        """Trains a new brain in one or more parts and returns its output."""
        network, memory = signatures.new_brain(seed)
        for part in lives:
            signatures.train(network, memory, part)
        return network.forward(self.STATE)

    def test_network_shape(self):
        network, memory = signatures.new_brain(0)
        self.assertEqual(len(network.layers), 2)
        self.assertEqual(len(network.layers[0].neurons), signatures.HIDDEN_NEURONS)
        self.assertEqual(len(network.layers[0].neurons[0].weights), INPUTS)
        self.assertEqual(len(network.layers[1].neurons), len(signatures.ACTION_NAMES))
        self.assertEqual(network.layers[1].neurons[0].activation, "linear")
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.capacity, signatures.MEMORY_CAPACITY)

    def test_same_seed_same_network(self):
        self.assertEqual(self.trained(4, 3), self.trained(4, 3))

    def test_different_seeds_differ(self):
        self.assertNotEqual(self.trained(0, 3), self.trained(1, 3))

    def test_training_changes_the_network(self):
        self.assertNotEqual(self.trained(0, 0), self.trained(0, 3))

    def test_training_in_parts_matches_one_call(self):
        self.assertEqual(self.trained(2, 1, 2), self.trained(2, 3))

    def test_training_fills_memory(self):
        network, memory = signatures.new_brain(0)
        signatures.train(network, memory, 1)
        self.assertEqual(len(memory), LIFE_STEPS)

    def test_checkpoints_end_at_full_training(self):
        self.assertEqual(signatures.CHECKPOINTS[0], 0)
        self.assertEqual(signatures.CHECKPOINTS[-1], signatures.TRAINING_LIVES)
        self.assertEqual(signatures.CHECKPOINTS, sorted(set(signatures.CHECKPOINTS)))


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestSignaturesRun(unittest.TestCase):
    """Reproduces the recorded E1, E2, E4 and E5 results."""

    def test_results(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            signatures.main()
        expected = ["E1: Satiation (5 seeds, energy 0.5)"]
        for stomach, rate in zip(signatures.STOMACH_LEVELS, ("1.00", "1.00", "1.00", "0.96", "0.91", "0.71")):
            expected.append(f"Stomach {stomach:.1f} | Eats: {rate}")
        expected += ["", "E2: State-dependent eating (5 seeds)"]
        for level in signatures.ENERGY_LEVELS:
            rate = "0.73" if level == 1.0 else "1.00"
            expected.append(f"Energy {level:.1f} | Eats: {rate}")
        expected += [
            "",
            "E4: RPE transfer (5 seeds)",
            "Lives    0 | Food RPE: 2.443 | Cue RPE: -0.193",
            "Lives   50 | Food RPE: 0.185 | Cue RPE: -0.190",
            "Lives  200 | Food RPE: 0.173 | Cue RPE: -0.208",
            "Lives 1500 | Food RPE: -0.083 | Cue RPE: -0.063",
            "",
            "E5: Reward omission (5 seeds)",
            "Food delivered | RPE: -0.083",
            "Food omitted   | RPE: -0.874",
        ]
        self.assertEqual(output.getvalue().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
