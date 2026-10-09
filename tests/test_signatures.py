"""
Experiment tests for experiments/signatures.py.

The fast tests probe eating_rate, approach_rate, eating_rpe and cue_rpe with
hand-built networks whose predictions are known exactly, and check that
new_brain and train are reproducible. The slow test trains SYNO with all 5
seeds in the scarce world, in parallel (about 90 seconds) and reproduces the recorded E1 to
E5 results. It is skipped by default and runs only when SYNO_SLOW_TESTS is
set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import unittest
from unittest import mock

import experiments.scarcity as scarcity
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

    def test_uses_ten_seeds(self):
        self.assertEqual(signatures.SEEDS, 10)

    def test_clear_effect_thresholds(self):
        self.assertEqual(signatures.CLEAR_EFFECT, 0.1)
        self.assertEqual(signatures.CLEAR_DIP, 0.5)

    def test_stomach_levels_span_empty_to_full(self):
        self.assertEqual(signatures.STOMACH_LEVELS[0], 0.0)
        self.assertEqual(signatures.STOMACH_LEVELS[-1], 1.0)
        self.assertEqual(len(signatures.STOMACH_LEVELS), 6)

    def test_satiation_is_tested_half_hungry(self):
        self.assertEqual(signatures.SATIATION_ENERGY, 0.5)

    def test_trains_in_the_scarce_world(self):
        self.assertEqual(signatures.GRID_SIZE, 5)
        self.assertIs(signatures.live, scarcity.live)
        self.assertEqual(signatures.TRAINING_LIVES, scarcity.LIVES)


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
        # 5 of the 25 squares eat.
        above = CENTER - (2 * signatures.SENSE_RADIUS + 1)
        network = probe_network({above: -1.0}, -0.5)
        self.assertAlmostEqual(signatures.eating_rate(network, 0.5), 5 / 25)

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


def move_network(action_name: str) -> Network:
    """Builds a one-layer linear network that always prefers one action."""
    preferred = signatures.ACTION_NAMES.index(action_name)
    neurons = [Neuron([0.0] * INPUTS, 1.0 if action == preferred else 0.0, "linear")
               for action in range(len(signatures.ACTION_NAMES))]
    return Network([Layer(neurons)])


def up_when(up_weights: dict[int, float], up_bias: float) -> Network:
    """
    Builds a one-layer linear network that prefers up when its weighted
    inputs plus up_bias are above 0.0, and otherwise prefers eat.
    """
    up = signatures.ACTION_NAMES.index("up")
    neurons = []
    for action in range(len(signatures.ACTION_NAMES)):
        weights = [0.0] * INPUTS
        bias = -1.0
        if action == up:
            for index, weight in up_weights.items():
                weights[index] = weight
            bias = up_bias
        elif action == EAT:
            bias = 0.0
        neurons.append(Neuron(weights, bias, "linear"))
    return Network([Layer(neurons)])


class TestApproachRate(unittest.TestCase):
    """Verifies approach_rate against networks with known preferences."""

    def test_far_food_is_always_in_view(self):
        # Of the 396 pairs at least 3 steps apart, only 132 have the food
        # within SYNO's view. Food out of view is never tested.
        network = move_network("up")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, True), 66 / 132)
        network = move_network("left")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, True), 66 / 132)

    def test_never_approaches_while_eating(self):
        network = move_network("eat")
        for far in (False, True):
            self.assertEqual(signatures.approach_rate(network, 0.5, far), 0.0)

    def test_always_up_near(self):
        # Of the 80 adjacent pairs, food is directly above SYNO in 20.
        network = move_network("up")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, False), 20 / 80)

    def test_always_up_far(self):
        # Of the 132 visible pairs at least 3 steps apart, food is above SYNO in 66.
        network = move_network("up")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, True), 66 / 132)

    def test_always_right_far(self):
        network = move_network("right")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, True), 66 / 132)

    def test_moving_away_never_counts(self):
        # Food is 1 step away, and SYNO always goes down. It approaches only
        # food directly below it, in 20 of the 80 pairs, never the other 60.
        network = move_network("down")
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, False), 20 / 80)

    def test_sees_the_food(self):
        # Up wins only when food is directly above SYNO in its view.
        above = CENTER - (2 * signatures.SENSE_RADIUS + 1)
        network = up_when({above: 1.0}, -0.5)
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, False), 20 / 80)

    def test_sideways_moves_go_the_right_way(self):
        # Right wins only when food is directly right of SYNO in its view, so
        # it approaches in all 20 such pairs. Moving the wrong way would never
        # approach.
        right = signatures.ACTION_NAMES.index("right")
        neurons = []
        for action in range(len(signatures.ACTION_NAMES)):
            weights = [0.0] * INPUTS
            bias = 0.0 if action == EAT else -1.0
            if action == right:
                weights[CENTER + 1] = 1.0
                bias = -0.5
            neurons.append(Neuron(weights, bias, "linear"))
        network = Network([Layer(neurons)])
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, False), 20 / 80)

    def test_uses_the_given_energy(self):
        # Up wins only when the deficit is above 0.45.
        network = up_when({DEFICIT: 1.0}, -0.45)
        self.assertAlmostEqual(signatures.approach_rate(network, 0.2, True), 66 / 132)
        self.assertEqual(signatures.approach_rate(network, 0.8, True), 0.0)

    def test_stomach_is_empty(self):
        # Up wins only when the stomach input is 0.
        network = up_when({STOMACH: -1.0}, 0.001)
        self.assertAlmostEqual(signatures.approach_rate(network, 0.5, True), 66 / 132)


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

    def test_no_food_appears_after_a_meal(self):
        # In the scarce world, eaten food regrows only after REGROW_DELAY
        # steps, so the probe must never add food, and has no way to.
        self.assertFalse(hasattr(signatures, "spawn_food"))

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

    def test_food_appears_in_every_visible_square(self):
        # Food appears in turn on every square SYNO can see, except its own:
        # 336 squares across the 25 starting positions, one RPE each.
        network = probe_network({}, 2.0)
        with mock.patch.object(signatures, "reward_prediction_error",
                               wraps=signatures.reward_prediction_error) as rpe:
            signatures.cue_rpe(network)
        self.assertEqual(rpe.call_count, 336)

    def test_view_is_taken_after_syno_moves(self):
        # Up predicts 2.0, plus 1.0 when food is 2 squares above SYNO. SYNO
        # moves up, so food appears around its new square, and only food 2
        # squares above the new square raises the next prediction.
        width = 2 * signatures.SENSE_RADIUS + 1
        two_above = CENTER - 2 * width
        up = signatures.ACTION_NAMES.index("up")
        neurons = []
        for action in range(len(signatures.ACTION_NAMES)):
            weights = [0.0] * INPUTS
            if action == up:
                weights[two_above] = 1.0
            neurons.append(Neuron(weights, 2.0 if action == up else 0.0, "linear"))
        network = Network([Layer(neurons)])
        size = signatures.GRID_SIZE
        radius = signatures.SENSE_RADIUS
        total = 0.0
        count = 0
        for y in range(size):
            for x in range(size):
                # Before moving, a wall 2 squares above lowers up to 1.0.
                expected = 2.0 - (1.0 if y < 2 else 0.0)
                moved = y > 0
                after = (x, y - 1) if moved else (x, y)
                reward = step_reward(False, moved)
                for fy in range(size):
                    for fx in range(size):
                        if abs(fx - after[0]) <= radius and abs(fy - after[1]) <= radius \
                                and (fx, fy) != after:
                            wall = after[1] < 2
                            food = (fx, fy) == (after[0], after[1] - 2)
                            next_value = 2.0 - (1.0 if wall else 0.0) + (1.0 if food else 0.0)
                            total += reward + signatures.DISCOUNT * next_value - expected
                            count += 1
        self.assertAlmostEqual(signatures.cue_rpe(network), total / count)

    def test_food_never_appears_under_syno(self):
        # Eat predicts 0.1, plus 1.0 when food is under SYNO. Food under SYNO
        # would be eaten, not seen, so it is never one of the tested squares.
        network = probe_network({CENTER: 1.0}, 0.1)
        expected = step_reward(False) + signatures.DISCOUNT * 0.1 - 0.1
        self.assertAlmostEqual(signatures.cue_rpe(network), expected)

    def test_moving_costs_extra_energy(self):
        # Up always predicts 2.0 and is chosen. SYNO moves from the 20 squares
        # below the top row, and walks into the wall from the other 5. Each
        # position counts once for every square in view after the step.
        up = signatures.ACTION_NAMES.index("up")
        neurons = [Neuron([0.0] * INPUTS, 2.0 if action == up else 0.0, "linear")
                   for action in range(len(signatures.ACTION_NAMES))]
        network = Network([Layer(neurons)])
        size = signatures.GRID_SIZE
        radius = signatures.SENSE_RADIUS
        total = 0.0
        count = 0
        for y in range(size):
            for x in range(size):
                moved = y > 0
                after = (x, y - 1) if moved else (x, y)
                in_view = sum(
                    1
                    for fy in range(size)
                    for fx in range(size)
                    if abs(fx - after[0]) <= radius and abs(fy - after[1]) <= radius
                    and (fx, fy) != after
                )
                total += in_view * step_reward(False, moved)
                count += in_view
        expected = total / count + signatures.DISCOUNT * 2.0 - 2.0
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


class TestSpread(unittest.TestCase):
    """Verifies how results from several seeds are described."""

    def test_average_lowest_and_highest(self):
        self.assertEqual(signatures.spread([0.5, 1.0, 0.0], 2), "0.50 [0.00, 1.00]")

    def test_digits(self):
        self.assertEqual(signatures.spread([-0.0274, 0.1], 3), "0.036 [-0.027, 0.100]")

    def test_single_value(self):
        self.assertEqual(signatures.spread([0.25], 2), "0.25 [0.25, 0.25]")

    def test_order_does_not_matter_for_range(self):
        self.assertEqual(signatures.spread([1.0, 0.0], 1), signatures.spread([0.0, 1.0], 1))


class TestMeasure(unittest.TestCase):
    """Verifies that measure trains one SYNO and measures every signature."""

    def setUp(self):
        self.saved = signatures.CHECKPOINTS
        signatures.CHECKPOINTS = [0, 1, 2]

    def tearDown(self):
        signatures.CHECKPOINTS = self.saved

    def test_result_shapes(self):
        results = signatures.measure(0)
        self.assertEqual(len(results["food"]), 3)
        self.assertEqual(len(results["cue"]), 3)
        self.assertEqual(len(results["e1"]), len(signatures.STOMACH_LEVELS))
        self.assertEqual(len(results["e2"]), len(signatures.ENERGY_LEVELS))
        self.assertEqual(len(results["near"]), len(signatures.APPROACH_ENERGIES))
        self.assertEqual(len(results["far"]), len(signatures.APPROACH_ENERGIES))
        self.assertEqual(len(results["e5"]), 2)

    def test_matches_the_probes_on_the_same_network(self):
        results = signatures.measure(3)
        network, memory = signatures.new_brain(3)
        signatures.train(network, memory, 2)
        self.assertEqual(results["e2"], [signatures.eating_rate(network, e) for e in signatures.ENERGY_LEVELS])
        self.assertEqual(results["e1"], [signatures.eating_rate(network, signatures.SATIATION_ENERGY, s)
                                         for s in signatures.STOMACH_LEVELS])
        self.assertEqual(results["near"], [signatures.approach_rate(network, e, False)
                                           for e in signatures.APPROACH_ENERGIES])
        self.assertEqual(results["far"], [signatures.approach_rate(network, e, True)
                                          for e in signatures.APPROACH_ENERGIES])
        self.assertEqual(results["e5"], [signatures.eating_rpe(network, True), signatures.eating_rpe(network, False)])
        self.assertEqual(results["food"][-1], results["e5"][0])
        self.assertEqual(results["cue"][-1], signatures.cue_rpe(network))

    def test_first_checkpoint_is_untrained(self):
        results = signatures.measure(3)
        network, _ = signatures.new_brain(3)
        self.assertEqual(results["food"][0], signatures.eating_rpe(network, True))

    def test_seeds_differ(self):
        self.assertNotEqual(signatures.measure(0)["e5"], signatures.measure(1)["e5"])


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestSignaturesRun(unittest.TestCase):
    """Reproduces the recorded E1 to E5 results."""

    def test_results(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            signatures.main()
        expected = """E1: Satiation (10 seeds, energy 0.5)
Stomach 0.0 | Eats: 0.98 [0.88, 1.00]
Stomach 0.2 | Eats: 0.99 [0.96, 1.00]
Stomach 0.4 | Eats: 0.94 [0.76, 1.00]
Stomach 0.6 | Eats: 0.86 [0.20, 1.00]
Stomach 0.8 | Eats: 0.75 [0.00, 1.00]
Stomach 1.0 | Eats: 0.62 [0.00, 1.00]
Seeds: 7/10

E2: State-dependent eating (10 seeds)
Energy 0.0 | Eats: 0.99 [0.92, 1.00]
Energy 0.1 | Eats: 0.99 [0.96, 1.00]
Energy 0.2 | Eats: 0.99 [0.96, 1.00]
Energy 0.3 | Eats: 1.00 [0.96, 1.00]
Energy 0.4 | Eats: 1.00 [0.96, 1.00]
Energy 0.5 | Eats: 0.98 [0.88, 1.00]
Energy 0.6 | Eats: 0.98 [0.88, 1.00]
Energy 0.7 | Eats: 0.97 [0.84, 1.00]
Energy 0.8 | Eats: 0.95 [0.76, 1.00]
Energy 0.9 | Eats: 0.92 [0.64, 1.00]
Energy 1.0 | Eats: 0.89 [0.44, 1.00]
Seeds: 2/10

E3: Partial-fullness eating (10 seeds)
Energy 0.2 | Near: 0.95 [0.89, 1.00] | Far: 0.92 [0.82, 0.98]
Energy 0.5 | Near: 0.96 [0.90, 1.00] | Far: 0.92 [0.87, 0.98]
Energy 0.8 | Near: 0.93 [0.75, 1.00] | Far: 0.91 [0.80, 0.97]
Energy 1.0 | Near: 0.90 [0.75, 0.99] | Far: 0.88 [0.78, 0.97]
Seeds: 1/10

E4: RPE transfer (10 seeds)
Lives    0 | Food RPE: 2.335 [0.740, 3.153] | Cue RPE: -0.278 [-0.625, 0.049]
Lives   50 | Food RPE: 0.350 [0.143, 0.593] | Cue RPE: -0.189 [-0.287, -0.130]
Lives  200 | Food RPE: -0.138 [-0.315, 0.146] | Cue RPE: -0.207 [-0.331, -0.095]
Lives 1000 | Food RPE: -0.057 [-0.135, 0.066] | Cue RPE: 0.142 [0.070, 0.221]
Lives 3000 | Food RPE: -0.027 [-0.098, 0.036] | Cue RPE: 0.183 [0.098, 0.359]
Seeds: 10/10

E5: Reward omission (10 seeds)
Food delivered | RPE: -0.027 [-0.098, 0.036]
Food omitted   | RPE: -1.337 [-1.625, -1.197]
Seeds: 10/10""".splitlines()
        self.assertEqual(output.getvalue().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
