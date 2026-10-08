"""
Experiment tests for experiments/forage.py.

These tests confirm that SYNO learns to find and eat food from its own
reward prediction error, with no answer key. They check the experiment's
building blocks, reproduce the recorded learning curve, and compare a
trained SYNO against random behavior.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import random
import unittest

from experiments.forage import (
    ACTION_NAMES,
    DISCOUNT,
    EPSILON,
    GRID_SIZE,
    HIDDEN_NEURONS,
    LEARNING_RATE,
    MAX_STEPS,
    SENSE_RADIUS,
    random_habitat,
    run_episode,
)
from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.world.habitat import Habitat


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * SENSE_RADIUS + 1) ** 2
    return Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS),
    ])


def greedy_results(network: Network) -> list[int]:
    """
    Runs SYNO from every possible pair of starting squares for SYNO and food,
    always picking its best action and never learning.

    This judges what SYNO has learned without exploration or further
    training. Every pair is tried, so no random choice can hide a weakness.

    :return: The steps taken for each of the 72 pairs, MAX_STEPS if SYNO never ate.
    """
    squares = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]
    results = []
    for agent in squares:
        for food in squares:
            if agent == food:
                continue
            habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [food])
            for step in range(1, MAX_STEPS + 1):
                name = ACTION_NAMES[choose_action(network.forward(habitat.sense(SENSE_RADIUS)), 0.0)]
                if name == "eat":
                    if habitat.eat():
                        break
                else:
                    habitat.move(name)
            results.append(step)
    return results


class TestForageBuildingBlocks(unittest.TestCase):
    """Verifies the pieces of the experiment."""

    def test_random_habitat_never_starts_syno_on_food(self):
        random.seed(0)
        for _ in range(300):
            habitat = random_habitat()
            self.assertEqual(len(habitat.food), 1)
            self.assertNotEqual(habitat.agent, habitat.food[0])

    def test_random_habitat_uses_the_configured_grid(self):
        random.seed(1)
        habitat = random_habitat()
        self.assertEqual((habitat.width, habitat.height), (GRID_SIZE, GRID_SIZE))

    def test_senses_cover_the_whole_grid(self):
        # From any square, the window must include every square of the grid,
        # so the food is always visible.
        self.assertGreaterEqual(SENSE_RADIUS, GRID_SIZE - 1)

    def test_settings_are_in_valid_ranges(self):
        self.assertTrue(0.0 <= EPSILON <= 1.0)
        self.assertTrue(0.0 <= DISCOUNT <= 1.0)
        self.assertGreater(LEARNING_RATE, 0.0)

    def test_run_episode_returns_a_step_count_in_range(self):
        random.seed(2)
        network = new_network()
        for _ in range(20):
            self.assertIn(run_episode(network), range(1, MAX_STEPS + 1))


class TestForageLearning(unittest.TestCase):
    """Verifies that SYNO learns to forage."""

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.network = new_network()
        cls.steps = [run_episode(cls.network) for _ in range(3000)]

    def average(self, start: int, end: int) -> float:
        return sum(self.steps[start:end]) / (end - start)

    def test_seed_zero_matches_the_recorded_learning_curve(self):
        recorded = [38.748, 24.49, 10.22, 7.612, 6.21, 6.08]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                self.assertAlmostEqual(self.average(block * 500, (block + 1) * 500), expected)

    def test_last_episodes_are_much_faster_than_the_first(self):
        self.assertLess(self.average(2500, 3000), self.average(0, 500) / 4)

    def test_trained_syno_solves_most_starting_positions(self):
        # Without exploration, a deterministic policy can get stuck in a loop
        # on a few positions, so this requires most, not all, to be solved.
        results = greedy_results(self.network)
        solved = [steps for steps in results if steps < MAX_STEPS]
        self.assertGreaterEqual(len(solved), 60)
        # Solved positions should take close to the shortest path: on a 3x3
        # grid, the best possible average is 3 steps including eating.
        self.assertLess(sum(solved) / len(solved), 4.0)

    def test_trained_syno_beats_untrained_syno(self):
        random.seed(10)
        untrained = greedy_results(new_network())
        trained = greedy_results(self.network)
        self.assertGreater(
            sum(steps < MAX_STEPS for steps in trained),
            sum(steps < MAX_STEPS for steps in untrained),
        )

if __name__ == "__main__":
    unittest.main()
