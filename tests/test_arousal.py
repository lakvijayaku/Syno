"""
Experiment tests for experiments/arousal.py.

The fast tests check the experiment's settings, the learning rate set by
arousal, and that SYNO can live on the larger grid. The slow test runs both
versions of SYNO (about 80 seconds) and reproduces the recorded comparison.
It is skipped by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

import experiments.arousal as arousal
import experiments.consolidation as consolidation
from experiments.hunger import spawn_food
from experiments.xor import make_layer
from syno.body.hormones import Hormone
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.world.habitat import Habitat


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * arousal.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(arousal.HIDDEN_NEURONS, inputs),
        make_layer(len(arousal.ACTION_NAMES), arousal.HIDDEN_NEURONS, "linear"),
    ])


class TestArousalSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "NOVELTY_SCALE", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR", "LEARNING_RATE"):
            with self.subTest(name=name):
                self.assertEqual(getattr(arousal, name), getattr(consolidation, name))

    def test_world_grows_partway_through(self):
        self.assertLess(arousal.SMALL_GRID, arousal.LARGE_GRID)
        self.assertLess(arousal.MOVE_LIFE, arousal.LIVES)

    def test_learning_rate_range_contains_the_fixed_rate(self):
        self.assertLess(arousal.LR_MIN, arousal.LEARNING_RATE)
        self.assertGreater(arousal.LR_MAX, arousal.LEARNING_RATE)

    def test_spawn_food_works_on_any_grid_size(self):
        random.seed(0)
        for size in (2, 3, 5, 8):
            for _ in range(50):
                habitat = Habitat(size, size, (0, 0), [])
                spawn_food(habitat)
                self.assertTrue(habitat.in_bounds(habitat.food[0]))
                self.assertNotEqual(habitat.food[0], (0, 0))

    def test_spawn_food_reaches_every_square_of_a_large_grid(self):
        random.seed(1)
        squares = set()
        for _ in range(2000):
            habitat = Habitat(5, 5, (0, 0), [])
            spawn_food(habitat)
            squares.add(habitat.food[0])
        self.assertEqual(len(squares), 24)


class TestArousalLife(unittest.TestCase):
    """Verifies a life with and without arousal."""

    def test_live_on_both_grids(self):
        random.seed(2)
        network = new_network()
        memory = MemoryStore(arousal.MEMORY_CAPACITY)
        for size in (arousal.SMALL_GRID, arousal.LARGE_GRID):
            with self.subTest(size=size):
                energy = arousal.live(network, memory, size, None)
                self.assertGreaterEqual(energy, 0.0)
                self.assertLessEqual(energy, 1.0)

    def test_surprise_raises_arousal(self):
        random.seed(3)
        hormone = Hormone(arousal.AROUSAL_RATE)
        arousal.live(new_network(), MemoryStore(arousal.MEMORY_CAPACITY), arousal.SMALL_GRID, hormone)
        self.assertGreater(hormone.level, 0.0)

    def test_fixed_rate_life_is_reproducible(self):
        # A fixed-rate life must not depend on any hormone, so two runs with
        # the same seed give identical results.
        results = []
        for _ in range(2):
            random.seed(4)
            results.append(arousal.live(new_network(), MemoryStore(arousal.MEMORY_CAPACITY), arousal.SMALL_GRID, None))
        self.assertEqual(results[0], results[1])


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestArousalComparison(unittest.TestCase):
    """Verifies the recorded comparison (slow)."""

    @classmethod
    def setUpClass(cls):
        cls.fixed = arousal.run(False)
        cls.aroused = arousal.run(True)

    def blocks(self, energies: list[float]) -> list[float]:
        return [sum(energies[i:i + 250]) / 250 for i in range(0, arousal.LIVES, 250)]

    def test_seed_zero_matches_the_recorded_comparison(self):
        recorded_fixed = [0.563, 0.881, 0.897, 0.924, 0.456, 0.697, 0.759, 0.797, 0.778, 0.796]
        recorded_arousal = [0.552, 0.880, 0.914, 0.904, 0.650, 0.777, 0.781, 0.788, 0.779, 0.801]
        for block, (fixed, aroused) in enumerate(zip(self.blocks(self.fixed), self.blocks(self.aroused))):
            with self.subTest(block=block):
                self.assertAlmostEqual(fixed, recorded_fixed[block], places=3)
                self.assertAlmostEqual(aroused, recorded_arousal[block], places=3)

    def test_arousal_recovers_faster_after_the_move(self):
        self.assertGreater(self.blocks(self.aroused)[4], self.blocks(self.fixed)[4] + 0.1)


if __name__ == "__main__":
    unittest.main()
