"""
Experiment tests for experiments/consolidation.py.

The fast tests check that the consolidation experiment differs from the
replay experiment only in how memories are prioritized. The slow test trains
SYNO for the full 5000 lives (about 80 seconds) and reproduces the recorded
energy curve. It is skipped by default and runs only when SYNO_SLOW_TESTS is
set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

import experiments.consolidation as consolidation
import experiments.replay as replay
from experiments.xor import make_layer
from syno.brain.memory import MemoryStore
from syno.brain.network import Network


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * consolidation.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(consolidation.HIDDEN_NEURONS, inputs),
        make_layer(len(consolidation.ACTION_NAMES), consolidation.HIDDEN_NEURONS, "linear"),
    ])


class TestConsolidationSetup(unittest.TestCase):
    """Verifies the experiment's settings and memory priorities."""

    def test_shares_the_replay_settings(self):
        for name in ("MEMORY_CAPACITY", "REPLAYS_PER_STEP", "EPSILON", "LIVES", "LIFE_STEPS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(consolidation, name), getattr(replay, name))

    def test_priority_floor_is_small_and_positive(self):
        self.assertGreater(consolidation.PRIORITY_FLOOR, 0.0)
        self.assertLess(consolidation.PRIORITY_FLOOR, 0.1)

    def test_every_memory_has_a_priority_above_the_floor(self):
        random.seed(0)
        memory = MemoryStore(consolidation.MEMORY_CAPACITY)
        consolidation.live(new_network(), memory)
        self.assertEqual(len(memory.priorities), consolidation.LIFE_STEPS)
        for priority in memory.priorities:
            self.assertGreaterEqual(priority, consolidation.PRIORITY_FLOOR)

    def test_priorities_vary_with_surprise(self):
        # If priorities were all equal, prioritized replay would be no
        # different from uniform replay.
        random.seed(1)
        memory = MemoryStore(consolidation.MEMORY_CAPACITY)
        consolidation.live(new_network(), memory)
        self.assertGreater(max(memory.priorities), 2 * min(memory.priorities))

    def test_live_returns_an_average_energy(self):
        random.seed(2)
        energy = consolidation.live(new_network(), MemoryStore(consolidation.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestConsolidationLearning(unittest.TestCase):
    """Verifies that surprise-weighted replay learns quickly and stays stable (slow)."""

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        network = new_network()
        memory = MemoryStore(consolidation.MEMORY_CAPACITY)
        cls.energies = [consolidation.live(network, memory) for _ in range(consolidation.LIVES)]

    def test_seed_zero_matches_the_recorded_energy_curve(self):
        recorded = [0.722, 0.911, 0.923, 0.929, 0.928, 0.922, 0.929, 0.926, 0.919, 0.917]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                average = sum(self.energies[block * 500:(block + 1) * 500]) / 500
                self.assertAlmostEqual(average, expected, places=3)

    def test_no_large_dip_after_learning(self):
        smoothed = [sum(self.energies[i - 99:i + 1]) / 100 for i in range(99, len(self.energies))]
        self.assertGreater(min(smoothed[900:]), 0.85)


if __name__ == "__main__":
    unittest.main()
