"""
Experiment tests for experiments/replay.py.

The fast tests check that the replay experiment differs from the curiosity
experiment only by its memory, and that memory carries over between lives.
The slow test trains SYNO for the full 5000 lives (about 85 seconds) and
reproduces the recorded energy curve. It is skipped by default and runs only
when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

import experiments.curiosity as curiosity
import experiments.replay as replay
from experiments.xor import make_layer
from syno.brain.memory import MemoryStore
from syno.brain.network import Network


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * replay.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(replay.HIDDEN_NEURONS, inputs),
        make_layer(len(replay.ACTION_NAMES), replay.HIDDEN_NEURONS, "linear"),
    ])


class TestReplaySetup(unittest.TestCase):
    """Verifies the experiment's settings and memory use."""

    def test_shares_the_curiosity_settings(self):
        for name in ("EPSILON", "NOVELTY_SCALE", "HABITUATION", "RECOVERY", "LIVES", "LIFE_STEPS", "LEARNING_RATE"):
            with self.subTest(name=name):
                self.assertEqual(getattr(replay, name), getattr(curiosity, name))

    def test_replays_at_least_once_per_step(self):
        self.assertGreaterEqual(replay.REPLAYS_PER_STEP, 1)
        self.assertGreaterEqual(replay.MEMORY_CAPACITY, replay.LIFE_STEPS)

    def test_every_step_of_a_life_is_stored(self):
        random.seed(0)
        memory = MemoryStore(replay.MEMORY_CAPACITY)
        replay.live(new_network(), memory)
        self.assertEqual(len(memory), replay.LIFE_STEPS)

    def test_stored_experiences_are_complete(self):
        random.seed(1)
        memory = MemoryStore(replay.MEMORY_CAPACITY)
        replay.live(new_network(), memory)
        inputs = (2 * replay.SENSE_RADIUS + 1) ** 2 + 2
        for state, action, reward, next_state in memory.experiences:
            self.assertEqual(len(state), inputs)
            self.assertEqual(len(next_state), inputs)
            self.assertIn(action, range(len(replay.ACTION_NAMES)))
            self.assertIsInstance(reward, float)

    def test_memory_carries_over_between_lives(self):
        random.seed(2)
        memory = MemoryStore(replay.MEMORY_CAPACITY)
        network = new_network()
        replay.live(network, memory)
        replay.live(network, memory)
        self.assertEqual(len(memory), 2 * replay.LIFE_STEPS)

    def test_live_returns_an_average_energy(self):
        random.seed(3)
        energy = replay.live(new_network(), MemoryStore(replay.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestReplayLearning(unittest.TestCase):
    """Verifies that replay speeds up learning (slow)."""

    def test_seed_zero_matches_the_recorded_energy_curve(self):
        random.seed(0)
        network = new_network()
        memory = MemoryStore(replay.MEMORY_CAPACITY)
        energies = [replay.live(network, memory) for _ in range(replay.LIVES)]
        recorded = [0.239, 0.767, 0.870, 0.905, 0.923, 0.933, 0.919, 0.919, 0.933, 0.929]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                average = sum(energies[block * 500:(block + 1) * 500]) / 500
                self.assertAlmostEqual(average, expected, places=3)


if __name__ == "__main__":
    unittest.main()
