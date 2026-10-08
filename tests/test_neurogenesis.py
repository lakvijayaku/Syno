"""
Experiment tests for experiments/neurogenesis.py.

The fast tests check the experiment's settings and that live reports
surprise. The slow test trains SYNO from a one-neuron brain for the full 2000
lives (about 25 seconds), reproduces the recorded energy curve and growth
history, and checks that the brain grew only at check points. It is skipped
by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import random
import unittest

import experiments.consolidation as consolidation
import experiments.neurogenesis as neurogenesis
from experiments.xor import make_layer
from syno.brain.growth import grow_neuron
from syno.brain.memory import MemoryStore
from syno.brain.network import Network


def new_network() -> Network:
    """Builds an untrained network with the experiment's starting size."""
    inputs = (2 * neurogenesis.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(neurogenesis.START_HIDDEN, inputs),
        make_layer(len(neurogenesis.ACTION_NAMES), neurogenesis.START_HIDDEN, "linear"),
    ])


class TestNeurogenesisSetup(unittest.TestCase):
    """Verifies the experiment's settings and surprise measurement."""

    def test_starts_small_with_room_to_grow(self):
        self.assertEqual(neurogenesis.START_HIDDEN, 1)
        self.assertGreater(neurogenesis.MAX_HIDDEN, neurogenesis.START_HIDDEN)

    def test_surprise_drop_is_a_fraction(self):
        self.assertGreater(neurogenesis.SURPRISE_DROP, 0.0)
        self.assertLess(neurogenesis.SURPRISE_DROP, 1.0)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR", "LIFE_STEPS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(neurogenesis, name), getattr(consolidation, name))

    def test_live_returns_energy_and_surprise(self):
        random.seed(0)
        energy, surprise = neurogenesis.live(new_network(), MemoryStore(neurogenesis.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)
        self.assertGreater(surprise, 0.0)

    def test_live_works_after_growing(self):
        random.seed(1)
        network = new_network()
        grow_neuron(network, 0)
        grow_neuron(network, 0)
        energy, _ = neurogenesis.live(network, MemoryStore(neurogenesis.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "set SYNO_SLOW_TESTS=1 to run")
class TestNeurogenesisLearning(unittest.TestCase):
    """Verifies that a growing brain learns the task (slow)."""

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        network = new_network()
        memory = MemoryStore(neurogenesis.MEMORY_CAPACITY)
        cls.energies, surprises, cls.sizes = [], [], []
        check = neurogenesis.GROWTH_CHECK
        for life in range(1, neurogenesis.LIVES + 1):
            energy, surprise = neurogenesis.live(network, memory)
            cls.energies.append(energy)
            surprises.append(surprise)
            if life % check == 0 and life >= 2 * check:
                recent = sum(surprises[-check:])
                earlier = sum(surprises[-2 * check:-check])
                if recent > neurogenesis.SURPRISE_DROP * earlier and len(network.layers[0].neurons) < neurogenesis.MAX_HIDDEN:
                    grow_neuron(network, 0)
            cls.sizes.append(len(network.layers[0].neurons))

    def test_seed_zero_matches_the_recorded_energy_curve(self):
        recorded = [0.203, 0.203, 0.291, 0.802, 0.902, 0.930, 0.924, 0.926]
        for block, expected in enumerate(recorded):
            with self.subTest(block=block):
                average = sum(self.energies[block * 250:(block + 1) * 250]) / 250
                self.assertAlmostEqual(average, expected, places=3)

    def test_seed_zero_matches_the_recorded_growth(self):
        grew_at = [life + 1 for life in range(1, len(self.sizes)) if self.sizes[life] > self.sizes[life - 1]]
        self.assertEqual(grew_at, [300, 500, 600, 700, 800, 1200, 1600, 1800, 2000])

    def test_brain_never_shrinks_or_exceeds_the_limit(self):
        for previous, current in zip(self.sizes, self.sizes[1:]):
            self.assertIn(current - previous, (0, 1))
        self.assertLessEqual(max(self.sizes), neurogenesis.MAX_HIDDEN)


if __name__ == "__main__":
    unittest.main()
