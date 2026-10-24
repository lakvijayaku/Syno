"""
Experiment tests for experiments/blink.py.

The fast tests check the experiment's settings, what SYNO sees with its eyes
open and closed, and that working memory is emptied at the start of each
life and holds what SYNO saw before each step. The slow test trains SYNO for
the full 3000 lives (about 2 minutes) and reproduces the recorded energy
curve. It is skipped by default and runs only when SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import random
import unittest
from unittest import mock

import experiments.blink as blink
import experiments.consolidation as consolidation
import experiments.scarcity as scarcity
from experiments.xor import make_layer
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.brain.working_memory import WorkingMemory
from syno.world.habitat import Habitat

INPUTS = 2 * blink.WINDOW + 2


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(blink.HIDDEN_NEURONS, INPUTS),
        make_layer(len(blink.ACTION_NAMES), blink.HIDDEN_NEURONS, "linear"),
    ])


class TestBlinkSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_settings(self):
        self.assertEqual(blink.DARK_CHANCE, 0.5)
        self.assertEqual(blink.MEMORY_DECAY, 0.5)
        self.assertEqual(blink.HIDDEN_NEURONS, 16)
        self.assertEqual(blink.LIVES, 3000)
        self.assertEqual(blink.WINDOW, 25)

    def test_shares_the_scarcity_world(self):
        self.assertEqual(blink.GRID_SIZE, scarcity.GRID_SIZE)
        self.assertEqual(blink.REGROW_DELAY, scarcity.REGROW_DELAY)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "DISCOUNT", "FOOD_AMOUNT"):
            with self.subTest(name=name):
                self.assertEqual(getattr(blink, name), getattr(consolidation, name))


class TestView(unittest.TestCase):
    """Verifies what SYNO sees with its eyes open and closed."""

    def test_open_eyes_see_everything(self):
        habitat = Habitat(5, 5, (0, 0), [(1, 1)])
        self.assertEqual(blink.view(habitat, True), habitat.sense(blink.SENSE_RADIUS))

    def test_closed_eyes_hide_food(self):
        habitat = Habitat(5, 5, (2, 2), [(3, 2), (0, 0)])
        self.assertEqual(blink.view(habitat, False), [0.0] * blink.WINDOW)

    def test_closed_eyes_still_feel_walls(self):
        habitat = Habitat(5, 5, (0, 0), [(1, 0)])
        seen = habitat.sense(blink.SENSE_RADIUS)
        dark = blink.view(habitat, False)
        self.assertEqual([value == -1.0 for value in dark], [value == -1.0 for value in seen])
        self.assertNotIn(1.0, dark)

    def test_closed_eyes_feel_food_underneath(self):
        habitat = Habitat(5, 5, (2, 2), [(2, 2), (4, 4)])
        dark = blink.view(habitat, False)
        self.assertEqual(dark[blink.WINDOW // 2], 1.0)
        self.assertEqual(dark.count(1.0), 1)

    def test_view_does_not_change_the_habitat(self):
        habitat = Habitat(5, 5, (2, 2), [(2, 2)])
        blink.view(habitat, False)
        self.assertEqual(habitat.food, [(2, 2)])


class TestBlinkLive(unittest.TestCase):
    """Verifies how working memory is used during a life."""

    def run_life(self, eyes_open: bool, working_memory: WorkingMemory):
        """
        Runs one life with SYNO's eyes always open or always closed after the
        first step, and returns the (state, next_state) of every live update.
        """
        updates = []
        original = blink.learn

        def recording_learn(network, state, action, reward, next_state, *rest):
            """
            Records the state and next state of every update, then learns as
            normal.
            """
            updates.append((state, next_state))
            return original(network, state, action, reward, next_state, *rest)

        random.seed(0)
        network = new_network()
        chance = 0.99 if eyes_open else 0.0
        with mock.patch.object(blink, "learn", side_effect=recording_learn), \
                mock.patch.object(blink.random, "random", return_value=chance):
            blink.live(network, MemoryStore(blink.MEMORY_CAPACITY), working_memory)
        # Each step makes one live update and REPLAYS_PER_STEP replays.
        return updates[::1 + blink.REPLAYS_PER_STEP]

    def test_memory_is_emptied_at_the_start_of_a_life(self):
        working_memory = WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY)
        working_memory.update([1.0] * blink.WINDOW)
        updates = self.run_life(True, working_memory)
        first_state = updates[0][0]
        self.assertEqual(first_state[-blink.WINDOW:], [0.0] * blink.WINDOW)

    def test_memory_holds_what_was_seen_before_the_step(self):
        updates = self.run_life(True, WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY))
        first_state, first_next = updates[0]
        seen_before = first_state[:blink.WINDOW]
        expected = [0.5 * value for value in seen_before]
        self.assertEqual(first_next[-blink.WINDOW:], expected)

    def test_state_layout(self):
        updates = self.run_life(True, WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY))
        for state, next_state in updates[:5]:
            self.assertEqual(len(state), INPUTS)
            self.assertEqual(len(next_state), INPUTS)

    def test_closed_eyes_see_no_distant_food(self):
        updates = self.run_life(False, WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY))
        for _, next_state in updates:
            view = next_state[:blink.WINDOW]
            distant = view[:blink.WINDOW // 2] + view[blink.WINDOW // 2 + 1:]
            self.assertNotIn(1.0, distant)

    def test_memory_keeps_food_after_the_eyes_close(self):
        # The first view is always seen. With the eyes closed afterwards,
        # food seen at the start is still in memory one step later, at half
        # strength, even though the view no longer shows it.
        updates = self.run_life(False, WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY))
        first_state, first_next = updates[0]
        seen = first_state[:blink.WINDOW]
        remembered = first_next[-blink.WINDOW:]
        for index, value in enumerate(seen):
            if value == 1.0:
                self.assertEqual(remembered[index], 0.5)

    def test_returns_average_energy(self):
        random.seed(0)
        energy = blink.live(new_network(), MemoryStore(blink.MEMORY_CAPACITY),
                            WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestBlinkRun(unittest.TestCase):
    """Reproduces the recorded energy curve."""

    def test_energy_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            blink.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.271",
            "Life 1000 | Avg Energy: 0.390",
            "Life 1500 | Avg Energy: 0.417",
            "Life 2000 | Avg Energy: 0.437",
            "Life 2500 | Avg Energy: 0.461",
            "Life 3000 | Avg Energy: 0.485",
        ])


if __name__ == "__main__":
    unittest.main()
