"""
Experiment tests for experiments/scarcity.py.

The fast tests check the experiment's settings, that food regrows after
exactly REGROW_DELAY steps, and that SYNO lives and learns on the larger
grid. The slow test trains SYNO for the full 3000 lives (about 50 seconds)
and reproduces the recorded energy curve. It is skipped by default and runs
only when SYNO_SLOW_TESTS is set:

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

import experiments.consolidation as consolidation
import experiments.scarcity as scarcity
from experiments.xor import make_layer
from syno.brain.memory import MemoryStore
from syno.brain.network import Network

EAT = scarcity.ACTION_NAMES.index("eat")


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    inputs = (2 * scarcity.SENSE_RADIUS + 1) ** 2 + 2
    return Network([
        make_layer(scarcity.HIDDEN_NEURONS, inputs),
        make_layer(len(scarcity.ACTION_NAMES), scarcity.HIDDEN_NEURONS, "linear"),
    ])


class TestScarcitySetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_world_is_larger_than_the_view(self):
        self.assertEqual(scarcity.GRID_SIZE, 5)
        self.assertGreater(scarcity.GRID_SIZE, scarcity.SENSE_RADIUS + 1)

    def test_food_regrows_after_a_delay(self):
        self.assertEqual(scarcity.REGROW_DELAY, 10)

    def test_lives(self):
        self.assertEqual(scarcity.LIVES, 3000)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "HIDDEN_NEURONS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(scarcity, name), getattr(consolidation, name))

    def test_does_not_change_the_small_world(self):
        self.assertEqual(consolidation.GRID_SIZE, 3)


class TestRegrowth(unittest.TestCase):
    """Verifies when food reappears, with SYNO choosing to eat every step."""

    def run_eating_life(self):
        """
        Runs one life where SYNO always eats and new food always appears under
        it, and returns the steps at which food appeared and the meals eaten.
        """
        spawns = []
        meals = []
        step = [0]

        def food_under_syno(habitat):
            spawns.append(step[0])
            habitat.food.append(habitat.agent)

        def always_eat(values, epsilon):
            return EAT

        original_eat = scarcity.HomeostaticCore.eat

        def counting_eat(body, amount):
            meals.append(step[0])
            original_eat(body, amount)

        original_tick = scarcity.HomeostaticCore.tick

        def counting_tick(body, moved):
            original_tick(body, moved)
            step[0] += 1

        random.seed(0)
        with mock.patch.object(scarcity, "spawn_food", side_effect=food_under_syno), \
                mock.patch.object(scarcity, "choose_action", side_effect=always_eat), \
                mock.patch.object(scarcity.HomeostaticCore, "eat", counting_eat), \
                mock.patch.object(scarcity.HomeostaticCore, "tick", counting_tick):
            scarcity.live(new_network(), MemoryStore(scarcity.MEMORY_CAPACITY))
        return spawns, meals

    def test_meals_are_regrow_delay_apart(self):
        spawns, meals = self.run_eating_life()
        self.assertEqual(meals, list(range(0, scarcity.LIFE_STEPS, scarcity.REGROW_DELAY)))

    def test_food_reappears_after_regrow_delay(self):
        # The first spawn starts the life. Each later one comes at the end of
        # the step REGROW_DELAY - 1 steps after the meal, ready for the next.
        spawns, meals = self.run_eating_life()
        self.assertEqual(spawns[0], 0)
        expected = [meal + scarcity.REGROW_DELAY - 1 for meal in meals]
        self.assertEqual(spawns[1:], expected)

    def test_food_is_never_doubled(self):
        spawns, meals = self.run_eating_life()
        self.assertEqual(len(spawns), len(meals) + 1)


class TestRegrowthWhileMoving(unittest.TestCase):
    """Verifies that food regrows, and moving costs energy, while SYNO walks."""

    def test_food_regrows_while_syno_moves(self):
        # SYNO eats on the first step, then walks left and right. Food must
        # still reappear REGROW_DELAY - 1 steps later, and each step must
        # report whether SYNO moved.
        spawns = []
        moves = []
        step = [0]
        walk = [scarcity.ACTION_NAMES.index("left"), scarcity.ACTION_NAMES.index("right")]

        def food_under_syno(habitat):
            spawns.append(step[0])
            habitat.food.append(habitat.agent)

        def eat_then_walk(values, epsilon):
            return EAT if step[0] == 0 else walk[step[0] % 2]

        original_tick = scarcity.HomeostaticCore.tick

        def recording_tick(body, moved):
            moves.append(moved)
            original_tick(body, moved)
            step[0] += 1

        random.seed(0)
        with mock.patch.object(scarcity, "spawn_food", side_effect=food_under_syno), \
                mock.patch.object(scarcity, "choose_action", side_effect=eat_then_walk), \
                mock.patch.object(scarcity.HomeostaticCore, "tick", recording_tick):
            scarcity.live(new_network(), MemoryStore(scarcity.MEMORY_CAPACITY))
        self.assertEqual(spawns, [0, scarcity.REGROW_DELAY - 1])
        self.assertFalse(moves[0])
        self.assertTrue(any(moves[1:]))


class TestScarcityLive(unittest.TestCase):
    """Verifies that SYNO lives and learns on the larger grid."""

    def test_live_returns_average_energy(self):
        random.seed(0)
        energy = scarcity.live(new_network(), MemoryStore(scarcity.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)

    def test_agent_uses_the_whole_grid(self):
        positions = set()
        original = scarcity.Habitat.move

        def recording_move(habitat, name):
            moved = original(habitat, name)
            positions.add(habitat.agent)
            return moved

        random.seed(0)
        network = new_network()
        memory = MemoryStore(scarcity.MEMORY_CAPACITY)
        with mock.patch.object(scarcity.Habitat, "move", recording_move):
            for _ in range(5):
                scarcity.live(network, memory)
        self.assertTrue(any(max(position) >= 3 for position in positions))

    def test_live_learns(self):
        random.seed(0)
        network = new_network()
        state = [0.0] * len(network.layers[0].neurons[0].weights)
        before = network.forward(state)
        memory = MemoryStore(scarcity.MEMORY_CAPACITY)
        scarcity.live(network, memory)
        self.assertNotEqual(network.forward(state), before)
        self.assertEqual(len(memory), scarcity.LIFE_STEPS)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestScarcityRun(unittest.TestCase):
    """Reproduces the recorded energy curve."""

    def test_energy_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scarcity.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.302",
            "Life 1000 | Avg Energy: 0.545",
            "Life 1500 | Avg Energy: 0.632",
            "Life 2000 | Avg Energy: 0.650",
            "Life 2500 | Avg Energy: 0.659",
            "Life 3000 | Avg Energy: 0.663",
        ])


if __name__ == "__main__":
    unittest.main()
