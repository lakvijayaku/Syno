"""
Experiment tests for experiments/pain.py.

The fast tests check the experiment's settings, where thorns are placed,
what SYNO senses, and that stepping onto a thorn hurts SYNO while standing
on one, eating, and walking elsewhere do not. The slow test trains SYNO for
the full 3000 lives (about 2 minutes) and reproduces the recorded energy and
thorn curve. It is skipped by default and runs only when SYNO_SLOW_TESTS is
set:

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
import experiments.pain as pain
import experiments.scarcity as scarcity
from experiments.xor import make_layer
from syno.body.homeostasis import InjuredCore
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.world.habitat import Habitat

WINDOW = (2 * pain.SENSE_RADIUS + 1) ** 2
INPUTS = 2 * WINDOW + 3


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(pain.HIDDEN_NEURONS, INPUTS),
        make_layer(len(pain.ACTION_NAMES), pain.HIDDEN_NEURONS, "linear"),
    ])


class TestPainSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_settings(self):
        self.assertEqual(pain.THORNS, 2)
        self.assertEqual(pain.THORN_DAMAGE, 0.4)
        self.assertEqual(pain.HIDDEN_NEURONS, 16)

    def test_shares_the_scarcity_world(self):
        self.assertEqual(pain.GRID_SIZE, scarcity.GRID_SIZE)
        self.assertEqual(pain.REGROW_DELAY, scarcity.REGROW_DELAY)
        self.assertEqual(pain.LIVES, scarcity.LIVES)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "DISCOUNT", "FOOD_AMOUNT"):
            with self.subTest(name=name):
                self.assertEqual(getattr(pain, name), getattr(consolidation, name))

    def test_actions_are_unchanged(self):
        self.assertEqual(pain.ACTION_NAMES, ["up", "down", "left", "right", "eat"])


class TestFreeSquare(unittest.TestCase):
    """Verifies that thorns land only on empty squares."""

    def test_avoids_syno_food_and_thorns(self):
        random.seed(0)
        for _ in range(200):
            habitat = Habitat(3, 1, (0, 0), [(1, 0)], hazards=[])
            habitat.hazards.append(pain.free_square(habitat))
            self.assertEqual(habitat.hazards, [(2, 0)])

    def test_two_thorns_never_share_a_square(self):
        random.seed(1)
        for _ in range(200):
            habitat = Habitat(3, 1, (0, 0), [])
            for _ in range(2):
                habitat.hazards.append(pain.free_square(habitat))
            self.assertEqual(sorted(habitat.hazards), [(1, 0), (2, 0)])


class TestPainSenses(unittest.TestCase):
    """Verifies what SYNO senses."""

    def test_layout(self):
        habitat = Habitat(5, 5, (2, 2), [(3, 2)], hazards=[(1, 2)])
        body = InjuredCore(0.7, 0.4, 0.3)
        state = pain.senses(habitat, body)
        self.assertEqual(len(state), INPUTS)
        self.assertEqual(state[:WINDOW], habitat.sense(pain.SENSE_RADIUS))
        self.assertEqual(state[WINDOW:2 * WINDOW], habitat.sense_hazard(pain.SENSE_RADIUS))
        self.assertAlmostEqual(state[-3], 0.3)
        self.assertEqual(state[-2:], [0.4, 0.3])


class TestPainLive(unittest.TestCase):
    """Verifies when SYNO is hurt during a life."""

    def run_life(self, actions):
        """
        Runs one life in a fixed 3x1 world where SYNO starts at the left, a
        thorn is in the middle, and food is on the thorn, choosing the given
        actions in turn (the last one repeats). Returns live's result and
        SYNO's injury after every step.
        """
        injuries = []
        tick = InjuredCore.tick

        def recording_tick(body, moved):
            tick(body, moved)
            injuries.append(body.injury)

        def fixed_habitat(width, height, agent, food):
            return Habitat(3, 1, (0, 0), [])

        plan = iter(actions)
        last = [actions[-1]]

        def next_action(values, epsilon):
            last[0] = next(plan, last[0])
            return pain.ACTION_NAMES.index(last[0])

        thorns = iter([(1, 0), (2, 0)])
        random.seed(0)
        with mock.patch.object(pain, "Habitat", side_effect=fixed_habitat), \
                mock.patch.object(pain, "free_square", side_effect=lambda h: next(thorns)), \
                mock.patch.object(pain, "spawn_food", side_effect=lambda h: h.food.append((1, 0))), \
                mock.patch.object(pain, "choose_action", side_effect=next_action), \
                mock.patch.object(InjuredCore, "tick", recording_tick):
            result = pain.live(new_network(), MemoryStore(pain.MEMORY_CAPACITY))
        return result, injuries

    def test_stepping_onto_a_thorn_hurts(self):
        result, injuries = self.run_life(["right", "eat"])
        self.assertAlmostEqual(injuries[0], pain.THORN_DAMAGE - 0.02)
        self.assertEqual(result[1], 1)

    def test_standing_and_eating_on_a_thorn_do_not_hurt(self):
        _, injuries = self.run_life(["right", "eat"])
        for before, after in zip(injuries, injuries[1:]):
            self.assertLessEqual(after, before)

    def test_bumping_a_wall_does_not_hurt(self):
        result, injuries = self.run_life(["left"])
        self.assertEqual(result[1], 0)
        self.assertEqual(max(injuries), 0.0)

    def test_bumping_a_wall_on_a_thorn_does_not_hurt(self):
        # Two steps onto thorns, then SYNO stands on the thorn at (2, 0) and
        # walks into the wall on every remaining step.
        result, injuries = self.run_life(["right", "right", "right"])
        self.assertEqual(result[1], 2)
        for before, after in zip(injuries[1:], injuries[2:]):
            self.assertLessEqual(after, before)

    def test_every_step_onto_a_thorn_counts(self):
        # right onto the thorn at (1, 0), right onto the thorn at (2, 0),
        # then left and right in turn, each a step onto a thorn.
        result, _ = self.run_life(["right", "right"] + ["left", "right"] * 49)
        self.assertEqual(result[1], pain.LIFE_STEPS)

    def test_thorns_are_placed_before_food(self):
        order = []
        random.seed(0)
        with mock.patch.object(pain, "free_square", side_effect=lambda h: order.append("thorn") or (0, 0)), \
                mock.patch.object(pain, "spawn_food", side_effect=lambda h: order.append("food")), \
                mock.patch.object(pain, "LIFE_STEPS", 0):
            try:
                pain.live(new_network(), MemoryStore(pain.MEMORY_CAPACITY))
            except ZeroDivisionError:
                pass
        self.assertEqual(order[:3], ["thorn", "thorn", "food"])

    def test_starts_unhurt(self):
        _, injuries = self.run_life(["left"])
        self.assertEqual(injuries[0], 0.0)

    def test_returns_averages_in_range(self):
        random.seed(0)
        energy, hits = pain.live(new_network(), MemoryStore(pain.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)
        self.assertGreaterEqual(hits, 0)
        self.assertLessEqual(hits, pain.LIFE_STEPS)


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestPainRun(unittest.TestCase):
    """Reproduces the recorded energy and thorn curve."""

    def test_energy_and_thorn_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pain.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.247 | Thorn Hits: 0.49",
            "Life 1000 | Avg Energy: 0.382 | Thorn Hits: 0.60",
            "Life 1500 | Avg Energy: 0.501 | Thorn Hits: 0.53",
            "Life 2000 | Avg Energy: 0.500 | Thorn Hits: 0.65",
            "Life 2500 | Avg Energy: 0.501 | Thorn Hits: 0.65",
            "Life 3000 | Avg Energy: 0.565 | Thorn Hits: 0.64",
        ])


if __name__ == "__main__":
    unittest.main()
