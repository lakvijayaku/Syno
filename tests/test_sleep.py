"""
Experiment tests for experiments/sleep.py.

The fast tests check the experiment's settings, what SYNO senses awake and
asleep, that a sleeping SYNO stays still, ticks its body as asleep, sees
with its eyes closed, and gets no novelty, and how many memories it replays
awake and asleep. The slow test trains SYNO for the full 3000 lives (about
80 seconds) and reproduces the recorded energy and sleep curve. It is
skipped by default and runs only when SYNO_SLOW_TESTS is set:

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
import experiments.sleep as sleep
from experiments.xor import make_layer
from syno.body.homeostasis import FatiguedCore
from syno.brain.memory import MemoryStore
from syno.brain.network import Network
from syno.brain.novelty import NoveltySystem
from syno.world.habitat import Habitat

INPUTS = sleep.WINDOW + 3
SLEEP = sleep.ACTIONS.index("sleep")
UP = sleep.ACTIONS.index("up")


def new_network() -> Network:
    """Builds an untrained network with the experiment's shape."""
    return Network([
        make_layer(sleep.HIDDEN_NEURONS, INPUTS),
        make_layer(len(sleep.ACTIONS), sleep.HIDDEN_NEURONS, "linear"),
    ])


class TestSleepSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_settings(self):
        self.assertEqual(sleep.ACTIONS, ["up", "down", "left", "right", "eat", "sleep"])
        self.assertEqual(sleep.HIDDEN_NEURONS, 16)
        self.assertEqual(sleep.LIVES, 3000)
        self.assertEqual(sleep.WINDOW, 25)

    def test_does_not_change_the_shared_actions(self):
        self.assertNotIn("sleep", sleep.ACTION_NAMES)

    def test_shares_the_scarcity_world(self):
        self.assertEqual(sleep.GRID_SIZE, scarcity.GRID_SIZE)
        self.assertEqual(sleep.REGROW_DELAY, scarcity.REGROW_DELAY)

    def test_shares_the_consolidation_settings(self):
        for name in ("EPSILON", "MEMORY_CAPACITY", "REPLAYS_PER_STEP", "PRIORITY_FLOOR",
                     "LIFE_STEPS", "LEARNING_RATE", "REWARD_SCALE", "DISCOUNT", "FOOD_AMOUNT"):
            with self.subTest(name=name):
                self.assertEqual(getattr(sleep, name), getattr(consolidation, name))


class TestSleepSenses(unittest.TestCase):
    """Verifies what SYNO senses awake and asleep."""

    def test_layout(self):
        habitat = Habitat(5, 5, (2, 2), [(3, 2)])
        body = FatiguedCore(0.7, 0.4, 0.3)
        state = sleep.senses(habitat, body, True)
        self.assertEqual(len(state), INPUTS)
        self.assertEqual(state[:sleep.WINDOW], blink.view(habitat, True))
        self.assertAlmostEqual(state[-3], 0.3)
        self.assertEqual(state[-2:], [0.4, 0.3])

    def test_closed_eyes_use_the_dark_view(self):
        habitat = Habitat(5, 5, (2, 2), [(3, 2)])
        state = sleep.senses(habitat, FatiguedCore(1.0, 0.0, 0.0), False)
        self.assertEqual(state[:sleep.WINDOW], blink.view(habitat, False))
        self.assertNotIn(1.0, state[:sleep.WINDOW])

    def test_does_not_change_the_body(self):
        body = FatiguedCore(0.7, 0.4, 0.3)
        sleep.senses(Habitat(5, 5, (2, 2), []), body, True)
        self.assertEqual((body.energy, body.stomach, body.fatigue), (0.7, 0.4, 0.3))


class TestSleepLive(unittest.TestCase):
    """Verifies a life when SYNO always sleeps or always stays awake."""

    def run_life(self, action: int) -> dict:
        """
        Runs one life choosing the same action on every step, and records
        every live update, every body tick, every novelty tick, and the
        habitat SYNO lives in.
        """
        record = {"updates": [], "ticks": [], "novelty_ticks": 0, "habitats": []}
        learn, tick = sleep.learn, FatiguedCore.tick
        novelty_tick, habitat_init = NoveltySystem.tick, Habitat.__init__

        def recording_learn(network, state, chosen, reward, next_state, *rest):
            """
            Records the state, reward, and next state of every update, then
            learns as normal.
            """
            record["updates"].append((state, reward, next_state))
            return learn(network, state, chosen, reward, next_state, *rest)

        def recording_tick(body, moved, slept=False):
            """Records how the body was ticked, then ticks it as normal."""
            record["ticks"].append((moved, slept))
            tick(body, moved, slept)

        def counting_novelty_tick(system):
            """Counts every novelty tick, then ticks as normal."""
            record["novelty_ticks"] += 1
            novelty_tick(system)

        def recording_habitat(habitat, *args, **kwargs):
            """
            Builds the habitat as normal, then records it and where SYNO
            started.
            """
            habitat_init(habitat, *args, **kwargs)
            record["habitats"].append(habitat)
            record["start"] = habitat.agent

        random.seed(0)
        with mock.patch.object(sleep, "learn", side_effect=recording_learn), \
                mock.patch.object(sleep, "choose_action", return_value=action), \
                mock.patch.object(FatiguedCore, "tick", recording_tick), \
                mock.patch.object(NoveltySystem, "tick", counting_novelty_tick), \
                mock.patch.object(Habitat, "__init__", recording_habitat):
            record["result"] = sleep.live(new_network(), MemoryStore(sleep.MEMORY_CAPACITY))
        # Each step makes one live update and REPLAYS_PER_STEP replays.
        record["updates"] = record["updates"][::1 + sleep.REPLAYS_PER_STEP]
        return record

    def test_always_asleep_reports_full_sleep(self):
        self.assertEqual(self.run_life(SLEEP)["result"][1], 1.0)

    def test_always_awake_reports_no_sleep(self):
        self.assertEqual(self.run_life(UP)["result"][1], 0.0)

    def test_sleep_ticks_the_body_asleep_and_still(self):
        ticks = self.run_life(SLEEP)["ticks"]
        self.assertEqual(len(ticks), sleep.LIFE_STEPS)
        self.assertEqual(set(ticks), {(False, True)})

    def test_awake_ticks_the_body_awake(self):
        ticks = self.run_life(UP)["ticks"]
        self.assertTrue(all(slept is False for _, slept in ticks))

    def test_sleeping_syno_stays_where_it_is(self):
        record = self.run_life(SLEEP)
        self.assertEqual(len(record["habitats"]), 1)
        self.assertEqual(record["habitats"][0].agent, record["start"])

    def test_awake_syno_moves(self):
        record = self.run_life(UP)
        self.assertEqual(record["habitats"][0].agent[1], 0)

    def test_eyes_are_closed_after_sleeping(self):
        for _, _, next_state in self.run_life(SLEEP)["updates"]:
            view = next_state[:sleep.WINDOW]
            distant = view[:sleep.WINDOW // 2] + view[sleep.WINDOW // 2 + 1:]
            self.assertNotIn(1.0, distant)

    def test_state_layout(self):
        for state, _, next_state in self.run_life(SLEEP)["updates"]:
            self.assertEqual(len(state), INPUTS)
            self.assertEqual(len(next_state), INPUTS)

    def test_no_novelty_while_asleep(self):
        self.assertEqual(self.run_life(SLEEP)["novelty_ticks"], 0)

    def test_novelty_every_waking_step(self):
        self.assertEqual(self.run_life(UP)["novelty_ticks"], sleep.LIFE_STEPS)

    def test_fatigue_falls_while_asleep(self):
        updates = self.run_life(SLEEP)["updates"]
        fatigues = [state[-1] for state, _, _ in updates] + [updates[-1][2][-1]]
        self.assertEqual(fatigues[-1], 0.0)
        self.assertTrue(all(b <= a for a, b in zip(fatigues, fatigues[1:])))

    def test_fatigue_rises_while_awake(self):
        updates = self.run_life(UP)["updates"]
        self.assertEqual(updates[-1][2][-1], 1.0)

    def test_returns_averages_in_range(self):
        random.seed(0)
        energy, asleep = sleep.live(new_network(), MemoryStore(sleep.MEMORY_CAPACITY))
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.0)
        self.assertGreaterEqual(asleep, 0.0)
        self.assertLessEqual(asleep, 1.0)


class TestSleepReplays(unittest.TestCase):
    """Verifies how many memories SYNO replays awake and asleep."""

    def count_learning(self, action: int, *replays: int) -> int:
        """
        Runs one life choosing the same action on every step, with the given
        replay counts, and returns how many times SYNO learned.
        """
        random.seed(0)
        with mock.patch.object(sleep, "learn", wraps=sleep.learn) as learned, \
                mock.patch.object(sleep, "choose_action", return_value=action):
            sleep.live(new_network(), MemoryStore(sleep.MEMORY_CAPACITY), *replays)
        return learned.call_count

    def test_defaults_replay_every_step(self):
        # One live update and REPLAYS_PER_STEP replays on every step.
        expected = sleep.LIFE_STEPS * (1 + sleep.REPLAYS_PER_STEP)
        self.assertEqual(self.count_learning(SLEEP), expected)
        self.assertEqual(self.count_learning(UP), expected)

    def test_sleep_replays_only_while_asleep(self):
        self.assertEqual(self.count_learning(SLEEP, 0, 6), sleep.LIFE_STEPS * 7)
        self.assertEqual(self.count_learning(UP, 0, 6), sleep.LIFE_STEPS)

    def test_awake_replays_only_while_awake(self):
        self.assertEqual(self.count_learning(UP, 5, 0), sleep.LIFE_STEPS * 6)
        self.assertEqual(self.count_learning(SLEEP, 5, 0), sleep.LIFE_STEPS)

    def test_no_replay(self):
        self.assertEqual(self.count_learning(SLEEP, 0, 0), sleep.LIFE_STEPS)

    def test_defaults_match_step_16(self):
        random.seed(0)
        default = sleep.live(new_network(), MemoryStore(sleep.MEMORY_CAPACITY))
        random.seed(0)
        explicit = sleep.live(new_network(), MemoryStore(sleep.MEMORY_CAPACITY),
                              sleep.REPLAYS_PER_STEP, sleep.REPLAYS_PER_STEP)
        self.assertEqual(default, explicit)

    def test_defaults(self):
        defaults = sleep.live.__defaults__
        self.assertEqual(defaults, (sleep.REPLAYS_PER_STEP, sleep.REPLAYS_PER_STEP))


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestSleepRun(unittest.TestCase):
    """Reproduces the recorded energy and sleep curve."""

    def test_energy_and_sleep_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sleep.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.323 | Time Asleep: 0.437",
            "Life 1000 | Avg Energy: 0.592 | Time Asleep: 0.311",
            "Life 1500 | Avg Energy: 0.675 | Time Asleep: 0.301",
            "Life 2000 | Avg Energy: 0.695 | Time Asleep: 0.300",
            "Life 2500 | Avg Energy: 0.695 | Time Asleep: 0.302",
            "Life 3000 | Avg Energy: 0.702 | Time Asleep: 0.301",
        ])


if __name__ == "__main__":
    unittest.main()
