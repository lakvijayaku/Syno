"""
Experiment tests for experiments/dreaming.py.

The fast tests check the experiment's settings and that every life runs the
sleep experiment's life with replay moved into sleep. The slow test trains
SYNO for the full 3000 lives (about 80 seconds) and reproduces the recorded
energy and sleep curve. It is skipped by default and runs only when
SYNO_SLOW_TESTS is set:

    SYNO_SLOW_TESTS=1 python3 -m unittest discover tests -v

Run the fast tests from the repository root with:
    python3 -m unittest discover tests -v
"""

import contextlib
import io
import os
import unittest
from unittest import mock

import experiments.dreaming as dreaming
import experiments.sleep as sleep


class TestDreamingSetup(unittest.TestCase):
    """Verifies the experiment's settings."""

    def test_settings(self):
        self.assertEqual(dreaming.AWAKE_REPLAYS, 0)
        self.assertEqual(dreaming.SLEEP_REPLAYS, 6)

    def test_replay_budget_is_close_to_the_sleep_experiment(self):
        # SYNO sleeps about 30% of the time, so replays per step average
        # about 1.8, within half a replay of the sleep experiment's 2.
        average = 0.3 * dreaming.SLEEP_REPLAYS + 0.7 * dreaming.AWAKE_REPLAYS
        self.assertLess(abs(average - sleep.REPLAYS_PER_STEP), 0.5)

    def test_uses_the_sleep_experiment(self):
        self.assertIs(dreaming.sleep, sleep)


class TestDreamingMain(unittest.TestCase):
    """Verifies how main trains SYNO, with a short run."""

    def run_main(self, lives: int):
        """Runs main for the given number of lives and records each life."""
        calls = []

        def recording_live(network, memory, *replays):
            calls.append((network, memory, replays))
            return 0.5, 0.25

        output = io.StringIO()
        with mock.patch.object(sleep, "live", side_effect=recording_live), \
                mock.patch.object(sleep, "LIVES", lives), \
                contextlib.redirect_stdout(output):
            dreaming.main()
        return calls, output.getvalue().splitlines()

    def test_every_life_replays_only_in_sleep(self):
        calls, _ = self.run_main(3)
        self.assertEqual(len(calls), 3)
        for _, _, replays in calls:
            self.assertEqual(replays, (0, 6))

    def test_network_and_memory_last_across_lives(self):
        calls, _ = self.run_main(3)
        self.assertEqual(len({id(network) for network, _, _ in calls}), 1)
        self.assertEqual(len({id(memory) for _, memory, _ in calls}), 1)
        self.assertEqual(calls[0][1].capacity, sleep.MEMORY_CAPACITY)

    def test_network_shape(self):
        calls, _ = self.run_main(1)
        network = calls[0][0]
        self.assertEqual(len(network.layers[0].neurons), sleep.HIDDEN_NEURONS)
        self.assertEqual(len(network.layers[0].neurons[0].weights), sleep.WINDOW + 3)
        self.assertEqual(len(network.layers[-1].neurons), len(sleep.ACTIONS))

    def test_prints_every_500_lives(self):
        _, lines = self.run_main(1000)
        self.assertEqual(lines, [
            "Life  500 | Avg Energy: 0.500 | Time Asleep: 0.250",
            "Life 1000 | Avg Energy: 0.500 | Time Asleep: 0.250",
        ])


@unittest.skipUnless(os.environ.get("SYNO_SLOW_TESTS"), "slow: set SYNO_SLOW_TESTS=1")
class TestDreamingRun(unittest.TestCase):
    """Reproduces the recorded energy and sleep curve."""

    def test_energy_and_sleep_curve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dreaming.main()
        self.assertEqual(output.getvalue().splitlines(), [
            "Life  500 | Avg Energy: 0.316 | Time Asleep: 0.398",
            "Life 1000 | Avg Energy: 0.530 | Time Asleep: 0.312",
            "Life 1500 | Avg Energy: 0.626 | Time Asleep: 0.304",
            "Life 2000 | Avg Energy: 0.673 | Time Asleep: 0.302",
            "Life 2500 | Avg Energy: 0.685 | Time Asleep: 0.303",
            "Life 3000 | Avg Energy: 0.693 | Time Asleep: 0.303",
        ])


if __name__ == "__main__":
    unittest.main()
