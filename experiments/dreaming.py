# ── SYNO · experiments/dreaming.py ──────────────────────
# SYNO replays only while it sleeps
# Laksheth Vijayakumar · 2026-10-15 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.memory import MemoryStore

# The world, body, senses, and life all come from the sleep experiment; only
# when SYNO replays its memories differs.
import experiments.sleep as sleep

# SYNO replays nothing while awake, so all of its consolidation happens in
# sleep, as in the hippocampus. It sleeps about 30% of the time, so 6 replays
# per sleeping step average about 1.8 per step, close to the sleep
# experiment's 2: replay is moved into sleep, not increased.
AWAKE_REPLAYS = 0
SLEEP_REPLAYS = 6


def main():
    """
    Trains SYNO over many lives, replaying memories only while it sleeps, and
    prints its average energy and time asleep every 500 lives.
    """
    random.seed(0)

    # The view, then the energy deficit, stomach fill, and fatigue.
    inputs = sleep.WINDOW + 3

    network = Network([
        make_layer(sleep.HIDDEN_NEURONS, inputs),
        make_layer(len(sleep.ACTIONS), sleep.HIDDEN_NEURONS, "linear")
    ])

    energies = []
    sleep_fractions = []
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(sleep.MEMORY_CAPACITY)

    for life in range(1, sleep.LIVES + 1):
        energy, sleep_fraction = sleep.live(network, memory, AWAKE_REPLAYS, SLEEP_REPLAYS)
        energies.append(energy)
        sleep_fractions.append(sleep_fraction)

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            avg_sleep = sum(sleep_fractions[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f} | Time Asleep: {avg_sleep:.3f}")


if __name__ == "__main__":
    main()
