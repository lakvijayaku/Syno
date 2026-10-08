# ── SYNO · experiments/signatures.py ────────────────────
# Measures SYNO's behavior against biological signatures
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.memory import MemoryStore
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore
from experiments.hunger import ACTION_NAMES, GRID_SIZE, HIDDEN_NEURONS, SENSE_RADIUS, senses
from experiments.replay import MEMORY_CAPACITY
from experiments.consolidation import live

# Each signature is measured on SYNO trained with surprise-weighted replay,
# the best learner so far, and averaged over several seeds, so a result is
# never the luck of one run. 1500 lives is enough to master the 3x3 world.
TRAINING_LIVES = 1500
SEEDS = 5
# The energy levels SYNO is tested at, from empty (0.0) to full (1.0).
ENERGY_LEVELS = [i / 10 for i in range(11)]

def train(seed: int) -> Network:
    """
    Trains a new SYNO from scratch, exactly as in the consolidation experiment.

    :param seed: The random seed, so the same seed gives the same network.
    :return: The trained Decision Network.
    """
    random.seed(seed)
    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2
    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    for _ in range(TRAINING_LIVES):
        live(network, memory)
    return network

def eating_rate(network: Network, energy: float) -> float:
    """
    Measures how often SYNO prefers to eat when standing on food (E2).

    SYNO is placed on food in every square of the grid, with an empty stomach
    and the given energy. Only its greedy choice is counted, so random
    exploration does not affect the result.

    :param network: The trained Decision Network.
    :param energy: SYNO's energy, from 0.0 to 1.0.
    :return: The fraction of squares where eating is SYNO's preferred action.
    """
    count = 0
    squares = GRID_SIZE * GRID_SIZE
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [(x, y)])
            body = HomeostaticCore(energy, 0.0)
            values = network.forward(senses(habitat, body))
            if values.index(max(values)) == ACTION_NAMES.index("eat"):
                count += 1
    return count / squares

def main():
    """
    Trains SYNO with SEEDS different seeds and prints, for every energy level,
    how often it prefers to eat, averaged over the seeds.
    """
    print(f"E2: State-dependent eating ({SEEDS} seeds)")

    level_totals = {level: 0.0 for level in ENERGY_LEVELS}

    for seed in range(SEEDS):
        network = train(seed)
        for level in ENERGY_LEVELS:
            level_totals[level] += eating_rate(network, level)

    for level in ENERGY_LEVELS:
        avg_rate = level_totals[level] / SEEDS
        print(f"Energy {level:.1f} | Eats: {avg_rate:.2f}")

if __name__ == "__main__":
    main()

