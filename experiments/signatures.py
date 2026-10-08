# ── SYNO · experiments/signatures.py ────────────────────
# Measures SYNO's behavior against biological signatures
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.memory import MemoryStore
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward
from syno.brain.dopamine import reward_prediction_error
from experiments.hunger import ACTION_NAMES, GRID_SIZE, HIDDEN_NEURONS, SENSE_RADIUS, senses, DISCOUNT, FOOD_AMOUNT, REWARD_SCALE, spawn_food
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


def eating_rpe(network: Network, food_present: bool) -> float:
    """
    Measures SYNO's RPE when it eats food it can see (E5).

    SYNO is placed on food in every square, half-full of energy, and eats.
    When food_present is False, the food vanishes after SYNO has seen it, so
    the meal it expected never comes. The step is otherwise identical to a
    step of live, including new food appearing after a meal.

    :param network: The trained Decision Network.
    :param food_present: False to remove the food just before SYNO eats.
    :return: SYNO's average RPE over every square.
    """
    total = 0.0
    squares = GRID_SIZE * GRID_SIZE
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [(x, y)])
            body = HomeostaticCore(0.5, 0.0)
            # SYNO's prediction is read while the food is still visible.
            expected = network.forward(senses(habitat, body))[ACTION_NAMES.index("eat")]
            if not food_present:
                habitat.food.clear()
            drive_before = body.drive()
            if habitat.eat():
                body.eat(FOOD_AMOUNT)
                spawn_food(habitat)
            body.tick(False)
            reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
            next_expected = max(network.forward(senses(habitat, body)))
            total += reward_prediction_error(reward, expected, next_expected, DISCOUNT, False)
    return total / squares


def main():
    """
    Trains SYNO with SEEDS different seeds, measures every signature on each
    network, and prints the results averaged over the seeds.
    """
    print(f"E2: State-dependent eating ({SEEDS} seeds)")
    level_totals = {level: 0.0 for level in ENERGY_LEVELS}
    delivered_total = 0.0
    omitted_total = 0.0
    for seed in range(SEEDS):
        network = train(seed)
        for level in ENERGY_LEVELS:
            level_totals[level] += eating_rate(network, level)
        # New food appears at random after a meal, so the same seed is used
        # for every network to keep the result reproducible.
        random.seed(0)
        delivered_total += eating_rpe(network, True)
        omitted_total += eating_rpe(network, False)
    for level in ENERGY_LEVELS:
        avg_rate = level_totals[level] / SEEDS
        print(f"Energy {level:.1f} | Eats: {avg_rate:.2f}")
    print()
    print(f"E5: Reward omission ({SEEDS} seeds)")
    print(f"Food delivered | RPE: {delivered_total / SEEDS:.3f}")
    print(f"Food omitted   | RPE: {omitted_total / SEEDS:.3f}")


if __name__ == "__main__":
    main()
