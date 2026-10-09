# ── SYNO · experiments/signatures.py ────────────────────
# Measures SYNO's behavior against biological signatures
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.memory import MemoryStore
from syno.world.habitat import Habitat, ACTIONS
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward
from syno.brain.dopamine import reward_prediction_error
from experiments.hunger import ACTION_NAMES, HIDDEN_NEURONS, SENSE_RADIUS, senses, DISCOUNT, FOOD_AMOUNT, REWARD_SCALE
from experiments.replay import MEMORY_CAPACITY
from experiments.scarcity import GRID_SIZE, LIVES, live

# Each signature is measured on SYNO trained in the scarce world, where food
# can be out of sight and takes time to regrow, and averaged over several
# seeds, so a result is never the luck of one run. Every probe tests every
# position, so measuring uses no random numbers and never changes training.
TRAINING_LIVES = LIVES
SEEDS = 5

# The points in training, in lives, at which E4 is measured.
CHECKPOINTS = [0, 50, 200, 1000, TRAINING_LIVES]

# The energy levels SYNO is tested at, from empty (0.0) to full (1.0).
ENERGY_LEVELS = [i / 10 for i in range(11)]

# For E1, SYNO is tested half-hungry at stomach fills from empty (0.0) to
# full (1.0), as if partway through a meal.
STOMACH_LEVELS = [i / 5 for i in range(6)]
SATIATION_ENERGY = 0.5

# For E3, SYNO is tested from hungry (0.2) to full (1.0), with food either
# next to it or at least 3 steps away.
APPROACH_ENERGIES = [0.2, 0.5, 0.8, 1.0]


def new_brain(seed: int) -> tuple[Network, MemoryStore]:
    """
    Creates an untrained SYNO, set up exactly as in the consolidation
    experiment.

    :param seed: The random seed, so the same seed gives the same brain and
        the same training afterwards.
    :return: The Decision Network and an empty Memory Store.
    """
    random.seed(seed)
    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2
    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    return network, memory


def train(network: Network, memory: MemoryStore, lives: int) -> None:
    """
    Trains SYNO for the given number of lives. Calling it several times in a
    row trains exactly as one longer call would, so SYNO can be measured part
    of the way through training.

    :param network: The Decision Network.
    :param memory: The Memory Store, shared across lives.
    :param lives: How many lives to train for.
    """
    for _ in range(lives):
        live(network, memory)


def eating_rate(network: Network, energy: float, stomach: float = 0.0) -> float:
    """
    Measures how often SYNO prefers to eat when standing on food (E1, E2).

    SYNO is placed on food in every square of the grid, with the given energy
    and stomach fill. Only its greedy choice is counted, so random
    exploration does not affect the result.

    :param network: The trained Decision Network.
    :param energy: SYNO's energy, from 0.0 to 1.0.
    :param stomach: SYNO's stomach fill, from 0.0 (empty) to 1.0 (full).
    :return: The fraction of squares where eating is SYNO's preferred action.
    """
    count = 0
    squares = GRID_SIZE * GRID_SIZE
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [(x, y)])
            body = HomeostaticCore(energy, stomach)
            values = network.forward(senses(habitat, body))
            if values.index(max(values)) == ACTION_NAMES.index("eat"):
                count += 1
    return count / squares


def approach_rate(network: Network, energy: float, far: bool) -> float:
    """
    Measures how often SYNO moves toward food it can see (E3).

    Every pairing of SYNO's square and a food square at the chosen distance,
    within SYNO's view, is tested, with an empty stomach and the given energy. A move counts if
    it brings SYNO closer to the food, so either route to diagonal food
    counts. Only SYNO's greedy choice is counted.

    :param network: The trained Decision Network.
    :param energy: SYNO's energy, from 0.0 to 1.0.
    :param far: True for food at least 3 steps away, False for food 1 step away.
    :return: The fraction of pairings where SYNO's preferred move approaches the food.
    """
    pairs = []
    for ay in range(GRID_SIZE):
        for ax in range(GRID_SIZE):
            for fy in range(GRID_SIZE):
                for fx in range(GRID_SIZE):
                    distance = abs(ax - fx) + abs(ay - fy)
                    if ((far and distance >= 3) or (not far and distance == 1)) and abs(ax - fx) <= SENSE_RADIUS and abs(ay - fy) <= SENSE_RADIUS:
                        pairs.append(((ax, ay), (fx, fy)))

    # Distance is counted in steps (Manhattan distance), since SYNO cannot
    # move diagonally.
    count = 0
    for agent, food in pairs:
        habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [food])
        body = HomeostaticCore(energy, 0.0)
        values = network.forward(senses(habitat, body))
        name = ACTION_NAMES[values.index(max(values))]
        if name in ACTIONS:
            dx, dy = ACTIONS[name]
            distance = abs(agent[0] - food[0]) + abs(agent[1] - food[1])
            next_distance = abs(agent[0] + dx - food[0]) + abs(agent[1] + dy - food[1])
            if next_distance < distance:
                count += 1
    return count / len(pairs)


def eating_rpe(network: Network, food_present: bool) -> float:
    """
    Measures SYNO's RPE when it eats food it can see (E5).

    SYNO is placed on food in every square, half-full of energy, and eats.
    When food_present is False, the food vanishes after SYNO has seen it, so
    the meal it expected never comes. The step is otherwise identical to a
    step of live. As in the scarce world, no new food appears straight after
    a meal.

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
            body.tick(False)
            reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
            next_expected = max(network.forward(senses(habitat, body)))
            total += reward_prediction_error(reward, expected, next_expected, DISCOUNT, False)
    return total / squares


def cue_rpe(network: Network) -> float:
    """
    Measures SYNO's RPE when food appears in view (E4).

    SYNO is placed in every square of an empty grid, half-full of energy, and
    takes its preferred action. Food then appears, in turn, on every square
    SYNO can see from where it ended up, except its own, so the next state
    contains food that SYNO did not see when it chose.

    :param network: The trained Decision Network.
    :return: SYNO's average RPE over every starting square and food square.
    """
    total = 0.0
    count = 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [])
            body = HomeostaticCore(0.5, 0.0)
            values = network.forward(senses(habitat, body))
            action = values.index(max(values))
            expected = values[action]
            name = ACTION_NAMES[action]
            drive_before = body.drive()
            moved = False
            if name == "eat":
                habitat.eat()
            else:
                moved = habitat.move(name)
            body.tick(moved)
            reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
            # Food appears only after SYNO has acted, so seeing it is news.
            ax, ay = habitat.agent
            for fy in range(GRID_SIZE):
                for fx in range(GRID_SIZE):
                    if abs(fx - ax) <= SENSE_RADIUS and abs(fy - ay) <= SENSE_RADIUS and (fx, fy) != habitat.agent:
                        habitat.food = [(fx, fy)]
                        next_expected = max(network.forward(senses(habitat, body)))
                        total += reward_prediction_error(reward, expected, next_expected, DISCOUNT, False)
                        count += 1
    return total / count


def main():
    """
    Trains SYNO with SEEDS different seeds, measures every signature on each
    network, and prints the results averaged over the seeds.
    """
    level_totals = {level: 0.0 for level in ENERGY_LEVELS}
    stomach_totals = {stomach: 0.0 for stomach in STOMACH_LEVELS}
    near_totals = {energy: 0.0 for energy in APPROACH_ENERGIES}
    far_totals = {energy: 0.0 for energy in APPROACH_ENERGIES}
    food_totals = {checkpoint: 0.0 for checkpoint in CHECKPOINTS}
    cue_totals = {checkpoint: 0.0 for checkpoint in CHECKPOINTS}
    delivered_total = 0.0
    omitted_total = 0.0
    for seed in range(SEEDS):
        network, memory = new_brain(seed)
        trained = 0
        for checkpoint in CHECKPOINTS:
            train(network, memory, checkpoint - trained)
            trained = checkpoint
            food_totals[checkpoint] += eating_rpe(network, True)
            cue_totals[checkpoint] += cue_rpe(network)
        for level in ENERGY_LEVELS:
            level_totals[level] += eating_rate(network, level)
        for stomach in STOMACH_LEVELS:
            stomach_totals[stomach] += eating_rate(network, SATIATION_ENERGY, stomach)
        for energy in APPROACH_ENERGIES:
            near_totals[energy] += approach_rate(network, energy, False)
            far_totals[energy] += approach_rate(network, energy, True)
        delivered_total += eating_rpe(network, True)
        omitted_total += eating_rpe(network, False)
    print(f"E1: Satiation ({SEEDS} seeds, energy {SATIATION_ENERGY})")
    for stomach in STOMACH_LEVELS:
        avg_rate = stomach_totals[stomach] / SEEDS
        print(f"Stomach {stomach:.1f} | Eats: {avg_rate:.2f}")
    print()
    print(f"E2: State-dependent eating ({SEEDS} seeds)")
    for level in ENERGY_LEVELS:
        avg_rate = level_totals[level] / SEEDS
        print(f"Energy {level:.1f} | Eats: {avg_rate:.2f}")
    print()
    print(f"E3: Partial-fullness eating ({SEEDS} seeds)")
    for energy in APPROACH_ENERGIES:
        near_rate = near_totals[energy] / SEEDS
        far_rate = far_totals[energy] / SEEDS
        print(f"Energy {energy:.1f} | Near: {near_rate:.2f} | Far: {far_rate:.2f}")
    print()
    print(f"E4: RPE transfer ({SEEDS} seeds)")
    for checkpoint in CHECKPOINTS:
        food_rpe = food_totals[checkpoint] / SEEDS
        cue_rpe_avg = cue_totals[checkpoint] / SEEDS
        print(f"Lives {checkpoint:4d} | Food RPE: {food_rpe:.3f} | Cue RPE: {cue_rpe_avg:.3f}")
    print()
    print(f"E5: Reward omission ({SEEDS} seeds)")
    print(f"Food delivered | RPE: {delivered_total / SEEDS:.3f}")
    print(f"Food omitted   | RPE: {omitted_total / SEEDS:.3f}")


if __name__ == "__main__":
    main()
