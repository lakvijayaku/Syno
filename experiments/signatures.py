# ── SYNO · experiments/signatures.py ────────────────────
# Measures SYNO's behavior against biological signatures
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random
from multiprocessing import Pool

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
SEEDS = 10

# A seed shows a signature only when its own effect is clearly larger than
# chance: a rate that changes by at least CLEAR_EFFECT, or an RPE that dips by
# at least CLEAR_DIP. Seeds are counted, not averaged, because different
# seeds can learn very different habits.
CLEAR_EFFECT = 0.1
CLEAR_DIP = 0.5

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


def measure(seed: int) -> dict:
    """
    Trains one SYNO and measures every signature on it.

    Each seed runs in its own process, so the results are returned rather
    than added to shared totals.

    :param seed: The random seed for this SYNO.
    :return: The results by signature: "food" and "cue" (E4, one value per
        checkpoint), "e1", "e2", "near" and "far" (E3), and "e5".
    """
    network, memory = new_brain(seed)
    results = {"food": [], "cue": []}
    trained = 0
    for checkpoint in CHECKPOINTS:
        train(network, memory, checkpoint - trained)
        trained = checkpoint
        results["food"].append(eating_rpe(network, True))
        results["cue"].append(cue_rpe(network))
    results["e1"] = [eating_rate(network, SATIATION_ENERGY, s) for s in STOMACH_LEVELS]
    results["e2"] = [eating_rate(network, e) for e in ENERGY_LEVELS]
    results["near"] = [approach_rate(network, e, False) for e in APPROACH_ENERGIES]
    results["far"] = [approach_rate(network, e, True) for e in APPROACH_ENERGIES]
    results["e5"] = [eating_rpe(network, True), eating_rpe(network, False)]
    return results


def spread(values: list[float], digits: int) -> str:
    """
    Describes results from several seeds as their average, lowest, and highest.

    :param values: One result per seed.
    :param digits: How many decimal places to show.
    :return: Text such as "0.98 [0.88, 1.00]".
    """
    return f"{sum(values) / len(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def main():
    """
    Trains SYNO with SEEDS different seeds in parallel, measures every
    signature on each network, and prints each result as its average and
    range over the seeds, followed by how many seeds show the signature.
    """
    # One process per CPU core. pool.map returns the results in seed order,
    # so the output is the same as measuring the seeds one after another.
    with Pool() as pool:
        results = pool.map(measure, range(SEEDS))

    print(f"E1: Satiation ({SEEDS} seeds, energy {SATIATION_ENERGY})")
    for i, stomach in enumerate(STOMACH_LEVELS):
        avg_rate = spread([r["e1"][i] for r in results], 2)
        print(f"Stomach {stomach:.1f} | Eats: {avg_rate}")
    n = sum(r["e1"][-1] <= r["e1"][0] - CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E2: State-dependent eating ({SEEDS} seeds)")
    for i, level in enumerate(ENERGY_LEVELS):
        avg_rate = spread([r["e2"][i] for r in results], 2)
        print(f"Energy {level:.1f} | Eats: {avg_rate}")
    n = sum(r["e2"][-1] <= r["e2"][0] - CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E3: Partial-fullness eating ({SEEDS} seeds)")
    for i, energy in enumerate(APPROACH_ENERGIES):
        near_rate = spread([r["near"][i] for r in results], 2)
        far_rate = spread([r["far"][i] for r in results], 2)
        print(f"Energy {energy:.1f} | Near: {near_rate} | Far: {far_rate}")
    n = sum(r["far"][2] <= r["near"][2] - CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E4: RPE transfer ({SEEDS} seeds)")
    for i, checkpoint in enumerate(CHECKPOINTS):
        food_rpe = spread([r["food"][i] for r in results], 3)
        cue_rpe_avg = spread([r["cue"][i] for r in results], 3)
        print(f"Lives {checkpoint:4d} | Food RPE: {food_rpe} | Cue RPE: {cue_rpe_avg}")
    n = sum(r["cue"][-1] > 0 and r["food"][-1] < r["food"][0] for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E5: Reward omission ({SEEDS} seeds)")
    delivered_rpe = spread([r["e5"][0] for r in results], 3)
    omitted_rpe = spread([r["e5"][1] for r in results], 3)
    print(f"Food delivered | RPE: {delivered_rpe}")
    print(f"Food omitted   | RPE: {omitted_rpe}")
    n = sum(r["e5"][1] <= r["e5"][0] - CLEAR_DIP for r in results)
    print(f"Seeds: {n}/{SEEDS}")


if __name__ == "__main__":
    main()
