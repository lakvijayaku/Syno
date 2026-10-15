# ── SYNO · experiments/signatures.py ────────────────────
# Measures SYNO's behavior against biological signatures
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random
from multiprocessing import Pool

import experiments.thirst as thirst
import experiments.blink as blink
import experiments.sleep as sleep
import experiments.dreaming as dreaming
from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.memory import MemoryStore
from syno.brain.working_memory import WorkingMemory
from syno.world.habitat import Habitat, ACTIONS
from syno.body.homeostasis import HomeostaticCore, HydratedCore, FatiguedCore, homeostatic_reward
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
# chance: a rate that changes by at least CLEAR_EFFECT, an RPE that dips by at
# least CLEAR_DIP, or a sleep bout that lasts at least CLEAR_REBOUND steps
# longer. Seeds are counted, not averaged, because different
# seeds can learn very different habits.
CLEAR_EFFECT = 0.1
CLEAR_DIP = 0.5
CLEAR_REBOUND = 1

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

# For E7, one need is nearly met (NEED_HIGH) and the other is urgent
# (NEED_LOW), in a SYNO trained in the thirst experiment.
NEED_HIGH = 0.8
NEED_LOW = 0.2

# For E10 and E11, SYNO is nearly full (SLEEP_ENERGY) with no food in the
# world, so fatigue is the only reason to sleep. E10 compares how often a
# rested and a tired SYNO fall asleep. E11 compares how long SYNO sleeps
# after a short time awake and after being kept awake, and follows a sleep
# bout for at most MAX_BOUT steps, so a SYNO that never wakes cannot run
# forever.
SLEEP_ENERGY = 0.8
MAX_BOUT = 30
RESTED_FATIGUE = 0.2
TIRED_FATIGUE = 0.8
SHORT_WAKE_FATIGUE = 0.4
LONG_WAKE_FATIGUE = 0.9

# For E12, a SYNO is judged by its average energy over its last FINAL_LIVES
# lives of training, the same window the experiments print.
FINAL_LIVES = 500


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


def switch_rate(network: Network, energy: float, water: float) -> tuple[float, float]:
    """
    Measures which need SYNO acts on when food and water pull in opposite
    directions (E7).

    For every square and direction, food is placed one step one way and the
    pond one step the opposite way, and SYNO's greedy move is checked. Pairs
    where either would be off the grid are skipped.

    :param network: A Decision Network trained in the thirst experiment.
    :param energy: SYNO's energy, from 0.0 to 1.0.
    :param water: SYNO's water, from 0.0 to 1.0.
    :return: The fraction of pairs where SYNO moves onto the pond, and the
        fraction where it moves onto the food.
    """
    to_water = 0
    to_food = 0
    pairs = 0
    for y in range(thirst.GRID_SIZE):
        for x in range(thirst.GRID_SIZE):
            for dx, dy in ACTIONS.values():
                food = (x + dx, y + dy)
                pond = (x - dx, y - dy)
                habitat = Habitat(thirst.GRID_SIZE, thirst.GRID_SIZE, (x, y), [], [])
                if not habitat.in_bounds(food) or not habitat.in_bounds(pond):
                    continue
                habitat.food.append(food)
                habitat.water.append(pond)
                values = network.forward(thirst.senses(habitat, HydratedCore(energy, 0.0, water)))
                name = thirst.ACTION_NAMES[values.index(max(values))]
                pairs += 1
                if name in ACTIONS:
                    move_dx, move_dy = ACTIONS[name]
                    position = (x + move_dx, y + move_dy)
                    if position == pond:
                        to_water += 1
                    elif position == food:
                        to_food += 1
    return to_water / pairs, to_food / pairs


def delayed_rate(network: Network, seen: bool) -> float:
    """
    Measures whether SYNO heads for food it can no longer see (E8), the
    delayed-response task.

    For every pairing of SYNO's square and a food square at least 2 steps
    away and in view, SYNO's working memory is emptied and then given one
    view, with the food either seen or not. Its eyes then close, and its
    greedy move is checked. Not seeing the food gives the chance level.

    :param network: A Decision Network trained in the blink experiment.
    :param seen: True if SYNO saw the food before its eyes closed.
    :return: The fraction of pairings where SYNO's move brings it closer to
        the food.
    """
    working_memory = WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY)
    count = 0
    pairs = 0
    for y in range(blink.GRID_SIZE):
        for x in range(blink.GRID_SIZE):
            for fy in range(blink.GRID_SIZE):
                for fx in range(blink.GRID_SIZE):
                    distance = abs(x - fx) + abs(y - fy)
                    if distance < 2 or abs(x - fx) > SENSE_RADIUS or abs(y - fy) > SENSE_RADIUS:
                        continue
                    habitat = Habitat(blink.GRID_SIZE, blink.GRID_SIZE, (x, y), [(fx, fy)])
                    body = HomeostaticCore(0.5, 0.0)
                    working_memory.reset()
                    working_memory.update(blink.view(habitat, seen))
                    state = blink.view(habitat, False) + [body.deficit(), body.stomach] + working_memory.contents
                    values = network.forward(state)
                    name = blink.ACTION_NAMES[values.index(max(values))]
                    pairs += 1
                    if name in ACTIONS:
                        dx, dy = ACTIONS[name]
                        next_distance = abs(x + dx - fx) + abs(y + dy - fy)
                        if next_distance < distance:
                            count += 1
    return count / pairs


def value_ramp(network: Network) -> list[float]:
    """
    Measures how SYNO's predicted value changes as food gets closer (E9).

    For every pairing of SYNO's square and a food square in view, SYNO is
    half-hungry with an empty stomach, and the value of its best action is
    averaged by distance to the food. Dopamine ramps up as an animal nears a
    reward, tracking this value.

    :param network: A Decision Network trained in the scarce world.
    :return: The average value at distances 0 to 4, in that order.
    """
    totals = [0.0] * 5
    counts = [0] * 5
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            for fy in range(GRID_SIZE):
                for fx in range(GRID_SIZE):
                    if abs(fx - x) <= SENSE_RADIUS and abs(fy - y) <= SENSE_RADIUS:
                        distance = abs(fx - x) + abs(fy - y)
                        habitat = Habitat(GRID_SIZE, GRID_SIZE, (x, y), [(fx, fy)])
                        body = HomeostaticCore(0.5, 0.0)
                        totals[distance] += max(network.forward(senses(habitat, body)))
                        counts[distance] += 1
    return [total / count for total, count in zip(totals, counts)]


def sleep_rate(network: Network, fatigue: float) -> float:
    """
    Measures how often SYNO chooses to sleep at a given fatigue (E10).

    SYNO is placed on every square with no food in the world and energy
    SLEEP_ENERGY, so fatigue is the only reason to sleep. Animals are more likely to
    fall asleep the longer they have been awake (sleep pressure).

    :param network: A Decision Network trained in the sleep experiment.
    :param fatigue: SYNO's fatigue, from 0.0 to 1.0.
    :return: The fraction of squares where SYNO's preferred action is sleep.
    """
    count = 0
    squares = sleep.GRID_SIZE * sleep.GRID_SIZE
    for y in range(sleep.GRID_SIZE):
        for x in range(sleep.GRID_SIZE):
            habitat = Habitat(sleep.GRID_SIZE, sleep.GRID_SIZE, (x, y), [])
            body = FatiguedCore(SLEEP_ENERGY, 0.0, fatigue)
            values = network.forward(sleep.senses(habitat, body, True))
            if values.index(max(values)) == sleep.ACTIONS.index("sleep"):
                count += 1
    return count / squares


def sleep_bout(network: Network, fatigue: float) -> float:
    """
    Measures how long SYNO sleeps once it falls asleep at a given fatigue
    (E11).

    From every square, with no food in the world and energy SLEEP_ENERGY,
    SYNO decides with its eyes open, then keeps choosing with its eyes closed
    and its body recovering, until it chooses anything but sleep or MAX_BOUT
    steps have passed. Animals kept awake sleep longer afterwards (sleep rebound).

    :param network: A Decision Network trained in the sleep experiment.
    :param fatigue: SYNO's fatigue when it first decides, from 0.0 to 1.0.
    :return: The average number of steps slept in a row, from 0 to
        MAX_BOUT.
    """
    total_bout = 0
    squares = sleep.GRID_SIZE * sleep.GRID_SIZE
    for y in range(sleep.GRID_SIZE):
        for x in range(sleep.GRID_SIZE):
            habitat = Habitat(sleep.GRID_SIZE, sleep.GRID_SIZE, (x, y), [])
            body = FatiguedCore(SLEEP_ENERGY, 0.0, fatigue)
            values = network.forward(sleep.senses(habitat, body, True))
            name = sleep.ACTIONS[values.index(max(values))]
            bout = 0
            # Each step asleep lowers fatigue, so a bout ends when SYNO is
            # rested enough to wake.
            for _ in range(MAX_BOUT):
                if name != "sleep":
                    break
                bout += 1
                body.tick(False, True)
                values = network.forward(sleep.senses(habitat, body, False))
                name = sleep.ACTIONS[values.index(max(values))]
            total_bout += bout
    return total_bout / squares


def trained_energy(seed: int, awake_replays: int, sleep_replays: int) -> float:
    """
    Trains a SYNO in the sleep experiment with the given replay counts and
    measures how well it ends up keeping itself fed (E12).

    Unlike the other probes, E12 is about how well SYNO learns, so the
    measurement is the training itself. Blocking replay during sleep, as
    suppressing hippocampal ripples does in rats, should slow learning.

    :param seed: The random seed, so every condition starts from the same
        network and the same world.
    :param awake_replays: How many memories SYNO replays after a step awake.
    :param sleep_replays: How many memories SYNO replays after a step asleep.
    :return: SYNO's average energy over its last FINAL_LIVES lives.
    """
    random.seed(seed)
    inputs = sleep.WINDOW + 3
    network = Network([
        make_layer(sleep.HIDDEN_NEURONS, inputs),
        make_layer(len(sleep.ACTIONS), sleep.HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    energies = []
    for _ in range(sleep.LIVES):
        energy, _ = sleep.live(network, memory, awake_replays, sleep_replays)
        energies.append(energy)
    final = energies[-FINAL_LIVES:]
    return sum(final) / len(final)


def measure(seed: int) -> dict:
    """
    Trains six SYNOs from the same seed, in the scarce world, the thirst
    experiment, the blink experiment, the sleep experiment, and the dreaming
    experiment with its sleep replay kept and blocked, and measures every
    signature.

    Each seed runs in its own process, so the results are returned rather
    than added to shared totals.

    :param seed: The random seed for this SYNO.
    :return: The results by signature: "food" and "cue" (E4, one value per
        checkpoint), "e1", "e2", "near" and "far" (E3), "e5", "thirsty"
        and "hungry" (E7), "e8", "e9" (one value per distance), "e10", and
        "e11", and "e12" (sleep replay kept, then blocked).
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
    results["e9"] = value_ramp(network)

    # E7 needs a SYNO with two needs, trained in the thirst experiment from
    # the same seed.
    random.seed(seed)
    inputs = 2 * (2 * SENSE_RADIUS + 1) ** 2 + 3
    network = Network([
        make_layer(thirst.HIDDEN_NEURONS, inputs),
        make_layer(len(thirst.ACTION_NAMES), thirst.HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    for _ in range(thirst.LIVES):
        thirst.live(network, memory)
    results["thirsty"] = switch_rate(network, NEED_HIGH, NEED_LOW)
    results["hungry"] = switch_rate(network, NEED_LOW, NEED_HIGH)

    # E8 needs a SYNO with working memory, trained in the blink experiment
    # from the same seed.
    random.seed(seed)
    inputs = 2 * blink.WINDOW + 2
    network = Network([
        make_layer(blink.HIDDEN_NEURONS, inputs),
        make_layer(len(blink.ACTION_NAMES), blink.HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    working_memory = WorkingMemory(blink.WINDOW, blink.MEMORY_DECAY)
    for _ in range(blink.LIVES):
        blink.live(network, memory, working_memory)
    results["e8"] = [delayed_rate(network, True), delayed_rate(network, False)]

    # E10 and E11 need a SYNO that gets tired and can sleep, trained in the
    # sleep experiment from the same seed.
    random.seed(seed)
    inputs = sleep.WINDOW + 3
    network = Network([
        make_layer(sleep.HIDDEN_NEURONS, inputs),
        make_layer(len(sleep.ACTIONS), sleep.HIDDEN_NEURONS, "linear")
    ])
    memory = MemoryStore(MEMORY_CAPACITY)
    for _ in range(sleep.LIVES):
        sleep.live(network, memory)
    results["e10"] = [sleep_rate(network, RESTED_FATIGUE), sleep_rate(network, TIRED_FATIGUE)]
    results["e11"] = [sleep_bout(network, SHORT_WAKE_FATIGUE), sleep_bout(network, LONG_WAKE_FATIGUE)]
    # E12 compares the dreaming SYNO with the same SYNO whose sleep replay
    # is blocked. Both sleep as much as they choose; only replay differs.
    results["e12"] = [trained_energy(seed, dreaming.AWAKE_REPLAYS, dreaming.SLEEP_REPLAYS),
                      trained_energy(seed, 0, 0)]
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
    print()
    print(f"E7: Need switching ({SEEDS} seeds)")
    thirsty_water = spread([r["thirsty"][0] for r in results], 2)
    thirsty_food = spread([r["thirsty"][1] for r in results], 2)
    hungry_water = spread([r["hungry"][0] for r in results], 2)
    hungry_food = spread([r["hungry"][1] for r in results], 2)
    print(f"Thirsty | Water: {thirsty_water} | Food: {thirsty_food}")
    print(f"Hungry  | Water: {hungry_water} | Food: {hungry_food}")
    n = sum(
        r["thirsty"][0] >= r["thirsty"][1] + CLEAR_EFFECT
        and r["hungry"][1] >= r["hungry"][0] + CLEAR_EFFECT
        for r in results
    )
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E8: Delayed response ({SEEDS} seeds)")
    seen_rate = spread([r["e8"][0] for r in results], 2)
    not_seen_rate = spread([r["e8"][1] for r in results], 2)
    print(f"Seen     | Approaches: {seen_rate}")
    print(f"Not seen | Approaches: {not_seen_rate}")
    n = sum(r["e8"][0] >= r["e8"][1] + CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E9: Dopamine ramping ({SEEDS} seeds)")
    for d in range(4, -1, -1):
        value = spread([r["e9"][d] for r in results], 3)
        print(f"Distance {d} | Value: {value}")
    n = sum(all(r["e9"][d - 1] > r["e9"][d] for d in range(1, 5)) for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E10: Sleep pressure ({SEEDS} seeds)")
    rested_rate = spread([r["e10"][0] for r in results], 2)
    tired_rate = spread([r["e10"][1] for r in results], 2)
    print(f"Fatigue {RESTED_FATIGUE:.1f} | Sleeps: {rested_rate}")
    print(f"Fatigue {TIRED_FATIGUE:.1f} | Sleeps: {tired_rate}")
    n = sum(r["e10"][1] >= r["e10"][0] + CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E11: Sleep rebound ({SEEDS} seeds)")
    rested_bout = spread([r["e11"][0] for r in results], 2)
    tired_bout = spread([r["e11"][1] for r in results], 2)
    print(f"Fatigue {SHORT_WAKE_FATIGUE:.1f} | Bout: {rested_bout}")
    print(f"Fatigue {LONG_WAKE_FATIGUE:.1f} | Bout: {tired_bout}")
    n = sum(r["e11"][1] >= r["e11"][0] + CLEAR_REBOUND for r in results)
    print(f"Seeds: {n}/{SEEDS}")
    print()
    print(f"E12: Sleep consolidation ({SEEDS} seeds)")
    sleep_replay_energy = spread([r["e12"][0] for r in results], 3)
    blocked_energy = spread([r["e12"][1] for r in results], 3)
    print(f"Sleep replay | Energy: {sleep_replay_energy}")
    print(f"Blocked      | Energy: {blocked_energy}")
    n = sum(r["e12"][0] >= r["e12"][1] + CLEAR_EFFECT for r in results)
    print(f"Seeds: {n}/{SEEDS}")


if __name__ == "__main__":
    main()
