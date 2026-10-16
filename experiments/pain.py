# ── SYNO · experiments/pain.py ──────────────────────────
# SYNO forages in a world with thorns
# Laksheth Vijayakumar · 2026-10-15 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import InjuredCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.brain.memory import MemoryStore
from experiments.consolidation import PRIORITY_FLOOR

# Shared settings and helpers are reused from the earlier experiments, so the
# only differences from the scarcity experiment are defined in this file.
from experiments.hunger import (
    ACTION_NAMES,
    DISCOUNT,
    FOOD_AMOUNT,
    LEARNING_RATE,
    LIFE_STEPS,
    REWARD_SCALE,
    SENSE_RADIUS,
    spawn_food,
)

from experiments.curiosity import (
    EPSILON,
    NOVELTY_SCALE,
    HABITUATION,
    RECOVERY,
)

from experiments.replay import (
    MEMORY_CAPACITY,
    REPLAYS_PER_STEP,
)

from experiments.scarcity import GRID_SIZE, REGROW_DELAY, LIVES

# Every life has THORNS thorns on random squares. Stepping onto one hurts
# SYNO by THORN_DAMAGE, which takes 20 steps to heal. With 0.5, SYNO starved;
# with 0.3, hunger overrode pain in only some seeds; 0.4 gives both.
THORNS = 2
THORN_DAMAGE = 0.4

# SYNO senses a second window, for thorns, as in the thirst experiment, so it
# needs a larger brain.
HIDDEN_NEURONS = 16


def free_square(habitat: Habitat) -> tuple[int, int]:
    """
    Picks a random square with nothing on it, so two thorns never share a
    square and no thorn starts under SYNO.

    :param habitat: The habitat to search.
    :return: The (x, y) position of a free square.
    """
    while True:
        position = (random.randrange(habitat.width), random.randrange(habitat.height))
        if position != habitat.agent and position not in habitat.food and position not in habitat.hazards:
            return position


def senses(habitat: Habitat, body: InjuredCore) -> list[float]:
    """
    Combines what SYNO sees with how its body feels, including how hurt it
    is.

    :param habitat: The habitat SYNO is in.
    :param body: SYNO's body.
    :return: The food window, the thorn window, then the energy deficit,
        stomach fill, and injury.
    """
    return habitat.sense(SENSE_RADIUS) + habitat.sense_hazard(SENSE_RADIUS) + [body.deficit(), body.stomach, body.injury]


def live(network: Network, memory: MemoryStore) -> tuple[float, int]:
    """
    Runs one life of LIFE_STEPS steps in the scarce world with thorns.

    Identical to scarcity.live, except that thorns are placed at the start of
    the life, SYNO senses them and its injury, and stepping onto a thorn
    hurts it. Food can grow on a thorn, so hunger and pain can compete.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :return: SYNO's average energy over the life, and how many times it
        stepped onto a thorn.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    # Thorns are placed before food. spawn_food does not avoid thorns, so food
    # can grow on one, like berries in a thornbush.
    for _ in range(THORNS):
        habitat.hazards.append(free_square(habitat))
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = InjuredCore(random.uniform(0.3, 0.9), 0.0)
    state = senses(habitat, body)
    total_energy = 0.0
    thorn_hits = 0
    regrow_timer = 0

    for _ in range(LIFE_STEPS):
        action = choose_action(network.forward(state), EPSILON)
        name = ACTION_NAMES[action]

        drive_before = body.drive()
        moved = False

        if name == "eat":
            if habitat.eat():
                body.eat(FOOD_AMOUNT)
                regrow_timer = REGROW_DELAY
        else:
            moved = habitat.move(name)
            # Only stepping onto a thorn hurts, not standing on one or
            # bumping into a wall. Whether to step there is learned.
            if moved and habitat.agent in habitat.hazards:
                body.hurt(THORN_DAMAGE)
                thorn_hits += 1

        if regrow_timer > 0:
            regrow_timer -= 1
            if regrow_timer == 0:
                spawn_food(habitat)

        body.tick(moved)

        reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
        novelty.tick()
        reward += novelty.signal(habitat.agent)

        next_state = senses(habitat, body)

        rpe = learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)
        memory.store((state, action, reward, next_state), abs(rpe) + PRIORITY_FLOOR)

        for _ in range(REPLAYS_PER_STEP):
            index = memory.sample_by_priority()
            old_state, old_action, old_reward, old_next_state = memory.experiences[index]
            old_rpe = learn(
                network,
                old_state,
                old_action,
                old_reward,
                old_next_state,
                False,
                DISCOUNT,
                LEARNING_RATE,
            )
            memory.set_priority(index, abs(old_rpe) + PRIORITY_FLOOR)

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS, thorn_hits


def main():
    """
    Trains SYNO over many lives in the world with thorns and prints its
    average energy and thorn hits per life every 500 lives.
    """
    random.seed(0)

    # Two windows, one for food and one for thorns, plus three body values.
    inputs = 2 * (2 * SENSE_RADIUS + 1) ** 2 + 3

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    energies = []
    thorn_hits = []
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(MEMORY_CAPACITY)

    for life in range(1, LIVES + 1):
        energy, hits = live(network, memory)
        energies.append(energy)
        thorn_hits.append(hits)

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            avg_hits = sum(thorn_hits[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f} | Thorn Hits: {avg_hits:.2f}")


if __name__ == "__main__":
    main()
