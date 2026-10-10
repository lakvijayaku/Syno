# ── SYNO · experiments/thirst.py ────────────────────────
# SYNO keeps itself fed and watered, and dies if either runs out
# Laksheth Vijayakumar · 2026-10-09 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import HydratedCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.brain.memory import MemoryStore
from experiments.consolidation import PRIORITY_FLOOR

# Shared settings are reused from the earlier experiments, so the only
# differences from the scarcity experiment are defined in this file.
from experiments.hunger import (
    DISCOUNT,
    FOOD_AMOUNT,
    LEARNING_RATE,
    LIFE_STEPS,
    REWARD_SCALE,
    SENSE_RADIUS,
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

from experiments.scarcity import GRID_SIZE, REGROW_DELAY

# SYNO can now drink as well as eat. The world has one pond, which is never
# used up, and food that regrows as in the scarcity experiment.
ACTION_NAMES = ["up", "down", "left", "right", "eat", "drink"]
DRINK_AMOUNT = 0.3

# Two needs mean twice the senses and a harder choice. With 8 hidden neurons,
# SYNO did not learn to choose between food and water; with 16 it did.
HIDDEN_NEURONS = 16
LIVES = 3000


def free_square(habitat: Habitat) -> tuple[int, int]:
    """
    Picks a random square with nothing on it. Unlike hunger.spawn_food, it
    also avoids water, so food never appears on the pond.

    :param habitat: The habitat to search.
    :return: The (x, y) position of a free square.
    """
    while True:
        position = (random.randrange(habitat.width), random.randrange(habitat.height))
        if position != habitat.agent and position not in habitat.food and position not in habitat.water:
            return position


def senses(habitat: Habitat, body: HydratedCore) -> list[float]:
    """
    Combines what SYNO sees with how its body feels.

    :param habitat: The habitat SYNO is in.
    :param body: SYNO's body.
    :return: The food window, the water window, then the energy deficit,
        stomach fill, and water deficit.
    """
    return habitat.sense(SENSE_RADIUS) + habitat.sense_water(SENSE_RADIUS) + [body.deficit(), body.stomach, body.water_deficit()]


def live(network: Network, memory: MemoryStore) -> int:
    """
    Runs one life, which ends after LIFE_STEPS steps or when SYNO dies of
    hunger or thirst.

    Identical to scarcity.live, except for the pond, the drink action, the
    two-need body, and death. When SYNO dies, the step is learned as final,
    with no future value after it.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :return: How many steps SYNO lived.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    habitat.water.append(free_square(habitat))
    habitat.food.append(free_square(habitat))

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = HydratedCore(random.uniform(0.3, 0.9), 0.0, random.uniform(0.3, 0.9))
    state = senses(habitat, body)
    steps = 0
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
        elif name == "drink":
            if habitat.drink():
                body.drink(DRINK_AMOUNT)
        else:
            moved = habitat.move(name)

        if regrow_timer > 0:
            regrow_timer -= 1
            if regrow_timer == 0:
                habitat.food.append(free_square(habitat))

        body.tick(moved)
        # Death ends the life, so the step is learned with done=True.
        dead = body.is_dead()

        reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
        novelty.tick()
        reward += novelty.signal(habitat.agent)

        next_state = senses(habitat, body)

        rpe = learn(network, state, action, reward, next_state, dead, DISCOUNT, LEARNING_RATE)
        memory.store((state, action, reward, next_state, dead), abs(rpe) + PRIORITY_FLOOR)

        for _ in range(REPLAYS_PER_STEP):
            index = memory.sample_by_priority()
            old_state, old_action, old_reward, old_next_state, old_dead = memory.experiences[index]
            old_rpe = learn(
                network,
                old_state,
                old_action,
                old_reward,
                old_next_state,
                old_dead,
                DISCOUNT,
                LEARNING_RATE,
            )
            memory.set_priority(index, abs(old_rpe) + PRIORITY_FLOOR)

        state = next_state
        steps += 1
        if dead:
            break

    return steps


def main():
    """
    Trains SYNO over many lives and prints its average lifespan every 500
    lives. A lifespan of LIFE_STEPS means SYNO survived the whole life.
    """
    random.seed(0)

    # Two windows, one for food and one for water, plus three body values.
    inputs = 2 * (2 * SENSE_RADIUS + 1) ** 2 + 3

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    lifespans = []
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(MEMORY_CAPACITY)

    for life in range(1, LIVES + 1):
        lifespan = live(network, memory)
        lifespans.append(lifespan)

        if life % 500 == 0:
            avg = sum(lifespans[-500:]) / 500
            print(f"Life {life:4d} | Avg Lifespan: {avg:.1f} steps")


if __name__ == "__main__":
    main()
