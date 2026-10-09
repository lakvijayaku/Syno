# ── SYNO · experiments/scarcity.py ──────────────────────
# SYNO forages in a larger world where food takes time to regrow
# Laksheth Vijayakumar · 2026-10-08 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.brain.memory import MemoryStore

# Shared settings and helpers are reused from the earlier experiments, so the
# only differences from the consolidation experiment are defined in this file.
from experiments.hunger import (
    ACTION_NAMES,
    DISCOUNT,
    FOOD_AMOUNT,
    HIDDEN_NEURONS,
    LEARNING_RATE,
    LIFE_STEPS,
    REWARD_SCALE,
    SENSE_RADIUS,
    senses,
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

from experiments.consolidation import PRIORITY_FLOOR

# The world is larger than SYNO's view, so food can be out of sight, and food
# takes REGROW_DELAY steps to reappear after a meal, so it is scarce. Seeing
# food therefore predicts a meal, as a cue does for an animal.
GRID_SIZE = 5
REGROW_DELAY = 10

# SYNO learns more slowly in this world, and its energy levels off by about
# 3000 lives.
LIVES = 3000


def live(network: Network, memory: MemoryStore) -> float:
    """
    Runs one life of LIFE_STEPS steps in the larger world, learning after
    every step and then replaying memories in proportion to how surprising
    they are.

    Identical to consolidation.live, except for the grid size and that eaten
    food regrows after REGROW_DELAY steps instead of at once.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :return: SYNO's average energy over the life.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = HomeostaticCore(random.uniform(0.3, 0.9), 0.0)
    state = senses(habitat, body)
    total_energy = 0.0
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

        # Counted down every step, including the step of the meal, so new food
        # appears REGROW_DELAY steps after the meal.
        if regrow_timer > 0:
            regrow_timer -= 1
            if regrow_timer == 0:
                spawn_food(habitat)

        body.tick(moved)

        reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
        novelty.tick()
        reward += novelty.signal(habitat.agent)

        next_state = senses(habitat, body)

        # The size of the RPE measures how surprising the step was, and so how
        # much it still has to teach.
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
            # Re-measure the surprise after replaying. As a lesson is learned,
            # its RPE shrinks and the memory is recalled less often.
            memory.set_priority(index, abs(old_rpe) + PRIORITY_FLOOR)

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS


def main():
    """
    Trains SYNO over many lives in the larger world and prints its average
    energy every 500 lives.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    energies = []
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(MEMORY_CAPACITY)

    for life in range(1, LIVES + 1):
        energy = live(network, memory)
        energies.append(energy)

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f}")


if __name__ == "__main__":
    main()
