# ── SYNO · experiments/blink.py ─────────────────────────
# SYNO forages with its eyes closed half the time, using working memory
# Laksheth Vijayakumar · 2026-10-11 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.brain.working_memory import WorkingMemory
from syno.brain.memory import MemoryStore

# Shared settings are reused from the earlier experiments, so the only
# differences from the scarcity experiment are defined in this file.
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

from experiments.scarcity import GRID_SIZE, REGROW_DELAY
from experiments.consolidation import PRIORITY_FLOOR

# On each step, SYNO's eyes are closed with this chance. It then still feels
# walls and any food under it, but cannot see food further away.
DARK_CHANCE = 0.5

# Working memory keeps half of what it held at every step, so a view fades
# to a quarter after two steps and an eighth after three.
MEMORY_DECAY = 0.5

# SYNO senses its view, its body, and its remembered view, so it needs a
# larger brain than in the scarcity experiment.
HIDDEN_NEURONS = 16
LIVES = 3000

# The number of squares in SYNO's view, and so in its working memory.
WINDOW = (2 * SENSE_RADIUS + 1) ** 2


def view(habitat: Habitat, visible: bool) -> list[float]:
    """
    What SYNO sees this step, with its eyes open or closed.

    :param habitat: The habitat SYNO is in.
    :param visible: True if SYNO's eyes are open.
    :return: The view from Habitat.sense when visible. Otherwise the same
        window with only walls (-1.0) and any food under SYNO (1.0), which it
        can still feel.
    """
    seen = habitat.sense(SENSE_RADIUS)
    if visible:
        return seen
    dark = [-1.0 if value == -1.0 else 0.0 for value in seen]
    if habitat.agent in habitat.food:
        dark[WINDOW // 2] = 1.0
    return dark


def live(network: Network, memory: MemoryStore, working_memory: WorkingMemory) -> float:
    """
    Runs one life of LIFE_STEPS steps, with SYNO's eyes closed at random.

    Identical to scarcity.live, except that SYNO's view can be dark, and
    SYNO also senses a working memory of what it saw before. The memory is
    emptied at the start of each life.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :param working_memory: SYNO's working memory, shared across lives.
    :return: SYNO's average energy over the life.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = HomeostaticCore(random.uniform(0.3, 0.9), 0.0)
    working_memory.reset()
    seen = view(habitat, True)
    state = seen + [body.deficit(), body.stomach] + working_memory.contents
    total_energy = 0.0
    regrow_timer = 0

    for _ in range(LIFE_STEPS):
        action = choose_action(network.forward(state), EPSILON)
        # The decision used the memory from before this step. Now what SYNO
        # just saw is added, so the next decision can use it in the dark.
        working_memory.update(seen)
        name = ACTION_NAMES[action]

        drive_before = body.drive()
        moved = False

        if name == "eat":
            if habitat.eat():
                body.eat(FOOD_AMOUNT)
                regrow_timer = REGROW_DELAY
        else:
            moved = habitat.move(name)

        if regrow_timer > 0:
            regrow_timer -= 1
            if regrow_timer == 0:
                spawn_food(habitat)

        # Whether SYNO's eyes are open for the next view.
        visible = random.random() >= DARK_CHANCE

        body.tick(moved)

        reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
        novelty.tick()
        reward += novelty.signal(habitat.agent)

        seen = view(habitat, visible)
        next_state = seen + [body.deficit(), body.stomach] + working_memory.contents

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

    return total_energy / LIFE_STEPS


def main():
    """
    Trains SYNO over many lives with working memory and prints its average
    energy every 500 lives.
    """
    random.seed(0)

    # The view, the energy deficit and stomach fill, and the remembered view.
    inputs = 2 * WINDOW + 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    energies = []
    # Both memories are created once. The Memory Store keeps experiences
    # across lives; working memory is emptied at the start of each life.
    memory = MemoryStore(MEMORY_CAPACITY)
    working_memory = WorkingMemory(WINDOW, MEMORY_DECAY)

    for life in range(1, LIVES + 1):
        energy = live(network, memory, working_memory)
        energies.append(energy)

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f}")


if __name__ == "__main__":
    main()
