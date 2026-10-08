# ── SYNO · experiments/consolidation.py ─────────────────
# SYNO replays its most surprising memories most often
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import os
import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.tools.recorder import Recorder
from syno.tools.plot import line_chart
from syno.tools.stats import moving_average
from syno.brain.memory import MemoryStore

# Shared settings and helpers are reused from the earlier experiments, so the
# only differences from the replay experiment are defined in this file.
from experiments.hunger import (
    ACTION_NAMES,
    DISCOUNT,
    FOOD_AMOUNT,
    GRID_SIZE,
    HIDDEN_NEURONS,
    LEARNING_RATE,
    LIFE_STEPS,
    LIVES,
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

# Added to every priority, so even a fully learned memory can still be
# recalled. With a priority of 0, a memory could never be revisited.
PRIORITY_FLOOR = 0.01

def live(network: Network, memory: MemoryStore) -> float:
    """
    Runs one life of LIFE_STEPS steps, learning after every step and then
    replaying memories in proportion to how surprising they are.

    Identical to replay.live, except that each memory's priority is the size
    of its RPE, and is updated every time the memory is replayed.

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

    for _ in range(LIFE_STEPS):
        action = choose_action(network.forward(state), EPSILON)
        name = ACTION_NAMES[action]

        drive_before = body.drive()
        moved = False

        if name == "eat":
            if habitat.eat():
                body.eat(FOOD_AMOUNT)
                spawn_food(habitat)
        else:
            moved = habitat.move(name)

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
    Trains SYNO over many lives with surprise-weighted replay, prints its
    average energy every 500 lives, and saves every life's energy to
    runs/consolidation.csv and a chart to runs/consolidation.svg.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    energies = []
    recorder = Recorder(["life", "energy"])
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(MEMORY_CAPACITY)

    for life in range(1, LIVES + 1):
        energy = live(network, memory)
        energies.append(energy)
        recorder.record({"life": life, "energy": energy})

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f}")

    os.makedirs("runs", exist_ok=True)
    recorder.save("runs/consolidation.csv")
    line_chart(
        moving_average(energies, 100),
        "runs/consolidation.svg",
        "With surprise-weighted replay (moving average of 100)",
    )
    print("Saved runs/consolidation.csv and runs/consolidation.svg")

if __name__ == "__main__":
    main()
