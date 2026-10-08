# ── SYNO · experiments/replay.py ────────────────────────
# SYNO learns faster by replaying its memories
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
# only differences from the curiosity experiment are defined in this file.
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

# SYNO remembers its last 1000 steps (10 lives), and after every real step
# re-learns from 2 of them chosen at random.
MEMORY_CAPACITY = 1000
REPLAYS_PER_STEP = 2

def live(network: Network, memory: MemoryStore) -> float:
    """
    Runs one life of LIFE_STEPS steps, learning after every step and then
    replaying memories.

    Identical to curiosity.live, except that every experience is stored and
    REPLAYS_PER_STEP past experiences are learned from again after each step.

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

        learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)

        # Remember this step, then replay random past steps. A rare, valuable
        # experience is learned from many times instead of once, and mixing
        # old lessons in with new ones reduces forgetting.
        memory.store((state, action, reward, next_state))
        for _ in range(REPLAYS_PER_STEP):
            old_state, old_action, old_reward, old_next_state = memory.sample()
            learn(
                network,
                old_state,
                old_action,
                old_reward,
                old_next_state,
                False,
                DISCOUNT,
                LEARNING_RATE,
            )

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS

def main():
    """
    Trains SYNO over many lives with memory replay, prints its average energy
    every 500 lives, and saves every life's energy to runs/replay.csv and a
    chart to runs/replay.svg.
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
    recorder.save("runs/replay.csv")
    line_chart(
        moving_average(energies, 100),
        "runs/replay.svg",
        "With memory replay (moving average of 100)",
    )
    print("Saved runs/replay.csv and runs/replay.svg")

if __name__ == "__main__":
    main()
