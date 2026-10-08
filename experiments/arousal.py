# ── SYNO · experiments/arousal.py ───────────────────────
# An arousal hormone speeds up learning after the world changes
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
from syno.brain.memory import MemoryStore
from syno.tools.recorder import Recorder
from syno.body.hormones import Hormone
from syno.brain.novelty import NoveltySystem

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

from experiments.replay import (
    MEMORY_CAPACITY,
    REPLAYS_PER_STEP,
)

from experiments.consolidation import PRIORITY_FLOOR

from experiments.curiosity import (
    EPSILON,
    HABITUATION,
    NOVELTY_SCALE,
    RECOVERY,
)

# SYNO lives in the small world first, then moves to a larger one, where
# food is often out of sight and its learned habits stop working.
SMALL_GRID = 3
LARGE_GRID = 5
MOVE_LIFE = 1000
LIVES = 2500

# Arousal is a slow hormone fed by surprise (the size of the RPE). It sets
# the learning rate: LR_MIN when nothing is surprising, rising to LR_MAX once
# arousal reaches AROUSAL_SCALE. Based on the Pearce-Hall model, in which
# animals learn faster from situations that have recently been surprising.
AROUSAL_RATE = 0.01
LR_MIN = 0.05
LR_MAX = 0.2
AROUSAL_SCALE = 0.3

def live(network: Network, memory: MemoryStore, grid_size: int, arousal: Hormone | None) -> float:
    """
    Runs one life of LIFE_STEPS steps on a grid of the given size.

    Identical to consolidation.live, except for the grid size and the
    learning rate, which is fixed when arousal is None and otherwise set by
    the arousal hormone at the start of every step.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :param grid_size: The width and height of the habitat.
    :param arousal: The arousal hormone, shared across lives, or None for a
        fixed learning rate.
    :return: SYNO's average energy over the life.
    """
    agent = (random.randrange(grid_size), random.randrange(grid_size))
    habitat = Habitat(grid_size, grid_size, agent, [])
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = HomeostaticCore(random.uniform(0.3, 0.9), 0.0)
    state = senses(habitat, body)
    total_energy = 0.0

    for _ in range(LIFE_STEPS):
        # Higher arousal means faster learning. min caps the rate at LR_MAX.
        if arousal is None:
            current_lr = LEARNING_RATE
        else:
            current_lr = LR_MIN + (LR_MAX - LR_MIN) * min(1.0, arousal.level / AROUSAL_SCALE)

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

        rpe = learn(network, state, action, reward, next_state, False, DISCOUNT, current_lr)

        # Surprise feeds the arousal hormone. Because the hormone is slow,
        # only a sustained run of surprises raises the learning rate much.
        if arousal is not None:
            arousal.update(abs(rpe))

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
                current_lr,
            )
            memory.set_priority(index, abs(old_rpe) + PRIORITY_FLOOR)

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS

def run(use_arousal: bool) -> list[float]:
    """
    Trains SYNO from scratch, moving it to the large grid after MOVE_LIFE.

    Both versions use the same seed, so the only difference between them is
    the arousal hormone.

    :param use_arousal: True to set the learning rate with an arousal hormone.
    :return: SYNO's average energy for every life.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    memory = MemoryStore(MEMORY_CAPACITY)
    arousal = Hormone(AROUSAL_RATE) if use_arousal else None
    energies = []

    for life in range(1, LIVES + 1):
        grid_size = SMALL_GRID if life <= MOVE_LIFE else LARGE_GRID
        energy = live(network, memory, grid_size, arousal)
        energies.append(energy)

    return energies

def main():
    """
    Runs SYNO with a fixed learning rate and with arousal, prints their
    average energy side by side every 250 lives, and saves every life to
    runs/arousal.csv.
    """
    fixed = run(False)
    aroused = run(True)

    recorder = Recorder(["life", "fixed", "arousal"])

    for i in range(LIVES):
        recorder.record({
            "life": i + 1,
            "fixed": fixed[i],
            "arousal": aroused[i]
        })

    for block in range(0, LIVES, 250):
        start = block + 1
        end = block + 250
        fixed_avg = sum(fixed[block:end]) / 250
        aroused_avg = sum(aroused[block:end]) / 250
        print(f"Lives {start:4d}-{end:4d} | Fixed: {fixed_avg:.3f} | Arousal: {aroused_avg:.3f}")

        if end == MOVE_LIFE:
            print("--- SYNO moves to the 5x5 world ---")

    os.makedirs("runs", exist_ok=True)
    recorder.save("runs/arousal.csv")

if __name__ == "__main__":
    main()
