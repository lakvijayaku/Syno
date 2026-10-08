# ── SYNO · experiments/curiosity.py ─────────────────────
# Trains SYNO with combined homeostatic rewards and intrinsic novelty bonuses
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

# Shared settings and helpers are reused from the hunger experiment, so the
# only differences between the two experiments are defined in this file.
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

# Six times less random exploration than the hunger experiment. Curiosity
# does most of the exploring. With no randomness at all, SYNO did not learn.
EPSILON = 0.05

# Novelty System settings: the reward for a new square, how quickly squares
# become familiar, and how much familiarity remains after each tick.
NOVELTY_SCALE = 0.2
HABITUATION = 0.5
RECOVERY = 0.95

def live(network: Network) -> float:
    """
    Runs one life of LIFE_STEPS steps, learning after every step.

    Identical to hunger.live, except that a Novelty System adds a curiosity
    reward and the exploration rate is lower.

    :param network: The Decision Network, with one output per action.
    :return: SYNO's average energy over the life.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

    # Each life starts with a fresh memory. SYNO has seen its starting
    # square, so that square is familiar from the start.
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
        # Curiosity is added to the same reward as hunger, so both drives
        # share one dopamine signal.
        novelty.tick()
        reward += novelty.signal(habitat.agent)

        next_state = senses(habitat, body)

        learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS

def main():
    """
    Trains SYNO over many lives, prints its average energy every 500 lives
    for comparison with the hunger experiment, and saves every life's energy
    to runs/curiosity.csv and a chart to runs/curiosity.svg.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    # Every life is recorded, so the whole run can be saved and charted.
    energies = []
    recorder = Recorder(["life", "energy"])

    for life in range(1, LIVES + 1):
        energy = live(network)
        energies.append(energy)
        recorder.record({"life": life, "energy": energy})

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f}")

    # Results go in runs/, which git ignores: they can be recreated at any
    # time by running the experiment. The chart is smoothed with a moving
    # average, because energy varies a lot from one life to the next.
    os.makedirs("runs", exist_ok=True)
    recorder.save("runs/curiosity.csv")
    line_chart(
        moving_average(energies, 100),
        "runs/curiosity.svg",
        "SYNO average energy per life (moving average of 100)",
    )
    print("Saved runs/curiosity.csv and runs/curiosity.svg")

if __name__ == "__main__":
    main()
