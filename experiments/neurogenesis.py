# ── SYNO · experiments/neurogenesis.py ──────────────────
# SYNO grows its brain when it stops learning
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
from syno.brain.growth import grow_neuron

# Shared settings and helpers are reused from the earlier experiments, so the
# only differences from the consolidation experiment are defined in this file.
from experiments.hunger import (
    ACTION_NAMES,
    DISCOUNT,
    FOOD_AMOUNT,
    GRID_SIZE,
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

# SYNO starts with a single hidden neuron and grows up to MAX_HIDDEN.
START_HIDDEN = 1
MAX_HIDDEN = 16

# Every GROWTH_CHECK lives, SYNO grows one neuron unless its average surprise
# fell below SURPRISE_DROP times the previous GROWTH_CHECK lives' average.
GROWTH_CHECK = 100
SURPRISE_DROP = 0.95

# Fewer lives than the earlier experiments, which ran 5000: with replay,
# SYNO has mastered the task well within 2000.
LIVES = 2000

def live(network: Network, memory: MemoryStore) -> tuple[float, float]:
    """
    Runs one life of LIFE_STEPS steps, learning after every step and then
    replaying memories in proportion to how surprising they are.

    Identical to consolidation.live, except that it also measures how
    surprised SYNO was, on average, during the life.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :return: SYNO's average energy and average surprise (size of the RPE)
        over the life.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    body = HomeostaticCore(random.uniform(0.3, 0.9), 0.0)
    state = senses(habitat, body)
    total_energy = 0.0
    total_surprise = 0.0

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

        rpe = learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)
        # Surprise is the size of the RPE, a signal the brain produces
        # itself. It is what SYNO uses to decide when to grow.
        total_surprise += abs(rpe)
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

    return total_energy / LIFE_STEPS, total_surprise / LIFE_STEPS

def main():
    """
    Trains SYNO from a one-neuron brain that grows when learning stalls,
    prints its average energy and brain size every 250 lives, and saves every
    life to runs/neurogenesis.csv and a chart to runs/neurogenesis.svg.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    network = Network([
        make_layer(START_HIDDEN, inputs),
        make_layer(len(ACTION_NAMES), START_HIDDEN, "linear")
    ])

    memory = MemoryStore(MEMORY_CAPACITY)
    energies = []
    surprises = []
    recorder = Recorder(["life", "energy", "surprise", "hidden"])

    for life in range(1, LIVES + 1):
        energy, surprise = live(network, memory)
        energies.append(energy)
        surprises.append(surprise)

        # Growth depends on a trend across many lives, so it is checked here
        # rather than inside live. While SYNO is learning, its surprise keeps
        # falling. When it stops falling, the brain may be too small to learn
        # more, so it grows. Known limitation: surprise also stops falling
        # once the task is mastered, so SYNO keeps growing until MAX_HIDDEN.
        if life % GROWTH_CHECK == 0 and life >= 2 * GROWTH_CHECK:
            recent = sum(surprises[-GROWTH_CHECK:])
            earlier = sum(surprises[-2 * GROWTH_CHECK:-GROWTH_CHECK])
            hidden = len(network.layers[0].neurons)

            if recent > SURPRISE_DROP * earlier and hidden < MAX_HIDDEN:
                grow_neuron(network, 0)
                print(f"Life {life:4d} | Grew to {hidden + 1} hidden neurons")

        current_hidden = len(network.layers[0].neurons)
        recorder.record({
            "life": life,
            "energy": energy,
            "surprise": surprise,
            "hidden": current_hidden
        })

        if life % 250 == 0:
            avg = sum(energies[-250:]) / 250
            print(f"Life {life:4d} | Avg Energy: {avg:.3f} | Hidden: {current_hidden}")

    os.makedirs("runs", exist_ok=True)
    recorder.save("runs/neurogenesis.csv")
    line_chart(
        moving_average(energies, 100),
        "runs/neurogenesis.svg",
        "With a growing brain (moving average of 100)",
    )
    print("Saved runs/neurogenesis.csv and runs/neurogenesis.svg")

if __name__ == "__main__":
    main()
