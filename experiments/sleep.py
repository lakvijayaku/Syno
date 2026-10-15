# ── SYNO · experiments/sleep.py ─────────────────────────
# SYNO forages and sleeps
# Laksheth Vijayakumar · 2026-10-14 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import FatiguedCore, homeostatic_reward
from syno.brain.novelty import NoveltySystem
from syno.brain.memory import MemoryStore

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

from experiments.scarcity import GRID_SIZE, REGROW_DELAY
# SYNO's view with its eyes open or closed, from the blink experiment. A
# sleeping SYNO sees as it does in the dark.
from experiments.blink import view, WINDOW
from experiments.consolidation import PRIORITY_FLOOR

# Sleep is a sixth action, next to moving and eating. Whether and when to
# sleep is learned, never set by a rule.
ACTIONS = ACTION_NAMES + ["sleep"]

# SYNO senses its fatigue and has one more action than in the scarcity
# experiment, so it needs a larger brain, as in the thirst and blink
# experiments.
HIDDEN_NEURONS = 16
LIVES = 3000


def senses(habitat: Habitat, body: FatiguedCore, visible: bool) -> list[float]:
    """
    Combines what SYNO sees with how its body feels, including how tired it
    is, so it can learn when to sleep.

    :param habitat: The habitat SYNO is in.
    :param body: SYNO's body.
    :param visible: True if SYNO's eyes are open, False while it sleeps.
    :return: The view from blink.view, followed by the energy deficit,
        stomach fill, and fatigue.
    """
    return view(habitat, visible) + [body.deficit(), body.stomach, body.fatigue]


def live(network: Network, memory: MemoryStore,
         awake_replays: int = REPLAYS_PER_STEP,
         sleep_replays: int = REPLAYS_PER_STEP) -> tuple[float, float]:
    """
    Runs one life of LIFE_STEPS steps in the scarce world, where SYNO can
    also sleep.

    Identical to scarcity.live, except that the body gets tired, SYNO can
    sleep, and while asleep it does not move, its eyes are closed for the
    next view, and it gets no novelty bonus.

    :param network: The Decision Network, with one output per action.
    :param memory: The Memory Store, shared across lives.
    :param awake_replays: How many memories SYNO replays after each step
        awake.
    :param sleep_replays: How many memories SYNO replays after each step
        asleep. Both default to REPLAYS_PER_STEP, so replay does not depend
        on sleep unless an experiment chooses otherwise.
    :return: SYNO's average energy over the life, and the fraction of steps
        it spent asleep.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

    novelty = NoveltySystem(NOVELTY_SCALE, HABITUATION, RECOVERY)
    novelty.signal(habitat.agent)

    # Starting fatigue varies, so SYNO learns from both rested and tired
    # starts.
    body = FatiguedCore(random.uniform(0.3, 0.9), 0.0, random.uniform(0.0, 0.5))
    state = senses(habitat, body, True)
    total_energy = 0.0
    steps_asleep = 0
    regrow_timer = 0

    for _ in range(LIFE_STEPS):
        action = choose_action(network.forward(state), EPSILON)
        name = ACTIONS[action]
        slept = name == "sleep"

        drive_before = body.drive()
        moved = False

        if name == "eat":
            if habitat.eat():
                body.eat(FOOD_AMOUNT)
                regrow_timer = REGROW_DELAY
        # A sleeping SYNO stays where it is.
        elif not slept:
            moved = habitat.move(name)

        if regrow_timer > 0:
            regrow_timer -= 1
            if regrow_timer == 0:
                spawn_food(habitat)

        body.tick(moved, slept)

        reward = REWARD_SCALE * homeostatic_reward(drive_before, body.drive())
        # SYNO cannot notice new places with its eyes closed, so novelty
        # neither recovers nor rewards while it sleeps.
        if not slept:
            novelty.tick()
            reward += novelty.signal(habitat.agent)

        # The eyes stay closed for the view after a step asleep.
        next_state = senses(habitat, body, not slept)

        rpe = learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)
        memory.store((state, action, reward, next_state), abs(rpe) + PRIORITY_FLOOR)

        # How much SYNO replays depends on whether it slept this step.
        for _ in range(sleep_replays if slept else awake_replays):
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
        if slept:
            steps_asleep += 1

    return total_energy / LIFE_STEPS, steps_asleep / LIFE_STEPS


def main():
    """
    Trains SYNO over many lives with the option to sleep and prints its
    average energy and time asleep every 500 lives.
    """
    random.seed(0)

    # The view, then the energy deficit, stomach fill, and fatigue.
    inputs = WINDOW + 3

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTIONS), HIDDEN_NEURONS, "linear")
    ])

    energies = []
    sleep_fractions = []
    # Created once, so memories carry over from one life to the next.
    memory = MemoryStore(MEMORY_CAPACITY)

    for life in range(1, LIVES + 1):
        energy, sleep_fraction = live(network, memory)
        energies.append(energy)
        sleep_fractions.append(sleep_fraction)

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            avg_sleep = sum(sleep_fractions[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f} | Time Asleep: {avg_sleep:.3f}")


if __name__ == "__main__":
    main()
