# ── SYNO · experiments/hunger.py ────────────────────────
# Trains SYNO to manage its internal homeostatic energy balance
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat
from syno.body.homeostasis import HomeostaticCore, homeostatic_reward

# Each position matches one output neuron of the Decision Network.
ACTION_NAMES = ["up", "down", "left", "right", "eat"]
GRID_SIZE = 3
SENSE_RADIUS = 2
HIDDEN_NEURONS = 8
LEARNING_RATE = 0.1

# Higher than in forage.py. Eating off food means standing still, which costs
# less energy than moving, so with less exploration SYNO learned to stand
# still and never discovered that food helps.
EPSILON = 0.3
DISCOUNT = 0.9
FOOD_AMOUNT = 0.3

# Body rewards are around 0.03, too small next to the network's random
# starting values. Scaling them makes the signal strong enough to learn from.
REWARD_SCALE = 10.0
LIVES = 5000
LIFE_STEPS = 100

def spawn_food(habitat: Habitat) -> None:
    """
    Places one new piece of food on a random free square. Works for a
    habitat of any size.

    :param habitat: The habitat to add food to.
    """
    while True:
        position = (random.randrange(habitat.width), random.randrange(habitat.height))
        if position != habitat.agent and position not in habitat.food:
            habitat.food.append(position)
            return

def senses(habitat: Habitat, body: HomeostaticCore) -> list[float]:
    """
    Combines what SYNO sees with how its body feels.

    :param habitat: The habitat SYNO is in.
    :param body: SYNO's body.
    :return: The visual window, followed by the energy deficit and stomach fill.
    """
    return habitat.sense(SENSE_RADIUS) + [body.deficit(), body.stomach]

def live(network: Network) -> float:
    """
    Runs one life of LIFE_STEPS steps, learning after every step.

    Reward comes only from the body getting closer to its set-point. Eating
    is never rewarded directly, and food reappears each time it is eaten.

    :param network: The Decision Network, with one output per action.
    :return: SYNO's average energy over the life.
    """
    agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    habitat = Habitat(GRID_SIZE, GRID_SIZE, agent, [])
    spawn_food(habitat)

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
        next_state = senses(habitat, body)

        # done is always False. If starving ended a life, SYNO would learn
        # that ending the negative rewards is good, so death would look
        # attractive.
        learn(network, state, action, reward, next_state, False, DISCOUNT, LEARNING_RATE)

        state = next_state
        total_energy += body.energy

    return total_energy / LIFE_STEPS

def main():
    """
    Trains SYNO over many lives and prints its average energy every 500
    lives. A rising number means SYNO is learning to keep itself fed.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2 + 2

    # Sigmoid hidden neurons, linear outputs: values can be negative because
    # body rewards can be negative.
    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS, "linear")
    ])

    energies = []

    for life in range(1, LIVES + 1):
        energies.append(live(network))

        if life % 500 == 0:
            avg_energy = sum(energies[-500:]) / 500
            print(f"Life {life:4d} | Avg Energy: {avg_energy:.3f}")

if __name__ == "__main__":
    main()

