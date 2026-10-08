# ── SYNO · experiments/forage.py ────────────────────────
# SYNO learns to find and eat food on its own
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from experiments.xor import make_layer
from syno.brain.network import Network
from syno.brain.policy import choose_action
from syno.brain.learning import learn
from syno.world.habitat import Habitat

# Each position matches one output neuron of the Decision Network.
ACTION_NAMES = ["up", "down", "left", "right", "eat"]

# A 3x3 grid with a sense radius of 2 lets SYNO see the whole grid from any
# square. On a 5x5 grid, food was often out of sight and rewards were too
# rare to learn from in a reasonable number of episodes.
GRID_SIZE = 3
SENSE_RADIUS = 2
HIDDEN_NEURONS = 8
LEARNING_RATE = 1.0
# Temporary, hard-coded exploration rate (see syno/brain/policy.py).
EPSILON = 0.2
DISCOUNT = 0.9
EPISODES = 3000
MAX_STEPS = 50

def random_habitat() -> Habitat:
    """
    Builds a habitat with SYNO and one piece of food on different squares.

    :return: A new Habitat with random starting positions.
    """
    while True:
        agent = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
        food = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
        if agent != food:
            break

    return Habitat(GRID_SIZE, GRID_SIZE, agent, [food])

def run_episode(network: Network) -> int:
    """
    Lets SYNO act and learn in one habitat until it eats or runs out of time.

    The only reward is 1.0 for eating, a temporary stand-in for hunger
    reduction. Nothing tells SYNO how to reach the food.

    :param network: The Decision Network, with one output per action.
    :return: The number of steps taken, or MAX_STEPS if SYNO never ate.
    """
    habitat = random_habitat()
    state = habitat.sense(SENSE_RADIUS)

    for step in range(1, MAX_STEPS + 1):
        action = choose_action(network.forward(state), EPSILON)
        name = ACTION_NAMES[action]

        if name == "eat":
            reward = 1.0 if habitat.eat() else 0.0
        else:
            habitat.move(name)
            reward = 0.0

        # Only eating ends the episode. Running out of steps does not count as
        # done, because the world did not end; the experiment simply stopped
        # watching.
        done = reward > 0
        next_state = habitat.sense(SENSE_RADIUS)

        learn(
            network,
            state,
            action,
            reward,
            next_state,
            done,
            DISCOUNT,
            LEARNING_RATE
        )

        state = next_state

        if done:
            return step

    return MAX_STEPS

def main():
    """
    Trains SYNO over many episodes and prints how many steps it needs, on
    average, to find and eat the food. A falling number means SYNO is
    learning.
    """
    random.seed(0)

    inputs = (2 * SENSE_RADIUS + 1) ** 2

    network = Network([
        make_layer(HIDDEN_NEURONS, inputs),
        make_layer(len(ACTION_NAMES), HIDDEN_NEURONS)
    ])

    steps = []

    for episode in range(EPISODES):
        steps.append(run_episode(network))

        if (episode + 1) % 500 == 0:
            avg_steps = sum(steps[-500:]) / 500
            print(f"Episode {episode + 1} | Avg Steps: {avg_steps}")

if __name__ == "__main__":
    main()
