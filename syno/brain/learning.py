# ── SYNO · syno/brain/learning.py ───────────────────────
# Teaches SYNO from its own dopamine signal
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

from syno.brain.network import Network
from syno.brain.training import train_network_step
from syno.brain.dopamine import reward_prediction_error

def learn(
    network: Network,
    state: list[float],
    action: int,
    reward: float,
    next_state: list[float],
    done: bool,
    discount: float,
    learning_rate: float
) -> float:
    """
    Performs one step of Q-learning: SYNO updates its prediction for the
    action it took, using its own reward prediction error.

    :param network: The Decision Network, with one output per action.
    :param state: SYNO's senses before acting.
    :param action: The position of the action SYNO took.
    :param reward: The reward received for the step.
    :param next_state: SYNO's senses after acting.
    :param done: True if the episode ended with this step.
    :param discount: How much future reward counts, from 0.0 to 1.0.
    :param learning_rate: How large a step to take downhill.
    :return: The reward prediction error for this step.
    :raises ValueError: If action is not a valid position in the network's outputs.
    """
    # What SYNO predicted for the action it took, and the best it believes it
    # can do from the next state. Using the best next action lets SYNO learn
    # the value of acting well even while it is still exploring.
    values = network.forward(state)
    # Guard: a negative action is valid Python indexing (values[-1] is the
    # last item), so without this check a bad action would silently train
    # the wrong output.
    if not (0 <= action < len(values)):
        raise ValueError(f"Action index {action} is out of bounds for network output length {len(values)}.")
    expected = values[action]
    next_expected = max(network.forward(next_state))

    # The dopamine signal: how much better or worse the step went than
    # predicted.
    rpe = reward_prediction_error(reward, expected, next_expected, discount, done)

    # Only the taken action gets a new target. Every other action keeps its
    # current value as its target, so its error is zero and it does not
    # change. The list is copied so values keeps the original predictions.
    targets = list(values)
    targets[action] = expected + rpe

    train_network_step(network, state, targets, learning_rate)
    return rpe
