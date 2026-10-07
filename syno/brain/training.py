# ── SYNO · syno/brain/training.py ───────────────────────
# Teaches a neuron by nudging its weights downhill
# Laksheth Vijayakumar · 2026-10-06 · GPL-3.0
# ────────────────────────────────────────────────────────

from syno.brain.neuron import Neuron
from syno.brain.loss import mean_squared_error

def train_step(neuron: list[Neuron], inputs: list[float], target: float, learning_rate: float) -> float:
    """
    Performs one step of gradient descent on a single neuron.

    :param neuron: The neuron to train. Its weights and bias are updated in place.
    :param inputs: A list of input features (floats).
    :param target: The correct output for these inputs.
    :param learning_rate: How large a step to take downhill.
    :return: The loss before this update.
    :raises ValueError: If the number of inputs does not match the number of weights.
    """
    # Forward pass: compute the neuron's current output and how wrong it is.
    # Neuron.forward checks the input length before anything is changed.
    output = neuron.forward(inputs)
    loss = mean_squared_error([output], [target])

    # delta is the slope of the loss with respect to the weighted sum z, built
    # with the chain rule from two links:
    #   2 * (output - target)     how the loss changes with the output
    #   output * (1 - output)     how the sigmoid output changes with z
    delta = 2 * (output - target) * output * (1 - output)

    # Each weight's slope is delta times its input. Subtracting the slope moves
    # the weight downhill. The loop runs over positions so that each number is
    # assigned back into the list itself.
    for i in range(len(neuron.weights)):
        neuron.weights[i] -= learning_rate * delta * inputs[i]

    # The bias acts like a weight whose input is always 1, so its slope is
    # delta alone.
    neuron.bias -= learning_rate * delta

    return loss
