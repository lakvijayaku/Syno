# ── SYNO · syno/brain/training.py ───────────────────────
# Teaches neurons and networks by nudging their weights downhill
# Laksheth Vijayakumar · 2026-10-06 · GPL-3.0
# ────────────────────────────────────────────────────────

from syno.brain.neuron import Neuron
from syno.brain.network import Network
from syno.brain.loss import mean_squared_error

def train_step(neuron: Neuron, inputs: list[float], target: float, learning_rate: float) -> float:
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

def train_network_step(network: Network, inputs: list[float], targets: list[float], learning_rate: float) -> float:
    """
    Performs one step of backpropagation and gradient descent on a network.

    :param network: The network to train. Every weight and bias is updated in place.
    :param inputs: A list of input features (floats).
    :param targets: The correct outputs for these inputs, one per output neuron.
    :param learning_rate: How large a step to take downhill.
    :return: The loss before this update.
    :raises ValueError: If the inputs or targets do not match the network's size.
    """
    # Forward pass, recording what went into each layer. A layer's inputs are
    # the previous layer's outputs, so this one list provides both the inputs
    # for the weight updates and the hidden outputs for the backward pass.
    layer_inputs = []
    signal = inputs

    for layer in network.layers:
        layer_inputs.append(signal)
        signal = layer.forward(signal)

    # The loss is computed before any weight changes. mean_squared_error
    # raises if the targets do not match the outputs, so a bad call fails
    # before the network is modified.
    outputs = signal
    loss = mean_squared_error(outputs, targets)

    # Output deltas: the same chain rule as train_step, but with 2 / n in
    # place of 2, because the loss averages over n outputs.
    deltas = [ (2 / len(outputs)) * (output - target) * output * (1 - output) for output, target in zip(outputs, targets, strict=True)]

    # Backward pass: walk the layers from last to first.
    for layer_index in reversed(range(len(network.layers))):
        layer = network.layers[layer_index]
        layer_in = layer_inputs[layer_index]

        # Pass the blame back to the previous layer. Each hidden neuron's error
        # is the sum of the deltas of the neurons it feeds, weighted by its
        # connection to each, then multiplied by its own sigmoid slope. This
        # must run before the updates below, so it uses the same weights as
        # the forward pass. The first layer has no previous layer to blame.
        if layer_index > 0:
            previous_deltas = []
            for k in range(len(layer_in)):
                error = 0.0
                for j in range(len(layer.neurons)):
                    error += deltas[j] * layer.neurons[j].weights[k]
                previous_deltas.append(error * layer_in[k] * (1 - layer_in[k]))

        # Update every neuron in this layer, exactly as in train_step.
        for j in range(len(layer.neurons)):
            neuron = layer.neurons[j]
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= learning_rate * deltas[j] * layer_in[i]
            neuron.bias -= learning_rate * deltas[j]

        # The previous layer's deltas become the current deltas for the next
        # iteration of the loop.
        if layer_index > 0:
            deltas = previous_deltas

    return loss
