# ── SYNO · experiments/xor.py ───────────────────────────
# Trains a network to learn XOR
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from syno.brain.neuron import Neuron
from syno.brain.network import Network
from syno.brain.layer import Layer
from syno.brain.training import train_network_step

def make_layer(num_neurons: int, num_inputs: int, activation: str = "sigmoid"):
    """
    Builds a layer of neurons with random starting weights and biases.

    :param num_neurons: How many neurons the layer contains.
    :param num_inputs: How many inputs each neuron receives.
    :param activation: "sigmoid" (the default) or "linear", passed to every neuron.
    :return: A Layer whose weights and biases are drawn uniformly from [-1, 1].
    """
    # Random starting values break the symmetry between neurons. If every
    # neuron started identical, they would all receive identical updates and
    # the layer would behave like a single neuron.
    neurons = []
    for _ in range(num_neurons):
        weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        bias = random.uniform(-1.0, 1.0)
        neurons.append(Neuron(weights, bias, activation))
    return Layer(neurons)

def main():
    """
    Trains a 2-3-1 network on the four XOR cases and prints the results.

    XOR cannot be learned by a single neuron, because no single straight line
    separates its outputs. Solving it confirms that backpropagation trains
    hidden layers correctly.
    """
    # A fixed seed makes every run produce the same starting weights, so
    # results are reproducible.
    random.seed(0)

    # Three hidden neurons solved XOR on every one of 10 tested seeds, while
    # two hidden neurons sometimes got stuck in a local minimum.
    hidden_layer = make_layer(3, 2)
    output_layer = make_layer(1, 3)
    network = Network([hidden_layer, output_layer])

    # Each case pairs the inputs with the target, which is 1 only when the two
    # inputs differ.
    data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]

    # One epoch is one pass over all four cases. The average loss is printed
    # every 1000 epochs to show the network learning.
    for epoch in range(5001):
        total_loss = 0.0

        for inputs, targets in data:
            loss = train_network_step(network, inputs, targets, 1.0)
            total_loss += loss

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Avg Loss: {total_loss / len(data)}")

    print("\nFinal Results:")
    for inputs, targets in data:
        raw_prediction = network.forward(inputs)[0]
        prediction = round(raw_prediction, 3)
        print(f"Inputs: {inputs} -> Predicted: {prediction} | Target: {targets}")

if __name__ == "__main__":
    main()

