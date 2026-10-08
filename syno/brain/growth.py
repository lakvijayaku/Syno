# ── SYNO · syno/brain/growth.py ─────────────────────────
# The Growth Engine: adds neurons to a working network
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import random

from syno.brain.neuron import Neuron
from syno.brain.network import Network

def grow_neuron(network: Network, layer_index: int) -> None:
    """
    Adds one neuron to a hidden layer without changing the network's output.

    The new neuron gets random incoming weights, so it starts responding to
    its inputs, but every outgoing weight into the next layer starts at 0.0,
    so at first it has no effect. Training then turns those weights into
    useful ones, and nothing the network has learned is lost by growing.

    :param network: The network to grow. It is changed in place.
    :param layer_index: The position of the hidden layer to grow.
    :raises ValueError: If layer_index is not a hidden layer. The output
        layer cannot grow, because its size is fixed by the number of actions.
    """
    if not (0 <= layer_index < len(network.layers) - 1):
        raise ValueError("Invalid layer index. Only hidden layers can grow.")

    # The new neuron needs as many inputs as its neighbors, and the same
    # activation, so the layer stays uniform.
    layer = network.layers[layer_index]
    num_inputs = len(layer.neurons[0].weights)

    weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
    bias = random.uniform(-1.0, 1.0)
    activation = layer.neurons[0].activation

    layer.neurons.append(Neuron(weights, bias, activation))

    # Function-preserving step: the next layer receives one more input, but
    # with a weight of 0.0, so the network's output does not change.
    for neuron in network.layers[layer_index + 1].neurons:
        neuron.weights.append(0.0)
