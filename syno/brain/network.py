# ── SYNO · syno/brain/network.py ────────────────────────
# A stack of layers that feed into each other
# Laksheth Vijayakumar · 2026-10-04 · GPL-3.0
# ────────────────────────────────────────────────────────

from syno.brain.layer import Layer

class Network:

    def __init__(self, layers: list[Layer]):
        """
        Initializes a network from a stack of layers.

        :param layers: A list of Layer instances, ordered from the first
            hidden layer to the output layer.
        """
        # Store a copy of the list rather than a reference to the caller's
        # list, so adding or removing layers outside this class cannot alter
        # the network.
        self.layers = list(layers)

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Passes the inputs through every layer in order.

        :param inputs: A list of input features (floats).
        :return: The outputs of the final layer.
        :raises ValueError: If the number of values reaching a layer does not
            match its neurons' number of weights.
        """
        # Each layer's outputs become the next layer's inputs. signal always
        # holds the output of the most recent layer, so after the loop it holds
        # the network's final output. A loop is used instead of a list
        # comprehension because each step depends on the previous one.
        signal = inputs
        for layer in self.layers:
            signal = layer.forward(signal)
        return signal
