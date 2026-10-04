# ── SYNO · syno/brain/layer.py ─────────────────────────
# A layer of neurons that handle the same inputs.
# Laksheth Vijayakumar · 2026-10-04 · GPL-3.0
# ────────────────────────────────────────────────────────

from syno.brain.neuron import Neuron

class Layer:
    def __init__(self, neurons: list[Neuron]):
        """
        Initializes a layer of neurons.

        :param neurons: A list of Neuron instances that make up this layer.
        """
        # Store a copy of the list rather than a reference to the caller's
        # list, so adding or removing neurons outside this class cannot alter
        # the layer.
        self.neurons = list(neurons)

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Passes the same inputs through every neuron in the layer.

        :param inputs: A list of input features (floats).
        :return: A list of outputs, one from each neuron, in the same order
            as the neurons.
        :raises ValueError: If the number of inputs does not match a neuron's
            number of weights.
        """
        # Call each neuron's forward method with the same inputs and collect
        # the results in order. The input length is not checked here, because
        # Neuron.forward already raises a ValueError on a mismatch.
        return [neuron.forward(inputs) for neuron in self.neurons]
