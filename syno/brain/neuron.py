# ── SYNO · syno/brain/neuron.py ─────────────────────────
# A single artificial neuron with a stable sigmoid
# Laksheth Vijayakumar · 2026-10-04 · GPL-3.0
# ────────────────────────────────────────────────────────

import math

class Neuron:
    """
    A single artificial neuron: a weighted sum of its inputs plus a bias,
    passed through an activation function.

    It is the smallest building block of SYNO's Decision Network. Neurons
    are grouped into a Layer, and layers into a Network.
    """

    def __init__(self, weights: list[float], bias: float, activation: str = "sigmoid"):
        """
        Initializes a simple artificial neuron.

        :param weights: A list of weights (floats) for each input.
        :param bias: The bias value (float) of the neuron.
        :param activation: "sigmoid" squashes the output into the range 0 to
            1. "linear" outputs the weighted sum unchanged, so the neuron can
            produce any value, including negative ones.
        :raises ValueError: If activation is not "sigmoid" or "linear".
        """
        if activation not in ("sigmoid", "linear"):
            raise ValueError("activation must be either 'sigmoid' or 'linear'")
        # Store a copy of the weights rather than a reference to the caller's
        # list, so changes made to the original list outside this class cannot
        # alter the neuron's weights.
        self.weights = list(weights)
        # The bias shifts the neuron's activation threshold, just as b shifts
        # the line in y = mx + b. Without it, a sigmoid neuron would always
        # output 0.5, and a linear one 0.0, when every input is 0.
        self.bias = bias
        self.activation = activation

    def forward(self, inputs: list[float]) -> float:
        """
        Calculates the neuron's output: the weighted sum of its inputs plus
        the bias, passed through a numerically stable sigmoid, or returned
        unchanged by a linear neuron.

        :param inputs: A list of input features (floats).
        :return: The neuron's output, from 0.0 to 1.0 for a sigmoid neuron.
        :raises ValueError: If the number of inputs does not match the number of weights.
        """
        # Each weight must have exactly one input. Without this check, extra
        # inputs would be silently ignored, and missing ones would raise an
        # unclear IndexError partway through the loop.
        if len(self.weights) != len(inputs):
            raise ValueError(f"Mismatched input dimensions. Expected {len(self.weights)} inputs, got {len(inputs)}.")
        # Multiply each input by its matching weight and add them up.
        weighted_sum = 0.0
        for i in range(len(inputs)):
            weighted_sum += self.weights[i] * inputs[i]
        # The bias is added once, after the loop. Adding in this order keeps
        # every result identical to earlier versions, since floats round
        # differently when the same numbers are added in another order.
        weighted_sum += self.bias

        # A linear neuron skips the sigmoid, so its output is not limited to
        # the range 0 to 1. Output layers that predict values use this.
        if self.activation == "linear":
            return weighted_sum

        # Sigmoid is computed in two branches so that math.exp() only ever
        # receives a value <= 0. Floats overflow above roughly 1.8e308, so
        # math.exp() raises an OverflowError for inputs above about 709.
        #
        # For a non-negative weighted sum, use the standard sigmoid formula.
        # Here, -weighted_sum <= 0, so math.exp() cannot overflow.
        if weighted_sum >= 0:
            return 1 / (1 + math.exp(-weighted_sum))
        # For a negative weighted sum, use the equivalent alternative form of
        # the sigmoid formula. Here, math.exp(weighted_sum) shrinks toward 0
        # instead of growing toward infinity, so it cannot overflow.
        else:
            exp_z = math.exp(weighted_sum)
            return exp_z / (1 + exp_z)

    def slope(self, output: float) -> float:
        """
        Measures how fast the neuron's output changes with its weighted sum,
        for use in the chain rule during training.

        :param output: The output the neuron produced.
        :return: 1.0 for a linear neuron, or output * (1 - output) for a
            sigmoid neuron.
        """
        if self.activation == "linear":
            return 1.0
        return output * (1 - output)
