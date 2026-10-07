# ── SYNO · syno/brain/neuron.py ─────────────────────────
# A single artificial neuron with a stable sigmoid
# Laksheth Vijayakumar · 2026-10-04 · GPL-3.0
# ────────────────────────────────────────────────────────

import math

class Neuron:
    def __init__(self, weights: list[float], bias: float):
        """
        Initializes a simple artificial neuron.

        :param weights: A list of weights (floats) for each input.
        :param bias: The bias value (float) of the neuron.
        """
        # Store a copy of the weights rather than a reference to the caller's
        # list, so changes made to the original list outside this class cannot
        # alter the neuron's weights.
        self.weights = list(weights)
        # The bias shifts the neuron's activation threshold, just as b shifts
        # the line in y = mx + b. Without it, the output would always be 0.5
        # when every input is 0.
        self.bias = bias

    def forward(self, inputs: list[float]) -> float:
        """
        Calculates the neuron's output using a numerically stable sigmoid activation function.

        :param inputs: A list of input features (floats).
        :return: The sigmoid-activated output of the neuron.
        :raises ValueError: If the number of inputs does not match the number of weights.
        """
        # Pair each weight with its matching input and compute the weighted sum.
        # strict=True makes zip raise a ValueError when the lists differ in
        # length, instead of silently ignoring the extra values.
        try:
            weighted_sum = sum(w * x for w, x in zip(self.weights, inputs, strict=True)) + self.bias
        # Re-raise the error with a clearer message. "from e" preserves the
        # original error as the cause.
        except ValueError as e:
            raise ValueError(
                f"Mismatched input dimensions. Expected {len(self.weights)} inputs, got {len(inputs)}."
            ) from e

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

