# ── SYNO · syno/body/homeostasis.py ─────────────────────
# The Homeostatic Core: SYNO's energy and stomach
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

# Physiology constants. Hard-coding these is allowed by ADR 0001: they
# define the body, not behavior.
DIGESTION_RATE = 0.05
BASE_BURN = 0.01
MOVE_BURN = 0.01
STOMACH_CAPACITY = 1.0

class HomeostaticCore:
    """
    SYNO's body: energy that drains over time, and a stomach that digests
    food into energy.

    Both values range from 0.0 to 1.0. An energy of 1.0 is the set-point,
    meaning SYNO's need is fully satisfied.
    """

    def __init__(self, energy: float, stomach: float):
        """
        Initializes the body.

        :param energy: Starting energy, from 0.0 to 1.0.
        :param stomach: Starting stomach fill, from 0.0 to 1.0.
        :raises ValueError: If either value is outside 0.0 to 1.0.
        """
        if not (0.0 <= energy <= 1.0) or not (0.0 <= stomach <= 1.0):
            raise ValueError("Energy and stomach values must be between 0.0 and 1.0 inclusive.")
        self.energy = energy
        self.stomach = stomach

    def eat(self, amount: float) -> float:
        """
        Puts food into the stomach, up to its capacity.

        :param amount: How much food SYNO tries to eat.
        :return: How much food actually fit, which may be less than amount.
        :raises ValueError: If amount is negative.
        """
        if amount < 0.0:
            raise ValueError("Amount to eat cannot be negative.")

        # A full stomach refuses the rest. This is physics, not a rule about
        # SYNO's behavior.
        room = STOMACH_CAPACITY - self.stomach
        accepted = min(amount, room)
        self.stomach += accepted
        return accepted

    def tick(self, moved: bool) -> None:
        """
        Advances the body by one step: digest, burn energy, then clamp.

        :param moved: True if SYNO moved this step, which burns extra energy.
        """
        # Digestion moves food from the stomach into energy gradually, so
        # eating does not refill energy instantly.
        digested = min(self.stomach, DIGESTION_RATE)
        self.stomach -= digested
        self.energy += digested

        # Living always costs energy, and moving costs more.
        self.energy -= BASE_BURN
        if moved:
            self.energy -= MOVE_BURN

        # Energy beyond the set-point is not stored, so digested food is
        # wasted when energy is already full. This is a deliberate
        # simplification.
        self.energy = max(0.0, min(1.0, self.energy))

    def deficit(self) -> float:
        """
        Measures how far SYNO's energy is below its set-point.

        :return: 0.0 when fully satisfied, up to 1.0 when energy is empty.
        """
        return 1.0 - self.energy
