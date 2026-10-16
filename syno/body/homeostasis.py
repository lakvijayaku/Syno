# ── SYNO · syno/body/homeostasis.py ─────────────────────
# The Homeostatic Core: SYNO's energy, stomach, water, fatigue, and injury
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

# Physiology constants. Hard-coding these is allowed by ADR 0001: they
# define the body, not behavior.
DIGESTION_RATE = 0.05
BASE_BURN = 0.01
MOVE_BURN = 0.01
STOMACH_CAPACITY = 1.0
# Squaring the deficit makes the same amount of food worth more to a hungrier
# SYNO (alliesthesia).
DRIVE_EXPONENT = 2

# Water is lost every step, by breathing and sweating, whether or not SYNO
# moves. Used only by HydratedCore.
WATER_BURN = 0.01

# Fatigue builds up on every waking step, like sleep pressure, whether or
# not SYNO moves, and falls on every step spent asleep. Sleeping halves the
# base burn. Used only by FatiguedCore.
FATIGUE_RATE = 0.02
RECOVERY_RATE = 0.05
SLEEP_SAVING = 0.5

# Injury heals by 0.02 every step, so the worst injury (1.0) heals in 50
# steps. That is half a life, long enough that pain matters to SYNO for a
# while after it's hurt. Used only by InjuredCore.
HEAL_RATE = 0.02

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

    def drive(self) -> float:
        """
        Measures how strongly SYNO's body pushes it back toward its set-point.

        :return: The deficit raised to DRIVE_EXPONENT, from 0.0 to 1.0.
        """
        return self.deficit() ** DRIVE_EXPONENT

def homeostatic_reward(drive_before: float, drive_after: float) -> float:
    """
    Calculates reward as the reduction in drive over a step, following
    homeostatic reinforcement learning (Keramati & Gutkin, 2014).

    Reward is positive when a step brings SYNO closer to its set-point and
    negative when it moves SYNO further away. No behavior is rewarded
    directly.

    :param drive_before: The drive at the start of the step.
    :param drive_after: The drive at the end of the step.
    :return: The reward for the step.
    """
    return drive_before - drive_after


class HydratedCore(HomeostaticCore):
    """
    SYNO's body with a second need: water, which drains every step and is
    refilled by drinking.

    Energy, the stomach, and digestion work exactly as in HomeostaticCore.
    Water goes straight into the body when drunk, without a stomach. SYNO
    dies when either energy or water runs out.
    """

    def __init__(self, energy: float, stomach: float, water: float):
        """
        Initializes the body.

        :param energy: Starting energy, from 0.0 to 1.0.
        :param stomach: Starting stomach fill, from 0.0 to 1.0.
        :param water: Starting water, from 0.0 to 1.0.
        :raises ValueError: If any value is outside 0.0 to 1.0.
        """
        super().__init__(energy, stomach)
        if not 0.0 <= water <= 1.0:
            raise ValueError("Water must be between 0.0 and 1.0")
        self.water = water

    def drink(self, amount: float) -> float:
        """
        Adds water to the body, up to full.

        :param amount: How much water SYNO tries to drink.
        :return: How much water was accepted, which may be less than amount.
        :raises ValueError: If amount is negative.
        """
        if amount < 0.0:
            raise ValueError("amount must not be negative")
        accepted = min(amount, 1.0 - self.water)
        self.water += accepted
        return accepted

    def tick(self, moved: bool) -> None:
        """
        Advances the body by one step: everything HomeostaticCore does, then
        water loss.

        :param moved: True if SYNO moved this step, which burns extra energy.
        """
        super().tick(moved)
        self.water = max(0.0, self.water - WATER_BURN)

    def water_deficit(self) -> float:
        """
        Measures how far SYNO's water is below its set-point.

        :return: 0.0 when fully watered, up to 1.0 when water is empty.
        """
        return 1.0 - self.water

    def drive(self) -> float:
        """
        Measures the combined push of both needs. Each deficit is squared, so
        whichever need is larger dominates (Keramati & Gutkin, 2014).

        :return: The squared energy deficit plus the squared water deficit,
            from 0.0 to 2.0.
        """
        return self.deficit() ** DRIVE_EXPONENT + self.water_deficit() ** DRIVE_EXPONENT

    def is_dead(self) -> bool:
        """
        Checks whether SYNO has run out of energy or water.

        :return: True if either has reached 0.0.
        """
        return self.energy <= 0.0 or self.water <= 0.0


class FatiguedCore(HomeostaticCore):
    """
    SYNO's body with a second need: rest. Fatigue builds up while SYNO is
    awake and is relieved only by sleeping.

    Energy, the stomach, and digestion work exactly as in HomeostaticCore,
    except that a step spent asleep burns less energy. Fatigue ranges from
    0.0, fully rested, to 1.0, exhausted.
    """

    def __init__(self, energy: float, stomach: float, fatigue: float):
        """
        Initializes the body.

        :param energy: Starting energy, from 0.0 to 1.0.
        :param stomach: Starting stomach fill, from 0.0 to 1.0.
        :param fatigue: Starting fatigue, from 0.0 to 1.0.
        :raises ValueError: If any value is outside 0.0 to 1.0.
        """
        super().__init__(energy, stomach)
        if not 0.0 <= fatigue <= 1.0:
            raise ValueError("Fatigue must be between 0.0 and 1.0 inclusive.")
        self.fatigue = fatigue

    def tick(self, moved: bool, slept: bool = False) -> None:
        """
        Advances the body by one step: everything HomeostaticCore does, then
        fatigue builds up while awake or falls while asleep.

        :param moved: True if SYNO moved this step, which burns extra energy.
        :param slept: True if SYNO slept this step, which saves energy and
            relieves fatigue.
        """
        super().tick(moved)
        if slept:
            # HomeostaticCore has already burned the full BASE_BURN and
            # clamped energy, so part of that burn is given back here. A
            # dead SYNO gets nothing back, so sleep cannot revive it.
            if self.energy > 0.0:
                self.energy = min(1.0, self.energy + BASE_BURN * SLEEP_SAVING)
            self.fatigue = max(0.0, self.fatigue - RECOVERY_RATE)
        else:
            self.fatigue = min(1.0, self.fatigue + FATIGUE_RATE)

    def drive(self) -> float:
        """
        Measures the combined push of both needs. Each is squared, so
        whichever need is larger dominates (Keramati & Gutkin, 2014).

        :return: The squared energy deficit plus the squared fatigue, from
            0.0 to 2.0.
        """
        return self.deficit() ** DRIVE_EXPONENT + self.fatigue ** DRIVE_EXPONENT


class InjuredCore(HomeostaticCore):
    """
    SYNO's body with a second need: to be unhurt. Injury rises when the world
    hurts SYNO and heals slowly on its own.

    Energy, the stomach, and digestion work exactly as in HomeostaticCore.
    Injury ranges from 0.0, unhurt, to 1.0, badly hurt. Pain is not a rule
    about what SYNO should avoid: injury raises the drive, so being hurt is
    a negative reward and healing a small positive one.
    """

    def __init__(self, energy: float, stomach: float, injury: float = 0.0):
        """
        Initializes the body.

        :param energy: Starting energy, from 0.0 to 1.0.
        :param stomach: Starting stomach fill, from 0.0 to 1.0.
        :param injury: Starting injury, from 0.0 to 1.0. Unhurt by default.
        :raises ValueError: If any value is outside 0.0 to 1.0.
        """
        super().__init__(energy, stomach)
        if not 0.0 <= injury <= 1.0:
            raise ValueError("Injury must be between 0.0 and 1.0 inclusive.")
        self.injury = injury

    def hurt(self, amount: float) -> None:
        """
        Injures the body, up to the worst injury.

        The amount comes from what hurt SYNO, such as a thorn, so the body
        does not decide how much anything hurts.

        :param amount: How much injury is added.
        :raises ValueError: If amount is negative.
        """
        if amount < 0.0:
            raise ValueError("Amount to hurt cannot be negative.")
        self.injury = min(1.0, self.injury + amount)

    def tick(self, moved: bool) -> None:
        """
        Advances the body by one step: everything HomeostaticCore does, then
        injury heals by HEAL_RATE.

        :param moved: True if SYNO moved this step, which burns extra energy.
        """
        super().tick(moved)
        self.injury = max(0.0, self.injury - HEAL_RATE)

    def drive(self) -> float:
        """
        Measures the combined push of both needs. Each is squared, so a bad
        injury matters far more than a scratch, and whichever need is larger
        dominates (Keramati & Gutkin, 2014).

        :return: The squared energy deficit plus the squared injury, from 0.0
            to 2.0.
        """
        return self.deficit() ** DRIVE_EXPONENT + self.injury ** DRIVE_EXPONENT
