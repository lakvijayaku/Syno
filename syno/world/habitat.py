# ── SYNO · syno/world/habitat.py ────────────────────────
# The grid world SYNO lives in, with food and water
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

# Each movement action maps to its change in (x, y). Up decreases y, because
# y counts rows from the top of the grid.
ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

class Habitat:
    """
    A rectangular grid containing SYNO and food.

    Positions are (x, y) tuples: x is the column, counting left to right, and
    y is the row, counting top to bottom, so (0, 0) is the top-left corner.
    The Habitat defines only what is physically possible. It never decides
    what SYNO should do.
    """

    def in_bounds(self, position: tuple[int, int]) -> bool:
        """
        Checks whether a position lies on the grid.

        :param position: An (x, y) position.
        :return: True if the position is on the grid, otherwise False.
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def __init__(self, width: int, height: int, agent: tuple[int, int], food: list[tuple[int, int]], water: list[tuple[int, int]] = ()):
        """
        Initializes a habitat.

        :param width: The number of columns. Must be at least 1.
        :param height: The number of rows. Must be at least 1.
        :param agent: SYNO's starting (x, y) position.
        :param food: A list of (x, y) positions that contain food.
        :param water: A list of (x, y) positions that contain water. Empty by
            default, so worlds without water are unchanged.
        :raises ValueError: If the size is invalid or any position is off the grid.
        """
        # Guard clauses: reject an invalid world before storing anything.
        if width < 1 or height < 1:
            raise ValueError("Width and height must be at least 1.")
        # The size is stored before the position checks, because in_bounds
        # reads self.width and self.height.
        self.width = width
        self.height = height
        if not self.in_bounds(agent):
            raise ValueError(f"Agent position {agent} is out of bounds.")
        for position in food:
            if not self.in_bounds(position):
                raise ValueError(f"Food position {position} is out of bounds.")
        for position in water:
            if not self.in_bounds(position):
                raise ValueError(f"Water position {position} is out of bounds.")
        # Tuples cannot be changed, so the agent position is stored as given.
        # The food and water lists are copied, so changes to the caller's
        # lists cannot add or remove anything behind the habitat's back. The
        # default for water is an empty tuple rather than [], because a list
        # default would be created once and shared by every habitat.
        self.agent = agent
        self.food = list(food)
        self.water = list(water)

    def render(self) -> str:
        """
        Draws the habitat as text, one line per row.

        S marks SYNO, F marks food, W marks water, and . marks an empty
        square. When SYNO stands on food or water, the square shows S.

        :return: The grid as a multi-line string.
        """
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.agent:
                    row += "S"
                elif (x, y) in self.food:
                    row += "F"
                elif (x, y) in self.water:
                    row += "W"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def move(self, action: str) -> bool:
        """
        Moves SYNO one square in the given direction, if the grid allows it.

        :param action: One of the keys of ACTIONS: "up", "down", "left", or "right".
        :return: True if SYNO moved, or False if the edge of the grid blocked it.
        :raises ValueError: If the action is not recognized.
        """
        if action not in ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid options are: {list(ACTIONS.keys())}")
        dx, dy = ACTIONS[action]
        x, y = self.agent
        new_position = (x + dx, y + dy)
        # The edge of the grid acts as a wall: SYNO stays where it is. The
        # habitat only reports what happened; it never judges the move.
        if not self.in_bounds(new_position):
            return False
        self.agent = new_position
        return True

    def eat(self) -> bool:
        """
        Eats the food on SYNO's square, if there is any.

        :return: True if food was eaten, or False if the square had no food.
        """
        if self.agent not in self.food:
            return False
        # Eaten food is removed from the world. remove() deletes only the
        # first matching position.
        self.food.remove(self.agent)
        return True

    def drink(self) -> bool:
        """
        Drinks from SYNO's square, if it has water.

        Unlike food, water is not used up: it is a pond, not a meal.

        :return: True if SYNO's square has water, or False if it does not.
        """
        return self.agent in self.water

    def sense(self, radius: int) -> list[float]:
        """
        Describes the square window around SYNO as numbers its brain can read.

        Each square becomes 1.0 for food, 0.0 for empty, or -1.0 for off the
        grid. Values are ordered row by row from the top-left of the window,
        the same order as render. SYNO's own square is included.

        :param radius: How many squares SYNO can see in each direction.
        :return: A list of (2 * radius + 1) ** 2 values.
        :raises ValueError: If radius is negative.
        """
        if radius < 0:
            raise ValueError("Radius must be at least 0.")

        x, y = self.agent
        senses = []

        # This reports raw facts only. It never tells SYNO where to go; the
        # brain must learn what the numbers mean. The range ends at radius + 1
        # because range excludes its end value.
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                position = (x + dx, y + dy)
                # Walls get their own value, so SYNO can tell the edge of the
                # world apart from empty floor.
                if not self.in_bounds(position):
                    senses.append(-1.0)
                elif position in self.food:
                    senses.append(1.0)
                else:
                    senses.append(0.0)

        return senses

    def sense_water(self, radius: int) -> list[float]:
        """
        Describes where water is in the square window around SYNO.

        Uses the same window and order as sense. Each square becomes 1.0 for
        water and 0.0 otherwise. Walls are already reported by sense, so they
        read 0.0 here.

        :param radius: How many squares SYNO can see in each direction.
        :return: A list of (2 * radius + 1) ** 2 values.
        :raises ValueError: If radius is negative.
        """
        if radius < 0:
            raise ValueError("Radius must be at least 0.")

        x, y = self.agent
        senses = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                position = (x + dx, y + dy)
                if self.in_bounds(position) and position in self.water:
                    senses.append(1.0)
                else:
                    senses.append(0.0)

        return senses
