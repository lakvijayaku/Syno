# ── SYNO · syno/world/habitat.py ────────────────────────
# The grid world SYNO lives in
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

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

    def __init__(self, width: int, height: int, agent: tuple[int, int], food: list[tuple[int, int]]):
        """
        Initializes a habitat.

        :param width: The number of columns. Must be at least 1.
        :param height: The number of rows. Must be at least 1.
        :param agent: SYNO's starting (x, y) position.
        :param food: A list of (x, y) positions that contain food.
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
        # Tuples cannot be changed, so the agent position is stored as given.
        # The food list is copied, so changes to the caller's list cannot add
        # or remove food behind the habitat's back.
        self.agent = agent
        self.food = list(food)

    def render(self) -> str:
        """
        Draws the habitat as text, one line per row.

        S marks SYNO, F marks food, and . marks an empty square. When SYNO
        stands on food, the square shows S.

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
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

