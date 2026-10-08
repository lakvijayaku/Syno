"""
Unit tests for syno.world.habitat.

These tests are intentionally stricter than the Habitat's own safeguards. They
verify the coordinate system, every boundary of the grid, input validation,
rendering, and that the habitat does not share outside data.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.world.habitat import Habitat


class TestHabitatBounds(unittest.TestCase):
    """Verifies which positions count as on the grid."""

    def setUp(self):
        self.habitat = Habitat(5, 3, (0, 0), [])

    def test_all_four_corners_are_in_bounds(self):
        for corner in [(0, 0), (4, 0), (0, 2), (4, 2)]:
            with self.subTest(corner=corner):
                self.assertTrue(self.habitat.in_bounds(corner))

    def test_positions_just_outside_each_edge_are_out_of_bounds(self):
        for position in [(-1, 0), (5, 0), (0, -1), (0, 3), (5, 3), (-1, -1)]:
            with self.subTest(position=position):
                self.assertFalse(self.habitat.in_bounds(position))

    def test_every_square_is_in_bounds(self):
        for y in range(3):
            for x in range(5):
                self.assertTrue(self.habitat.in_bounds((x, y)))

    def test_width_and_height_are_not_swapped(self):
        # (4, 0) exists on a 5-wide, 3-tall grid, but (0, 4) does not.
        self.assertTrue(self.habitat.in_bounds((4, 0)))
        self.assertFalse(self.habitat.in_bounds((0, 4)))


class TestHabitatValidation(unittest.TestCase):
    """Verifies that an invalid habitat fails loudly."""

    def test_zero_or_negative_size_raises_value_error(self):
        for width, height in [(0, 3), (5, 0), (-1, 3), (5, -2)]:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    Habitat(width, height, (0, 0), [])

    def test_agent_off_the_grid_raises_value_error(self):
        for agent in [(5, 1), (1, 3), (-1, 0), (0, -1)]:
            with self.subTest(agent=agent):
                with self.assertRaises(ValueError):
                    Habitat(5, 3, agent, [])

    def test_food_off_the_grid_raises_value_error(self):
        with self.assertRaises(ValueError):
            Habitat(5, 3, (0, 0), [(1, 1), (0, 3)])

    def test_smallest_possible_habitat(self):
        habitat = Habitat(1, 1, (0, 0), [])
        self.assertEqual(habitat.render(), "S")


class TestHabitatRender(unittest.TestCase):
    """Verifies the text drawing of the grid."""

    def test_step_3a_example(self):
        habitat = Habitat(5, 3, (1, 1), [(3, 0)])
        self.assertEqual(habitat.render(), "...F.\n.S...\n.....")

    def test_rows_and_columns_have_the_right_size(self):
        rows = Habitat(7, 4, (0, 0), []).render().split("\n")
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row), 7)

    def test_y_counts_from_the_top(self):
        rows = Habitat(3, 3, (0, 2), []).render().split("\n")
        self.assertEqual(rows[2], "S..")

    def test_multiple_food_positions(self):
        habitat = Habitat(3, 2, (0, 0), [(2, 0), (1, 1), (2, 1)])
        self.assertEqual(habitat.render(), "S.F\n.FF")

    def test_agent_on_food_shows_agent(self):
        habitat = Habitat(3, 1, (1, 0), [(1, 0)])
        self.assertEqual(habitat.render(), ".S.")

    def test_render_does_not_change_the_habitat(self):
        habitat = Habitat(5, 3, (1, 1), [(3, 0)])
        habitat.render()
        self.assertEqual(habitat.agent, (1, 1))
        self.assertEqual(habitat.food, [(3, 0)])


class TestHabitatStateIsolation(unittest.TestCase):
    """Verifies that the habitat does not share outside data."""

    def test_changing_original_food_list_does_not_affect_habitat(self):
        food = [(3, 0)]
        habitat = Habitat(5, 3, (1, 1), food)
        food.append((0, 0))
        self.assertEqual(habitat.food, [(3, 0)])

    def test_habitat_stores_a_different_food_list(self):
        food = [(3, 0)]
        habitat = Habitat(5, 3, (1, 1), food)
        self.assertIsNot(habitat.food, food)


if __name__ == "__main__":
    unittest.main()
