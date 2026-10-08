"""
Unit tests for syno.world.habitat.

These tests are intentionally stricter than the Habitat's own safeguards. They
verify the coordinate system, every boundary of the grid, input validation,
rendering, and that the habitat does not share outside data.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import unittest

from syno.world.habitat import ACTIONS, Habitat


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



class TestHabitatMove(unittest.TestCase):
    """Verifies movement and the edges of the grid."""

    def test_each_action_moves_one_square_in_its_direction(self):
        expected = {"up": (2, 0), "down": (2, 2), "left": (1, 1), "right": (3, 1)}
        for action, position in expected.items():
            with self.subTest(action=action):
                habitat = Habitat(5, 3, (2, 1), [])
                self.assertTrue(habitat.move(action))
                self.assertEqual(habitat.agent, position)

    def test_up_decreases_y(self):
        habitat = Habitat(3, 3, (1, 1), [])
        habitat.move("up")
        self.assertEqual(habitat.render().split("\n")[0], ".S.")

    def test_step_3b_sequence(self):
        habitat = Habitat(4, 3, (0, 0), [(3, 2)])
        results = [(habitat.move(a), habitat.agent) for a in ["right", "right", "down", "up", "up"]]
        self.assertEqual(results, [
            (True, (1, 0)),
            (True, (2, 0)),
            (True, (2, 1)),
            (True, (2, 0)),
            (False, (2, 0)),
        ])

    def test_every_edge_blocks_movement(self):
        cases = [
            ((0, 1), "left"),
            ((4, 1), "right"),
            ((2, 0), "up"),
            ((2, 2), "down"),
        ]
        for start, action in cases:
            with self.subTest(start=start, action=action):
                habitat = Habitat(5, 3, start, [])
                self.assertFalse(habitat.move(action))
                self.assertEqual(habitat.agent, start)

    def test_corner_blocks_two_directions(self):
        habitat = Habitat(5, 3, (4, 2), [])
        self.assertFalse(habitat.move("right"))
        self.assertFalse(habitat.move("down"))
        self.assertEqual(habitat.agent, (4, 2))

    def test_one_square_habitat_blocks_every_action(self):
        habitat = Habitat(1, 1, (0, 0), [])
        for action in ACTIONS:
            with self.subTest(action=action):
                self.assertFalse(habitat.move(action))
        self.assertEqual(habitat.agent, (0, 0))

    def test_opposite_moves_return_to_start(self):
        habitat = Habitat(5, 3, (2, 1), [])
        for there, back in [("up", "down"), ("left", "right")]:
            habitat.move(there)
            habitat.move(back)
            self.assertEqual(habitat.agent, (2, 1))

    def test_can_walk_the_whole_grid(self):
        # Snaking right, down, left, down, right must visit every square. Each
        # row uses a fixed number of moves, so a broken wall check fails this
        # test instead of looping forever.
        habitat = Habitat(4, 3, (0, 0), [])
        visited = {habitat.agent}
        for row in range(3):
            direction = "right" if row % 2 == 0 else "left"
            for _ in range(3):
                habitat.move(direction)
                visited.add(habitat.agent)
            if row < 2:
                habitat.move("down")
                visited.add(habitat.agent)
        self.assertEqual(visited, {(x, y) for x in range(4) for y in range(3)})

    def test_unknown_action_raises_value_error(self):
        habitat = Habitat(5, 3, (2, 1), [])
        for action in ["jump", "", "UP", "north"]:
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    habitat.move(action)

    def test_unknown_action_leaves_agent_in_place(self):
        habitat = Habitat(5, 3, (2, 1), [])
        with self.assertRaises(ValueError):
            habitat.move("jump")
        self.assertEqual(habitat.agent, (2, 1))

    def test_moving_does_not_change_food(self):
        habitat = Habitat(5, 3, (2, 1), [(0, 0), (4, 2)])
        for action in ["up", "left", "left", "down", "down"]:
            habitat.move(action)
        self.assertEqual(habitat.food, [(0, 0), (4, 2)])

    def test_actions_table_has_exactly_four_unit_moves(self):
        self.assertEqual(set(ACTIONS), {"up", "down", "left", "right"})
        for dx, dy in ACTIONS.values():
            self.assertEqual(abs(dx) + abs(dy), 1)



class TestHabitatEat(unittest.TestCase):
    """Verifies eating."""

    def test_step_3c_sequence(self):
        habitat = Habitat(4, 1, (0, 0), [(1, 0), (3, 0)])
        self.assertFalse(habitat.eat())
        self.assertEqual(habitat.food, [(1, 0), (3, 0)])
        habitat.move("right")
        self.assertTrue(habitat.eat())
        self.assertEqual(habitat.food, [(3, 0)])
        self.assertFalse(habitat.eat())
        self.assertEqual(habitat.food, [(3, 0)])

    def test_eating_removes_only_the_food_under_syno(self):
        habitat = Habitat(3, 3, (1, 1), [(0, 0), (1, 1), (2, 2)])
        habitat.eat()
        self.assertEqual(habitat.food, [(0, 0), (2, 2)])

    def test_eating_with_no_food_anywhere(self):
        habitat = Habitat(3, 3, (1, 1), [])
        self.assertFalse(habitat.eat())
        self.assertEqual(habitat.food, [])

    def test_eating_does_not_move_syno(self):
        habitat = Habitat(3, 3, (1, 1), [(1, 1)])
        habitat.eat()
        self.assertEqual(habitat.agent, (1, 1))

    def test_eaten_food_disappears_from_render(self):
        habitat = Habitat(3, 1, (0, 0), [(1, 0)])
        habitat.move("right")
        habitat.eat()
        habitat.move("right")
        self.assertEqual(habitat.render(), "..S")

    def test_eat_all_food(self):
        habitat = Habitat(3, 1, (0, 0), [(0, 0), (1, 0), (2, 0)])
        eaten = 0
        for _ in range(3):
            eaten += habitat.eat()
            habitat.move("right")
        self.assertEqual(eaten, 3)
        self.assertEqual(habitat.food, [])

    def test_eating_does_not_affect_callers_food_list(self):
        food = [(0, 0)]
        habitat = Habitat(2, 1, (0, 0), food)
        habitat.eat()
        self.assertEqual(food, [(0, 0)])



class TestHabitatSense(unittest.TestCase):
    """Verifies what SYNO perceives."""

    def test_step_3d_example(self):
        habitat = Habitat(4, 3, (0, 0), [(1, 0), (1, 1)])
        self.assertEqual(
            habitat.sense(1),
            [-1.0, -1.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        )

    def test_output_size_is_always_the_window_area(self):
        # The size must not change near walls, so the brain always receives
        # the same number of inputs.
        for agent in [(0, 0), (2, 1), (4, 2)]:
            habitat = Habitat(5, 3, agent, [])
            for radius in range(4):
                with self.subTest(agent=agent, radius=radius):
                    self.assertEqual(len(habitat.sense(radius)), (2 * radius + 1) ** 2)

    def test_radius_zero_sees_only_its_own_square(self):
        self.assertEqual(Habitat(3, 3, (1, 1), []).sense(0), [0.0])
        self.assertEqual(Habitat(3, 3, (1, 1), [(1, 1)]).sense(0), [1.0])

    def test_window_is_ordered_row_by_row(self):
        # Food directly above SYNO is the 2nd value; directly to the left is
        # the 4th; directly to the right is the 6th; directly below is the 8th.
        cases = {(1, 0): 1, (0, 1): 3, (2, 1): 5, (1, 2): 7}
        for food, index in cases.items():
            with self.subTest(food=food):
                senses = Habitat(3, 3, (1, 1), [food]).sense(1)
                self.assertEqual(senses[index], 1.0)
                self.assertEqual(senses.count(1.0), 1)

    def test_center_of_open_grid_sees_no_walls(self):
        senses = Habitat(5, 5, (2, 2), []).sense(2)
        self.assertEqual(senses, [0.0] * 25)

    def test_each_wall_appears_on_the_correct_side(self):
        # On a 1x1 grid, every square except SYNO's own is a wall.
        senses = Habitat(1, 1, (0, 0), []).sense(1)
        self.assertEqual(senses, [-1.0] * 4 + [0.0] + [-1.0] * 4)

    def test_far_wall_is_seen_only_within_radius(self):
        habitat = Habitat(5, 1, (0, 0), [])
        # Radius 1 sees the left wall but not the right wall 5 squares away.
        middle_row = habitat.sense(1)[3:6]
        self.assertEqual(middle_row, [-1.0, 0.0, 0.0])

    def test_food_outside_the_window_is_not_seen(self):
        habitat = Habitat(9, 1, (0, 0), [(5, 0)])
        self.assertNotIn(1.0, habitat.sense(2))
        self.assertIn(1.0, habitat.sense(5))

    def test_values_are_only_minus_one_zero_or_one(self):
        habitat = Habitat(4, 4, (1, 2), [(0, 0), (3, 3), (2, 2)])
        for value in habitat.sense(3):
            self.assertIn(value, (-1.0, 0.0, 1.0))
            self.assertIsInstance(value, float)

    def test_senses_follow_syno_after_moving(self):
        habitat = Habitat(3, 1, (0, 0), [(2, 0)])
        self.assertEqual(habitat.sense(1)[3:6], [-1.0, 0.0, 0.0])
        habitat.move("right")
        self.assertEqual(habitat.sense(1)[3:6], [0.0, 0.0, 1.0])

    def test_eaten_food_is_no_longer_sensed(self):
        habitat = Habitat(3, 3, (1, 1), [(1, 1)])
        habitat.eat()
        self.assertEqual(habitat.sense(0), [0.0])

    def test_negative_radius_raises_value_error(self):
        with self.assertRaises(ValueError):
            Habitat(3, 3, (1, 1), []).sense(-1)

    def test_sensing_does_not_change_the_habitat(self):
        habitat = Habitat(3, 3, (1, 1), [(0, 0)])
        habitat.sense(2)
        self.assertEqual(habitat.agent, (1, 1))
        self.assertEqual(habitat.food, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
