"""
Unit tests for syno.tools.plot.

These tests are intentionally stricter than line_chart's own safeguards. They
parse every saved file as XML, so a chart that a browser could not draw fails
the test. They also verify the exact pixel positions of the step 6b example,
the flipped y-axis, flat data, and titles containing special characters.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import os
import tempfile
import unittest
import xml.dom.minidom

from syno.tools.plot import HEIGHT, MARGIN, WIDTH, line_chart

SVG_NAMESPACE = "http://www.w3.org/2000/svg"


class TestLineChart(unittest.TestCase):
    """Verifies the saved SVG chart."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "chart.svg")

    def tearDown(self):
        self.directory.cleanup()

    def draw(self, values: list[float], title: str = "Test chart") -> xml.dom.minidom.Document:
        """Draws a chart and parses the saved file as XML."""
        line_chart(values, self.path, title)
        return xml.dom.minidom.parse(self.path)

    def points(self, document: xml.dom.minidom.Document) -> list[tuple[float, float]]:
        """Reads the polyline's points back as numbers."""
        text = document.getElementsByTagName("polyline")[0].getAttribute("points")
        return [tuple(float(n) for n in pair.split(",")) for pair in text.split()]

    def test_step_6b_points(self):
        document = self.draw([0.2, 0.5, 0.3, 0.9])
        polyline = document.getElementsByTagName("polyline")[0]
        self.assertEqual(polyline.getAttribute("points"), "40.0,260.0 213.3,165.7 386.7,228.6 560.0,40.0")

    def test_file_is_svg_with_the_exact_namespace(self):
        document = self.draw([0.2, 0.5])
        root = document.documentElement
        self.assertEqual(root.tagName, "svg")
        self.assertEqual(root.getAttribute("xmlns"), SVG_NAMESPACE)
        self.assertEqual(root.getAttribute("width"), str(WIDTH))
        self.assertEqual(root.getAttribute("height"), str(HEIGHT))

    def test_first_and_last_points_touch_the_margins(self):
        points = self.points(self.draw([3.0, 7.0, 5.0]))
        self.assertEqual(points[0][0], MARGIN)
        self.assertEqual(points[-1][0], WIDTH - MARGIN)

    def test_lowest_value_is_at_the_bottom_and_highest_at_the_top(self):
        points = self.points(self.draw([5.0, 1.0, 9.0]))
        self.assertEqual(points[1][1], HEIGHT - MARGIN)
        self.assertEqual(points[2][1], MARGIN)

    def test_larger_values_are_drawn_higher(self):
        points = self.points(self.draw([1.0, 2.0, 3.0, 4.0]))
        ys = [y for _, y in points]
        self.assertEqual(ys, sorted(ys, reverse=True))

    def test_points_are_evenly_spaced(self):
        points = self.points(self.draw([0.0] * 5 + [1.0]))
        gaps = [b[0] - a[0] for a, b in zip(points, points[1:])]
        for gap in gaps:
            self.assertAlmostEqual(gap, gaps[0], places=0)

    def test_every_point_stays_inside_the_margins(self):
        points = self.points(self.draw([-50.0, 3.0, 1e6, 0.0, -2.5]))
        for x, y in points:
            self.assertTrue(MARGIN <= x <= WIDTH - MARGIN)
            self.assertTrue(MARGIN <= y <= HEIGHT - MARGIN)

    def test_one_point_per_value(self):
        self.assertEqual(len(self.points(self.draw([0.1] * 37 + [0.2]))), 38)

    def test_flat_data_draws_a_flat_line_without_dividing_by_zero(self):
        points = self.points(self.draw([0.5, 0.5, 0.5]))
        self.assertEqual({y for _, y in points}, {HEIGHT - MARGIN})

    def test_labels_show_the_highest_and_lowest_values(self):
        texts = [node.firstChild.data for node in self.draw([0.2, 0.9]).getElementsByTagName("text")]
        self.assertIn("Test chart", texts)
        self.assertIn("0.90", texts)
        self.assertIn("0.20", texts)

    def test_title_with_special_characters_is_valid_svg(self):
        # & and < have special meanings in XML. Unescaped, they make the file
        # invalid, and browsers fail to draw it.
        document = self.draw([1.0, 2.0], "Energy & <food>")
        texts = [node.firstChild.data for node in document.getElementsByTagName("text")]
        self.assertIn("Energy & <food>", texts)

    def test_fewer_than_two_values_raise_value_error(self):
        for values in ([], [1.0]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    line_chart(values, self.path, "Test")


if __name__ == "__main__":
    unittest.main()
