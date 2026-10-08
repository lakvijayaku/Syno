"""
Unit tests for syno.tools.recorder.

These tests are intentionally stricter than the Recorder's own safeguards.
They verify that rows are kept in order, that rows with missing or extra
fields fail loudly, that recorded data cannot be changed from outside, and
that saved files can be read back exactly.

Run from the repository root with:
    python3 -m unittest discover tests -v
"""

import csv
import os
import tempfile
import unittest

from syno.tools.recorder import Recorder


class TestRecorderRecord(unittest.TestCase):
    """Verifies recording and reading rows."""

    def test_step_6a_example(self):
        recorder = Recorder(["life", "energy"])
        recorder.record({"life": 1, "energy": 0.2})
        recorder.record({"life": 2, "energy": 0.5})
        self.assertEqual(recorder.column("energy"), [0.2, 0.5])
        self.assertEqual(recorder.column("life"), [1, 2])

    def test_new_recorder_has_no_rows(self):
        recorder = Recorder(["a"])
        self.assertEqual(recorder.rows, [])
        self.assertEqual(recorder.column("a"), [])

    def test_rows_keep_their_order(self):
        recorder = Recorder(["step"])
        for step in range(10):
            recorder.record({"step": step})
        self.assertEqual(recorder.column("step"), list(range(10)))

    def test_key_order_in_a_row_does_not_matter(self):
        recorder = Recorder(["life", "energy"])
        recorder.record({"energy": 0.4, "life": 3})
        self.assertEqual(recorder.column("life"), [3])

    def test_missing_field_raises_value_error(self):
        recorder = Recorder(["life", "energy"])
        with self.assertRaises(ValueError):
            recorder.record({"life": 3})

    def test_extra_field_raises_value_error(self):
        recorder = Recorder(["life", "energy"])
        with self.assertRaises(ValueError):
            recorder.record({"life": 3, "energy": 0.1, "mood": 0.9})

    def test_misspelled_field_raises_value_error(self):
        recorder = Recorder(["life", "energy"])
        with self.assertRaises(ValueError):
            recorder.record({"life": 3, "enrgy": 0.1})

    def test_rejected_row_is_not_recorded(self):
        recorder = Recorder(["life", "energy"])
        with self.assertRaises(ValueError):
            recorder.record({"life": 3})
        self.assertEqual(recorder.rows, [])

    def test_unknown_column_raises_value_error(self):
        recorder = Recorder(["life", "energy"])
        with self.assertRaises(ValueError):
            recorder.column("mood")

    def test_empty_fields_raise_value_error(self):
        with self.assertRaises(ValueError):
            Recorder([])


class TestRecorderIsolation(unittest.TestCase):
    """Verifies that recorded data cannot be changed from outside."""

    def test_changing_original_fields_does_not_affect_recorder(self):
        fields = ["a"]
        recorder = Recorder(fields)
        fields.append("b")
        self.assertEqual(recorder.fields, ["a"])

    def test_changing_a_row_after_recording_does_not_affect_recorder(self):
        recorder = Recorder(["a"])
        row = {"a": 1}
        recorder.record(row)
        row["a"] = 99
        self.assertEqual(recorder.column("a"), [1])

    def test_reusing_one_dictionary_for_many_rows(self):
        # A common pattern in loops: one dictionary updated every step.
        recorder = Recorder(["step"])
        row = {"step": 0}
        for step in range(3):
            row["step"] = step
            recorder.record(row)
        self.assertEqual(recorder.column("step"), [0, 1, 2])


class TestRecorderSave(unittest.TestCase):
    """Verifies the saved CSV file."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "run.csv")

    def tearDown(self):
        self.directory.cleanup()

    def test_saved_file_matches_step_6a_example(self):
        recorder = Recorder(["life", "energy"])
        recorder.record({"life": 1, "energy": 0.2})
        recorder.record({"life": 2, "energy": 0.5})
        recorder.save(self.path)
        with open(self.path, newline="") as file:
            self.assertEqual(list(csv.reader(file)), [["life", "energy"], ["1", "0.2"], ["2", "0.5"]])

    def test_columns_follow_the_field_order(self):
        recorder = Recorder(["b", "a"])
        recorder.record({"a": 1, "b": 2})
        recorder.save(self.path)
        with open(self.path, newline="") as file:
            self.assertEqual(list(csv.reader(file)), [["b", "a"], ["2", "1"]])

    def test_saved_values_read_back_exactly(self):
        recorder = Recorder(["value"])
        values = [0.1, 1 / 3, -2.5e-7, 123456.789]
        for value in values:
            recorder.record({"value": value})
        recorder.save(self.path)
        with open(self.path, newline="") as file:
            read = [float(row["value"]) for row in csv.DictReader(file)]
        self.assertEqual(read, values)

    def test_empty_recorder_saves_only_the_header(self):
        Recorder(["life", "energy"]).save(self.path)
        with open(self.path, newline="") as file:
            self.assertEqual(list(csv.reader(file)), [["life", "energy"]])

    def test_saving_twice_replaces_the_file(self):
        recorder = Recorder(["a"])
        recorder.record({"a": 1})
        recorder.save(self.path)
        recorder.save(self.path)
        with open(self.path, newline="") as file:
            self.assertEqual(len(list(csv.reader(file))), 2)


if __name__ == "__main__":
    unittest.main()
