# ── SYNO · syno/tools/recorder.py ───────────────────────
# Records what happens during a run and saves it as CSV
# Laksheth Vijayakumar · 2026-10-07 · GPL-3.0
# ────────────────────────────────────────────────────────

import csv

class Recorder:
    """
    Collects rows of data during a run, in order, and saves them as a CSV
    file that opens in any spreadsheet program.

    Every row must contain exactly the recorder's fields, so a missing or
    misspelled field fails immediately instead of shifting values into the
    wrong column.
    """

    def __init__(self, fields: list[str]):
        """
        Initializes an empty recorder.

        :param fields: The column names every row must contain, in CSV order.
        :raises ValueError: If fields is empty.
        """
        if not fields:
            raise ValueError("The fields list cannot be empty.")
        self.fields = list(fields)
        self.rows = []

    def record(self, row: dict[str, float]) -> None: 
        """
        Adds one row of data.

        :param row: A value for every field, keyed by field name.
        :raises ValueError: If the row's keys are not exactly the recorder's fields.
        """
        if set(row) != set(self.fields):
            raise ValueError(f"Mismatched row keys. Expected fields {self.fields}, but got row keys {list(row.keys())}.")
        # The row is copied, so later changes to the caller's dictionary
        # cannot alter recorded data.
        self.rows.append(dict(row))

    def column(self, field: str) -> list[float]:
        """
        Returns every recorded value of one field, in the order recorded.

        :param field: The field to read.
        :return: One value per recorded row.
        :raises ValueError: If the field is not tracked by this recorder.
        """
        if field not in self.fields:
            raise ValueError(f"Field '{field}' is not tracked by this recorder.")
        return [r[field] for r in self.rows]

    def save(self, path: str) -> None:
        """
        Writes all rows to a CSV file, with the field names as the header.

        :param path: Where to write the file. An existing file is replaced.
        """
        # The with block closes the file even if writing fails. newline=""
        # is required by the csv module so rows are not split by extra blank
        # lines on some systems.
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
