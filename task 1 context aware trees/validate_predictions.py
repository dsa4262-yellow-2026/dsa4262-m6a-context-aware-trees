"""Validate prediction CSVs and optionally compare example scores."""

import argparse
import csv
import gzip
import json
import math
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

COLUMNS = ["transcript_id", "transcript_position", "score"]


def open_text(path):
    with Path(path).open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    return (gzip.open if compressed else open)(
        path, "rt", encoding="utf-8-sig", newline=""
    )


def site_keys(path):
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                for transcript, positions in json.loads(line).items():
                    for position in positions:
                        yield transcript, position


def validate_predictions(data, predictions, expected=None):
    """Require exact input keys/order, finite probabilities and the CSV schema."""
    with ExitStack() as stack:
        scores = csv.DictReader(stack.enter_context(open_text(predictions)))
        reference = (
            csv.DictReader(stack.enter_context(open_text(expected)))
            if expected is not None
            else None
        )
        for reader in [scores] + ([] if reference is None else [reference]):
            if reader.fieldnames != COLUMNS:
                raise ValueError(
                    "CSV header must be transcript_id,transcript_position,score."
                )
        count = 0
        for count, (key, row) in enumerate(zip_longest(site_keys(data), scores), 1):
            if key is None or row is None:
                raise ValueError("Input and prediction row counts differ.")
            if None in row or (row[COLUMNS[0]], row[COLUMNS[1]]) != key:
                raise ValueError(
                    f"Prediction row {count} does not preserve the input coordinates/order."
                )
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {count}: score must be a number.") from error
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(
                    f"Row {count}: score must be finite and between 0 and 1."
                )
            if reference is not None:
                target = next(reference, None)
                if target is None or (target[COLUMNS[0]], target[COLUMNS[1]]) != key:
                    raise ValueError(
                        f"Expected-result coordinates differ at row {count}."
                    )
                if not math.isclose(
                    score, float(target["score"]), rel_tol=1e-6, abs_tol=1e-7
                ):
                    raise ValueError(
                        f"Row {count}: score differs from the expected example result."
                    )
        if count == 0:
            raise ValueError("The input contains no sites.")
        if reference is not None and next(reference, None) is not None:
            raise ValueError("Expected-result file has extra rows.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Original JSON or JSON.gz input."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction CSV or CSV.gz to check.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="Expected CSV scores for the identical example input.",
    )
    args = parser.parse_args()
    try:
        count = validate_predictions(args.data, args.predictions, args.expected)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"FAIL: {error}\n")
    result = f"PASS: {count:,} sites; exact columns, input coordinates/order, finite scores in [0,1]"
    if args.expected:
        result += "; expected scores match"
    print(result + ".")


if __name__ == "__main__":
    main()
