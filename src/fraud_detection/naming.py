"""Helpers for project naming conventions."""

import re
from pathlib import Path

SEED_PATTERN = re.compile(r"_seed_(\d+)")


def seed_from_filename(filename: str | Path) -> int:
    """Extract the numeric seed from standard filenames.

    Filenames must include a substring of the form ``_seed_<number>``.
    """
    match = SEED_PATTERN.search(str(filename))
    if not match:
        raise ValueError(
            f"Expected a filename containing '_seed_<number>', got: {filename}"
        )
    return int(match.group(1))
