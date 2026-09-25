"""Train/test split by OpenNeuro study.

Whole studies go to one side, never both. Scans from the same study share
scanners, protocols and often subjects, so splitting per scan would let the
test set "leak" into training and flatter the results.

The side is decided by a hash of the study id, not by shuffling a list. So a
study's side never changes, even after downloading more data. Studies added
later simply land on one side or the other.
"""

from __future__ import annotations

import hashlib

TEST_FRACTION = 0.2


def split_of(dataset_id: str, test_fraction: float = TEST_FRACTION, seed: int = 0) -> str:
    """ "train" or "test" for an OpenNeuro study id such as "ds000117"."""
    digest = hashlib.sha256(f"{seed}:{dataset_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    return "test" if u < test_fraction else "train"
