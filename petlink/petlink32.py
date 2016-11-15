"""Tools for parsing PETLINK."""

import numpy as np
from .constants import TAGS
DTYPE = np.uint32


def count(events: np.array, tag: str) -> int:
    lower = TAGS[tag]
    upper = TAGS[
        list(TAGS)[list(TAGS).index(tag) + 1]]
    return _count_events_in_range(events, lower, upper)


def _count_events_in_range(events: np.array, lower: DTYPE,
                           upper: DTYPE) -> int:
    count = 0
    for event in events:
        if lower <= event < upper:
            count += 1
    return count
