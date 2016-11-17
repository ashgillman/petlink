"""Tools for parsing PETLINK (TM).

PETLINK is a trademark of Siemens. Siemens does not endorse, nor is
affiliated with this work.
"""

import numpy as np
from .constants import TAGS
DTYPE = np.uint32


def count(events, tag):
    """Count the total number of events of a given tag.

    Inputs:
    - events: Numpy array-like listing PETLINK (TM) events.
    - tag: String describing which tag to look up. See constants.TAGS.
    """
    lower = TAGS[tag]
    upper = TAGS[list(TAGS)[list(TAGS).index(tag) + 1]]
    return int(_mask_events_in_range(events, lower, upper).sum())


def _mask_events_in_range(events, lower, upper):
    """Make a binary mask representing which events correspond to a
    given range.

    Inputs:
    - events: Numpy array-like listing PETLINK (TM) events.
    - lower: Integer describing lower bound (inclusive) of range.
    - upper: Integer describing upper bound (exclusive) of range.
    """
    return (lower <= events) & (events < upper)
