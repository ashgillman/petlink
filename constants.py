"""Constants for dealing with Siemens PETLINK 32 bit format.
"""

from collections import OrderedDict
import numpy as np

PL_DTYPE = np.uint32
TAGS = OrderedDict((
    ('delay',    0b0000 << 28),
    ('prompt',   0b0100 << 28),
    ('time',     0b1000 << 28),
    ('deadtime', 0b1010 << 28),
    ('gantry',   0b1100 << 28),
    ('patient',  0b1110 << 28),
    ('control',  0b1111 << 28),
    ('__LAST__', 2 ** 32 - 1 ))) # Ignore 1111...1111
