"""Constants for dealing with Siemens PETLINK 32 bit format.
"""

from collections import OrderedDict
import numpy as np

KiB = 1024
MiB = 1024 * 1024

# PETLINK
PL_DTYPE = np.uint32
TAGS = OrderedDict([
    ('delay',    0b0000 << 28),
    ('prompt',   0b0100 << 28),
    ('time',     0b1000 << 28),
    ('deadtime', 0b1010 << 28),
    ('gantry',   0b1100 << 28),
    ('patient',  0b1110 << 28),
    ('control',  0b1111 << 28),
    ('__LAST__', 2 ** 32 - 1)])  # Ignore 1111...1111

# DICOM
DCM_N_ZEROS_BEFORE_MAGIC = 128
DCM_MAGIC = b'\x00' * DCM_N_ZEROS_BEFORE_MAGIC + b'DICM'
DCM_CSA_DATA_INFO = (0x0029, 0x1010)
# DCM_CSA_IMAGE_HEADER_INFO = (0x0029, 0x1110)
# DCM_CSA_SERIES_HEADER_INFO = (0x0029, 0x1120)
DCM_CSA_DATA = (0x7fe1, 0x1010)

# .ptd
PTD_MAX_DCM_SIZE = 1 * MiB

# Interfile
IFL_MAX_HEADER_SIZE = 1 * MiB  # max expected, to prevent loading data files
IFL_MAGIC = '!INTERFILE'
IFL_MAGIC_END = '!END OF INTERFILE'
IFL_NONE = '<NONE>'
IFL_SEP = ':='
IFL_LIST_START = '{'
IFL_LIST_END = '}'
IFL_UNITS_START = '('
IFL_UNITS_END = ')'
IFL_INDEX_START = '['
IFL_INDEX_END = ']'

IFL_DATA_FILE_KEY = 'name of data file'
IFL_OFFSET_KEY = 'data offset in bytes'
IFL_DATA_FORMAT_KEY = 'number format'
IFL_DATA_FORMAT_DEFAULT = 'UNSIGNED INTEGER'
IFL_DATA_SIZE_KEY = 'number of bytes per pixel'
IFL_DATA_SIZE_DEFAULT = 4
IFL_DATA_ORDER_KEY = 'imagedata byte order'
IFL_DATA_ORDER_DEFAULT = 'LITTLEENDIAN'
