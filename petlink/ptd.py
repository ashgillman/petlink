"""I/O with Siemens' .ptd format. These files are simply a data component and a
DICOM file mashed together.
"""

import os
import shutil
import mmap
import numpy as np
try:
    import pydicom
except ImportError:
    pydicom = None

from petlink.constants import (
    PL_DTYPE, PTD_MAX_DCM_SIZE, DCM_MAGIC, DCM_CSA_DATA)


def _get_start_of_dicom(filename):
    """Find where the DICOM component of the .ptd file begins."""
    with open(filename, 'rb') as fp:
        # map in the file
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        # search for DICM magic from somewhere near the end to be faster
        size = os.fstat(fp.fileno()).st_size
        return mm.find(DCM_MAGIC, max(0, size - PTD_MAX_DCM_SIZE))


def read_data(filename, dtype=PL_DTYPE):
    """Read the raw data component of the .ptd file."""
    length = _get_start_of_dicom(filename)
    event_length = length // PL_DTYPE().itemsize

    # map in everything up to DICM magic
    try:
        return np.memmap(
            filename, mode='r', shape=(event_length, ), dtype=dtype)
    except OverflowError as err:
        raise RuntimeError(
            "Can't load {} as .ptd, is it valid?".format(filename)) from err


def read_dcm(filename):
    """Read the DICOM component of the .ptd file."""
    if not pydicom:
        raise ImportError("Couldn't import pydicom")

    start = _get_start_of_dicom(filename)
    if start < 0:
        raise pydicom.errors.InvalidDicomError('Invalid .ptd, no DICOM magic.')

    # Read everything after DICM magic as a DICOM
    with open(filename, 'rb') as fp:
        fp.seek(start)
        return pydicom.filereader.read_partial(fp, defer_size=10*1024,
                                               stop_when=_at_lm_data)


def _at_lm_data(tag, VR, length):
    """Stop when list mode data is reached. For use in
    `read_partial'.
    """
    return tag == DCM_CSA_DATA


def write_ptd(data, dcm, filename):
    """Write a .ptd file. Assumes the data portion is in the desired data
    type.
    """
    with open(filename, 'wb+') as fp:
        data.tofile(fp)
        dcm.save_as(fp)
