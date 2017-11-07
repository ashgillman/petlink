#cython: language_level=3
#cython: linetrace=True, binding=True

include 'common.pxi'

cimport cython
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport ceil, cos, exp2
cimport numpy as np
import numpy as np

import logging
import tempfile

def find_time_index(np.ndarray[CPacket, ndim=1] lm, CPacket time):
    return _find_time_index(lm, lm.size, time)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def unlist(np.ndarray[CPacket, ndim=1] lm not None, shape not None,
           bool tof=True, bar=None):
    """Unlist list mode data into a sinogram.

    Args:
        lm: 1D Numpy array containing LM events.
        shape: Tuple containing sinogram unlisted shape.
        tof: (default True) Include a time-of-flight dimension in the Sinogram.
        bar: (optional) A progress bar to update.

    Returns:
        The prompt and delay sinograms.
    """
    assert len(shape) == 4
    for d in shape:
        assert d > 0

    cdef:
        CListIdx pidx, lastthrottle = 0, THROTTLE = 1000
        CPacket packet
        CSinoIdxElem E = shape[0]
        CSinoIdxElem A = shape[1]
        CSinoIdxElem S = shape[2]
        CSinoIdxElem T = shape[3]

    if tof:
        unlist_shape = shape # E, A, S, T
    else:
        unlist_shape = shape[:3] # E, A, S

    try:
        psino = np.zeros(unlist_shape, dtype=NpCount)
        dsino = np.zeros(unlist_shape, dtype=NpCount)
    except MemoryError:
        # out of memory... Let's revert to a memory-mapped file
        with tempfile.TemporaryFile() as pfile:
            psino = np.memmap(pfile, mode='w+',
                              shape=unlist_shape, dtype=NpCount)
        with tempfile.TemporaryFile() as dfile:
            dsino = np.memmap(dfile, mode='w+',
                              shape=unlist_shape, dtype=NpCount)

    for pidx in range(lm.size):
        packet = lm[pidx]
        # accumulate counts
        if is_event(packet):
            if tof:
                bin_event(dsino if is_event_delay(packet) else psino,
                          packet, E, A, S, T, n_axials=S, negate_delays=False)
            else:
                bin_event_no_tof(dsino if is_event_delay(packet) else psino,
                                 packet, E, A, S, T, n_axials=S,
                                 negate_delays=False)

        elif pidx > lastthrottle + THROTTLE:
            # Check if user C-c'ed
            PyErr_CheckSignals()

            lastthrottle = pidx
            if bar is not None:
                bar.update(pidx)

    # convert back to real Numpy
    return tuple(np.array(sino, dtype=NpCount) for sino in (psino, dsino))


@cython.cdivision(True)
#@cython.boundscheck(False)
@cython.wraparound(False)
def gated_unlist(
        np.ndarray[CPacket, ndim=1] lm not None, shape not None,
        np.ndarray[unsigned short, ndim=1] gates, unsigned short n_gates,
        CTimeIdx time_resolution, bool tof=True,
        bar=None):
    """Unlist list mode data into a sinograms based on a gating signal.

    Args:
        lm: 1D Numpy array containing LM events.
        shape: Tuple containing sinogram unlisted shape.
        gates: 1D Numpy array containing gates.
        n_gates: Number of gates.
        time_resolution: Number of ms each time point should cover.
        tof: (default True) Include a time-of-flight dimension in the Sinogram.
        bar: (optional) A progress bar to update.

    Returns:
        A list of 2-tuples containing prompt and delay sinograms for each gate.
    """
    assert len(shape) == 4
    for d in shape:
        assert d > 0
    assert len(gates) > 0
    assert n_gates > 0

    cdef:
        CListIdx gidx, pidx, lastbar = 0, THROTTLE = 10000
        CListIdx lm_size = lm.size
        CPacket packet, time
        CSinoIdxElem E = shape[0]
        CSinoIdxElem A = shape[1]
        CSinoIdxElem S = shape[2]
        CSinoIdxElem T = shape[3]
        CTimeIdx time_offset = begin(lm)
        CTimeIdx gate_idx = time_offset
        CTimeIdx gate_length = gates.size
        unsigned short gate = gates[0]

    logger = logging.getLogger(__name__)

    if tof:
        unlist_shape = (n_gates, ) + shape # n_gates, E, A, S, T
    else:
        unlist_shape = (n_gates, ) + shape[:3] + (1, ) # n_gates, E, A, S, _

    try:
        logger.debug('Attempting to allocate memory for sinograms.')
        use_mmap = False
        psinos = np.zeros(unlist_shape, dtype=NpCount)
        dsinos = np.zeros(unlist_shape, dtype=NpCount)
        logger.debug('Memory allocated.')
    except MemoryError:
        logger.info('Out of memory, using memmaped unlist.')
        use_mmap = True
        # Create a temp file, and memmap from it, then unlink the tempfile
        # memmap open file will keep the unlinked file existing until it is
        # destroyed (and therefore the file closed).
        # https://stackoverflow.com/questions/44691030/numpy-memmap-with-file-deletion
        with tempfile.NamedTemporaryFile() as pfile:
            psinos = np.memmap(pfile.name, mode='w+',
                               shape=unlist_shape, dtype=NpCount)
        with tempfile.NamedTemporaryFile() as dfile:
            dsinos = np.memmap(dfile.name, mode='w+',
                               shape=unlist_shape, dtype=NpCount)

    logger.debug('Beginning unlisting')
    for pidx in range(lm_size):
        packet = lm[pidx]
        # accumulate counts
        if is_event(packet):
            if tof:
                bin_event((dsinos if is_event_delay(packet)
                           else psinos)[gate, ...],
                          packet, E, A, S, T, n_axials=S, negate_delays=False)
            else:
                bin_event_no_tof((dsinos if is_event_delay(packet)
                                  else psinos)[gate, ..., 0],
                                 packet, E, A, S, T, n_axials=S,
                                 negate_delays=False)

        elif is_tag_time(packet):
            # Check if user C-c'ed
            PyErr_CheckSignals() # do this here to throttle checking

            time = packet & TAG_UNMASK
            gate_idx = (time - time_offset) // time_resolution
            if gate_idx >= gate_length:
                logger.warn('Gates finished at %s of %s', time, duration(lm))
                break
            gate = gates[gate_idx]

            if bar is not None and pidx > lastbar + THROTTLE:
                lastbar = pidx
                bar.update(pidx)

    if gate_idx < gate_length:
        logger.warn('LM finished at %s of %s', time,
                    time_offset + gate_length * time_resolution)

    logger.debug('Unlisted %i prompts and %i delays.',
                 psinos.sum(), dsinos.sum())
    # convert back to real Numpy
    ret = []
    for gidx in range(n_gates):
        if use_mmap:
            # if mem-mapped, create new file for each gate
            gate_sinos = []
            for sinos in (psinos, dsinos):
                with tempfile.NamedTemporaryFile() as fp:
                    sino = np.memmap(fp.name, mode='w+',
                                     shape=sinos.shape, dtype=psinos.dtype)
                    sino[:] = sinos[gidx, ...]
                    gate_sinos.append(sino)
            ret.append(tuple(gate_sinos))
        else:
            # just return array for each gate if fits in memory
            ret.append(tuple(np.array(sinos[gidx, ...], dtype=NpCount)
                             for sinos in (psinos, dsinos)))
    return tuple(ret)
