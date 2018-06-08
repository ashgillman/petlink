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


cdef factors(int n):
    """Find the factors of a number n."""
    fs = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            fs.append(i)
            fs.append(n // i)
    return set(fs)


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
        CListIdx pidx, lastthrottle = 0, THROTTLE = 10000
        CPacket packet
        CSinoIdxElem E = shape[0]
        CSinoIdxElem A = shape[1]
        CSinoIdxElem S = shape[2]
        CSinoIdxElem T = shape[3]

    if tof:
        unlist_shape = shape              # E, A, S, T
    else:
        unlist_shape = shape[:3] + (1, )  # E, A, S

    cdef:
        CCount[:, :, :, :] psino = np.zeros(unlist_shape, dtype=NpCount)
        CCount[:, :, :, :] dsino = np.zeros(unlist_shape, dtype=NpCount)
    # except MemoryError:
    #     # out of memory... Let's revert to a memory-mapped file
    #     with tempfile.TemporaryFile() as pfile:
    #         psino = np.memmap(pfile, mode='w+',
    #                           shape=unlist_shape, dtype=NpCount)
    #     with tempfile.TemporaryFile() as dfile:
    #         dsino = np.memmap(dfile, mode='w+',
    #                           shape=unlist_shape, dtype=NpCount)

    for pidx in range(len(lm)):
        packet = lm[pidx]
        # accumulate counts
        if is_event(packet):
            if is_event_delay(packet):
                bin_event(dsino, packet, E, A, S, T, n_axials=S,
                          tof=tof, negate_delays=False)
            else:
                bin_event(psino, packet, E, A, S, T, n_axials=S,
                          tof=tof, negate_delays=False)

        elif pidx > lastthrottle + THROTTLE:
            # Check if user C-c'ed
            PyErr_CheckSignals()

            lastthrottle = pidx
            if bar is not None:
                bar.update(pidx)

    # convert back to real Numpy
    if tof:
        np_psino = np.asarray(psino)
        np_dsino = np.asarray(dsino)
    else:
        np_psino = np.asarray(psino).squeeze()
        np_dsino = np.asarray(dsino).squeeze()
    return np_psino, np_dsino


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
            if is_event_delay(packet):
                bin_event(dsinos[gate, ...], packet, E, A, S, T, n_axials=S,
                          tof=tof, negate_delays=False)
            else:
                bin_event(psinos[gate, ...], packet, E, A, S, T, n_axials=S,
                          tof=tof, negate_delays=False)

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


@cython.cdivision(True)
@cython.cdivision_warnings(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unlist_series_low_res(
        np.ndarray[CPacket, ndim=1] lm, orig_shape, max_shape,
        CSinoIdxElem[:] segments_def, CTimeIdx time_resolution, bar=None):
    """Unlist LM into a low resolution sinogram series."""
    cdef:
        CPacket packet, time
        CListIdx pidx
        CSinoIdxElem E = orig_shape[0]
        CSinoIdxElem A = orig_shape[1]
        CSinoIdxElem S = orig_shape[2]
        CSinoIdxElem T = orig_shape[3]
        CSinoIdxElem n_axials = segments_def[0]
        CSinoIdxElem E_max = max_shape[0]
        CSinoIdxElem A_max = max_shape[1]
        CSinoIdxElem S_max = max_shape[2]
        CTimeIdx begin_time = begin(lm)
        CTimeIdx length = <CTimeIdx>ceil((<double>duration(lm) - begin_time)
                                         / time_resolution)
        CSinoIdxElem e, a, s
        CTimeIdx time_idx = 0, new_time_idx = 0
        CSinoIdxElem E_, A_, S_

    # new shape is each largest factors of original shape less than max shape
    for fact in reversed(sorted(factors(E))):
        if fact <= E_max:
            E_ = fact
            break
    for fact in reversed(sorted(factors(A))):
        if fact <= A_max:
            A_ = fact
            break

    cdef:
        unsigned int E_dec = E // E_
        unsigned int A_dec = A // A_
        # unsigned int S_dec = n_axials // S_
        CCount[:, :, :, :] sino_series = np.zeros((length, E_, A_, n_axials),
                                                  dtype=NpCount)
        CSinoIdxElem[:] ssrb_lut

    assert segments_def[0] == segments_def[1] + 1, 'Span > 1 not yet supported'

    seg_luts = []
    for seg_idx, seg_size in enumerate(segments_def):
        seg_luts.append(np.arange(seg_size) + (seg_idx + 2) // 4)
    ssrb_lut = np.concatenate(seg_luts).astype(NpSinoIdxElem)
    for s in ssrb_lut:
        assert s < n_axials

    # print(ssrb_lut.shape, np.bincount(ssrb_lut))
    # print([x for x in ssrb_lut])
    # print(min_s, max_s, [x for x in ssrb_lut[min_s:max_s]])
    # for x in ssrb_lut:
    #     print(x, end=' ')
    # print()

    # for each event
    for pidx in range(len(lm)):
        packet = lm[pidx]
        # build a sinogram
        if is_event(packet):
            s = get_s(E, A, S, T, packet)
            # only consider segment 0
            # if not (min_s <= s < max_s):
            #     continue
            # s = get_ssrb_axial(s, segments_def)
            assert s < S
            s = ssrb_lut[s]
            e = get_e(E, A, S, T, packet) // E_dec
            a = get_a(E, A, S, T, packet) // A_dec
            # s //= S_dec

            if is_event_delay(packet):
                sino_series[time_idx, e, a, s] -= 1 # delay, -1
            else:
                sino_series[time_idx, e, a, s] += 1 # prompt, +1

        # update time index
        elif is_tag_time(packet):
            time = packet & TAG_UNMASK
            new_time_idx = (time - begin_time) // time_resolution
            assert new_time_idx < length

            if new_time_idx != time_idx: # next time bin
                time_idx = new_time_idx

                PyErr_CheckSignals()
                if bar:
                    bar.update(pidx)

    if bar: bar.finish()

    # quick sensitivity correction
    sensitivity = np.hstack([np.arange(n_axials // 2)+1,
                             np.arange(n_axials // 2)[::-1]+1])
    sino_series = (sino_series
                   * (sensitivity.max() / sensitivity)).astype(NpCount)

    return np.array(sino_series, dtype=NpCount)
