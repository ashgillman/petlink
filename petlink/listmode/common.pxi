"""Common Cython utilities and constants for PETLINK and listmode.
"""

from libc.math cimport exp2
cimport cython
cimport numpy as np
import numpy as np


# Type definitions
ctypedef np.uint32_t   CPacket
ctypedef np.uint16_t   CSinoIdxElem
ctypedef np.uint32_t   CTimeIdx
ctypedef np.int32_t    CCount
ctypedef np.double_t   CSurrogateValue
ctypedef bint          bool
ctypedef unsigned long CListIdx

cdef struct Time_Index:
    CTimeIdx time
    CListIdx idx

NpPacket         = np.uint32 # a PETLINK packet
NpSinoIdxElem    = np.uint16 # a sinogram index
NpTimeIdx        = np.uint32 # a time index
NpCount          = np.int32  # sinogram val. Signed, since delays may make -'ve
NpSurrogateValue = np.double


# Use max values as invalid flag
Packet_invalid      = np.iinfo(NpPacket).max
SinoIdxElem_invalid = np.iinfo(NpSinoIdxElem).max
TimeIdx_invalid     = np.iinfo(NpTimeIdx).max
Count_invalid       = np.iinfo(NpCount).max

# Masks, taken from PETLINK spec
cdef CPacket EVENT_MASK    = 0b0000U << 28
cdef CPacket EVENT_UNMASK  = ~<CPacket>(0b1100U << 28)
cdef CPacket TAG_MASK      = 0b1000U << 28
cdef CPacket TAG_UNMASK    = ~<CPacket>(0b1111U << 28)
cdef CPacket DELAY_MASK    = 0b0000U << 28
cdef CPacket PROMPT_MASK   = 0b0100U << 28
cdef CPacket TIME_MASK     = 0b1000U << 28
cdef CPacket DEADTIME_MASK = 0b1010U << 28
cdef CPacket GANTRY_MASK   = 0b1100U << 28
cdef CPacket PATIENT_MASK  = 0b1110U << 28
cdef CPacket CONTROL_MASK  = 0b1111U << 28

cdef double PI = 3.141592654
cdef double TWO_PI = 2 * PI


# inspect packet
@cython.profile(False)
cdef inline bool is_event(CPacket packet) nogil:
    """Is a packet an event?"""
    return packet < TAG_MASK

@cython.profile(False)
cdef inline bool is_event_delay(CPacket event) nogil:
    """Given a packet is an event, is it a delay?"""
    return event < PROMPT_MASK

@cython.profile(False)
cdef inline bool is_tag_time(CPacket tag) nogil:
    """Given a packet is a tag, is it a time?"""
    return tag < DEADTIME_MASK

@cython.profile(False)
cdef inline bool is_time(CPacket packet) nogil:
    """Is a packet a time tag?"""
    return TIME_MASK <= packet < DEADTIME_MASK


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Time_Index find_next_time(np.ndarray[CPacket, ndim=1] lm,
                                      CListIdx lm_size, CListIdx idx):
    """Find the next instance of a time tag after an index, idx."""
    cdef:
        CTimeIdx packet
        Time_Index retval
    retval.time = TimeIdx_invalid
    retval.idx = lm_size

    while idx < lm_size:
        packet = lm[idx]
        if is_time(packet):
            retval.time = packet & TAG_UNMASK
            retval.idx = idx
            return retval
        idx += 1
    return retval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Time_Index find_prev_time(np.ndarray[CPacket, ndim=1] lm,
                                      CListIdx lm_size, CListIdx idx):
    """Find the previous instance of a time tag before an index, idx."""
    cdef:
        CTimeIdx packet
        Time_Index retval
    retval.time = TimeIdx_invalid
    retval.idx = 0

    while idx > 0:
        packet = lm[idx]
        if is_time(packet):
            retval.time = packet & TAG_UNMASK
            retval.idx = idx
            return retval
        idx -= 1
    return retval


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline CTimeIdx duration(np.ndarray[CPacket, ndim=1] lm):
    """Calcultate the duration, in ms, of the acquisition."""
    cdef CTimeIdx packet
    cdef CListIdx idx = lm.size
    cdef CTimeIdx dur = find_prev_time(lm, lm.size, lm.size-1).time
    if dur == TimeIdx_invalid:
        dur = 0
    return dur

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline CTimeIdx begin(np.ndarray[CPacket, ndim=1] lm):
    """Calcultate the begin time, in ms, of the acquisition."""
    return find_next_time(lm, lm.size, 0).time


# The following functions get the bin address (element, angle,
# segment, ToF bin) from a bin index.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline CSinoIdxElem get_e(CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S,
                               CSinoIdxElem T, CPacket event) nogil:
    return (event & EVENT_UNMASK) % E

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline CSinoIdxElem get_a(CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S,
                               CSinoIdxElem T, CPacket event) nogil:
    return (event & EVENT_UNMASK) // E % A

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline CSinoIdxElem get_s(CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S,
                               CSinoIdxElem T, CPacket event) nogil:
    return (event & EVENT_UNMASK) // (E * A) % S

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline CSinoIdxElem get_t(CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S,
                               CSinoIdxElem T, CPacket event) nogil:
    return (event & EVENT_UNMASK) // (E * A * S) % T


def extract_one(event, shape):
    """Extract the sinogram index of an event."""
    E, A, S, T = shape
    return (get_e(E, A, S, T, event), get_a(E, A, S, T, event),
            get_s(E, A, S, T, event), get_t(E, A, S, T, event))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void bin_event(
    CCount[:, :, :, :] sino, CPacket packet,
    CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S, CSinoIdxElem T,
    CSinoIdxElem n_axials, bool tof=False, bool negate_delays=True) nogil:
    """Bin an event into a sinogram. Delays are negative if negate_delays else
    positive."""
    # only consider segment 0?
    cdef CSinoIdxElem s = get_s(E, A, S, T, packet)
    if s >= n_axials:
        return

    cdef:
        CSinoIdxElem e = get_e(E, A, S, T, packet)
        CSinoIdxElem a = get_a(E, A, S, T, packet)
        CSinoIdxElem t = 0

    if tof:
        t = get_t(E, A, S, T, packet)

    if negate_delays and is_event_delay(packet): # is delay packet
        sino[e, a, s, t] -= 1
    else: # is event packet or delay but not negating
        sino[e, a, s, t] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void bin_event_no_tof(
    CCount[:, :, :] sino, CPacket packet,
    CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem S, CSinoIdxElem T,
    CSinoIdxElem n_axials, bool negate_delays=True):
    """Bin an event into a sinogram. Delays are negative if negate_delays else
    positive."""
    # if optional n_axials isn't set, use all segments
    cdef CPacket event = packet & EVENT_UNMASK

    # only consider segment 0?
    cdef CSinoIdxElem s = get_s(E, A, S, T, event)
    if s >= n_axials:
        return

    cdef:
        CSinoIdxElem e = get_e(E, A, S, T, event)
        CSinoIdxElem a = get_a(E, A, S, T, event)

    if negate_delays and is_event_delay(packet): # is delay packet
        sino[e, a, s] -= 1
    else: # is event packet or delay but not negating
        sino[e, a, s] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline CSurrogateValue decay_correct(CSurrogateValue uncorrected,
                                          CPacket time, double half_life):
    """Correct counts for decays since beginning of scan (or injection).

    Args:
        uncorrected: Uncorrected counts.
        time: Time since beginning of scan (or injection) in milliseconds.
        half_life: Half life of the radioisotope in seconds.
    """
    return uncorrected * exp2(<float>time / 1000. / half_life)


# cdef inline CListIdx _find_time_index(np.ndarray[CPacket, ndim=1] lm,
#                                       CPacket time):
# cdef inline CListIdx _find_time_index(CPacket[:] lm, CListIdx lm_size,
#                                       CPacket time):
cdef CListIdx _find_time_index(np.ndarray[CPacket, ndim=1] lm,
                               CListIdx lm_size, CPacket time):
    """Quickly find the index of a given time tag. Does a sort-of binary
    search.

    If time > lm duration, returns lm_size.
    If time < lm beginning, returns 0.

    Warning: if the required time tag is omitted from lm behaviour is
    undefined.
    """
    cdef:
        Time_Index lower, upper, current

    # search for first time tag as lower bound
    lower = find_next_time(lm, lm_size, 0)
    # check for early termination
    if lower.time >= time:
        return 0

    # search for last time tag as upper bound
    upper = find_prev_time(lm, lm_size, lm_size-1)
    if upper.time <= time:
        return lm_size

    if upper.time <= lower.time:
        print('Unexpected error in _find_time_index: upper<=lower')
        # no time tags?
        return 0

    # # check we are in range
    # if not (lower.time < time < upper.time):
    #     # raise RuntimeError('Time tag is not in range of lm.')
    #     return lm_size

    while True:
        # smartly guess the next position assuming stable distribution
        # print(lower.idx, '+ (((', time, '-', lower.time, ')')
        # print('  / (', upper.time, '-', lower.time, '))')
        # print(' * (', upper.idx, '-', lower.idx, '))')
        # print('=', lower.idx, '+',
        #       (<double>(time - lower.time) / (upper.time - lower.time)),
        #       '*', upper.idx - lower.idx)
        # print('=', lower.idx, '+',
        #       <double>(time - lower.time)
        #       / (upper.time - lower.time)
        #       * (upper.idx - lower.idx))
        current.idx = (lower.idx
                       + <CListIdx>(<double>(time - lower.time)
                                    / (upper.time - lower.time)
                                    * (upper.idx - lower.idx)))
        # print('=', current.idx)

        # print('<', upper.time, '@', upper.idx)
        # print('o', time, '<>', current.idx)
        current = find_next_time(lm, lm_size, current.idx+1)
        # print('?', current.time, '@', current.idx, '=', time)
        # print('>', lower.time, '@', lower.idx)

        assert current.time != TimeIdx_invalid

        # smartly guess the next position assuming stable distribution
        # check we are in bounds
        if current.time >= upper.time:
            current = find_prev_time(lm, lm_size, upper.idx-1)
        elif current.time <= lower.time:
            current = find_next_time(lm, lm_size, lower.idx+1)

        # check if we're done!
        if current.time == time:
            return current.idx

        # update and iterate
        if current.time < time:
            lower = current
        else:
            upper = current
