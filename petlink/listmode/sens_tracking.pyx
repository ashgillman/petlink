#cython: language_level=3
#cython: linetrace=False, binding=False

include 'common.pxi'

from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport ceil

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void accum_event(CSurrogateValue[:] sens, CPacket packet,
                             CTimeIdx time_idx):
    cdef CPacket event = packet & EVENT_UNMASK
    if is_event_delay(packet): # is delay packet
        sens[time_idx] -= 1
    else: # is event packet
        sens[time_idx] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void update_sens(
    CSurrogateValue[:] sens, CTimeIdx time_idx, CPacket time, double half_life):
    if sens[time_idx] > 0:
        sens[time_idx] = decay_correct(sens[time_idx], time, half_life)
    else:
        sens[time_idx] = 0 # clamp


@cython.binding(True)
@cython.cdivision(True)
@cython.cdivision_warnings(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sens_surrogate(np.ndarray[CPacket, ndim=1] lm not None, shape not None,
                   CSinoIdxElem n_axials,
                   CTimeIdx time_resolution, double axial_spacing,
                   double half_life,
                   # double ring_rad=0, # unused, for compat w/ com_surrogate
                   # unsigned int mash=0, # unused, for compat w/ com_surrogate
                   decay_correct=True,
                   bar=None,
                   save_model=False):
    """Calculate the sensitivity of the list data.

    Only considers segment 0.

    Args:
        lm: 1D Numpy array containing LM events.
        shape: Tuple containing sinogram unlisted shape.
        n_axials: How many axial positions are in segment 0.
        time_resolution: Number of ms each time point should cover.
        half_life: Half life of the radioisotope in seconds.
        bar: (optional) A progress bar to update.

    Returns:
        The sensitivity surrogate signal.
    """
    assert len(shape) == 4
    for d in shape:
        assert d > 0
    assert time_resolution > 0
    assert 0 < n_axials <= shape[2]

    cdef:
        CPacket packet, time = 0
        CTimeIdx begin_time = begin(lm)
        CTimeIdx length = <CTimeIdx>ceil((<double>duration(lm) - begin_time)
                                         / time_resolution)
        CTimeIdx time_idx = 0, new_time_idx = 0
        CSurrogateValue[:] sens = np.zeros((length, ), dtype=NpSurrogateValue)

    for pidx in range(len(lm)):
        packet = lm[pidx]
        # accumulate counts
        if is_event(packet):
            accum_event(sens, packet, time_idx)

        elif is_tag_time(packet):
            time = packet & TAG_UNMASK
            new_time_idx = (time - begin_time) // time_resolution
            assert new_time_idx < length

            if new_time_idx != time_idx: # next time bin
                if decay_correct: update_sens(sens, time_idx,
                                              time-begin_time, half_life)

                # update time
                time_idx = new_time_idx

                PyErr_CheckSignals()
                if bar:
                    bar.update(pidx)

    # final update
    update_sens(sens, time_idx, time, half_life)

    return np.array(sens, dtype=NpSurrogateValue) # convert back to real Numpy
