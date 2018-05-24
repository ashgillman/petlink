#cython: language_level=3
#cython: linetrace=False, binding=False

include 'common.pxi'

from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport ceil, cos

import logging


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void calc_xyz_distributions(
        CCount[:, :, :] sino,
        CCount[:] x_dist, CCount[:] y_dist, CCount[:] z_dist,
        CSinoIdxElem E, CSinoIdxElem A, CSinoIdxElem n_axials,
        CSinoIdxElem a_x_min, CSinoIdxElem a_x_max,
        CSinoIdxElem a_y_min, CSinoIdxElem a_y_max):
    cdef:
        CSinoIdxElem e, a, s
        CCount n
    for e in range(E):
     for a in range(A):
      for s in range(n_axials):
        n = sino[e, a, s]
        if n > 0: # ignore where delays outnumber prompts
            if a_x_min <= a < a_x_max:
                x_dist[e] += n
            elif a_y_min <= a < a_y_max:
                y_dist[e] += n
            z_dist[s] += n


cdef float scale_element_to_transaxial_distance(
        CSinoIdxElem e, CSinoIdxElem E, CSinoIdxElem A, double ring_rad):
    """Scale a projection index to the transaxial distance from the gantry
    axis.
    """
    # There are n_ang - 1= 2*A - 1 projection positions, corresponding to 180`
    # Need to account for non-covered margin at the edges
    cdef:
        CSinoIdxElem margin = A - E // 2
        float ang_pos = margin + e
        float pos = -cos(ang_pos / (2*A) * PI) * ring_rad
    return pos


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_com(
        CSurrogateValue[:, :] com, CCount[:, :, :] sino,
        double axial_spacing, double ring_rad,
        CTimeIdx time_idx,
        CSinoIdxElem a_x_min, CSinoIdxElem a_x_max, CSinoIdxElem a_y_min,
        CSinoIdxElem a_y_max, CSinoIdxElem E, CSinoIdxElem A,
        CSinoIdxElem n_axials):
    cdef:
        CTimeIdx i
        CSinoIdxElem s, e, a
        CCount n
        CSurrogateValue pos
        CCount[:] x_dist = np.zeros((E, ), dtype=NpCount)
        CCount[:] y_dist = np.zeros((E, ), dtype=NpCount)
        CCount[:] z_dist = np.zeros((n_axials, ), dtype=NpCount)
        CSurrogateValue[3] numer = np.zeros((3, )), denom = np.zeros((3, ))
    # calculate 1D distributions
    calc_xyz_distributions(sino, x_dist, y_dist, z_dist, E, A, n_axials,
                           a_x_min, a_x_max, a_y_min, a_y_max)

    # calc COMs
    for e, n in enumerate(x_dist):
        pos = scale_element_to_transaxial_distance(e, E, A, ring_rad)
        numer[0] += n * pos
        denom[0] += n
    for e, n in enumerate(y_dist):
        pos = scale_element_to_transaxial_distance(e, E, A, ring_rad)
        numer[1] += n * pos
        denom[1] += n
    for z, n in enumerate(z_dist):
        # NB: This currently assumes axial compresion := 1
        pos = (z - n_axials / 2) * axial_spacing
        numer[2] += n * pos
        denom[2] += n

    for i, (n, d) in enumerate(zip(numer, denom)):
        com[i, time_idx] = (n / d) if d > 0. else 0.


@cython.binding(True)
@cython.cdivision(True)
@cython.cdivision_warnings(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def com_surrogate(np.ndarray[CPacket, ndim=1] lm not None, shape not None,
                  CSinoIdxElem n_axials,
                  CTimeIdx time_resolution, double axial_spacing,
                  double ring_rad,
                  # double half_life=0, # unused, for compat w/ sens_surrogate
                  CSinoIdxElem horizontal_a=0, CSinoIdxElem angle_width=10,
                  bar=None,
                  save_model=False):
    """Calculate the COM in x-y-z of the list data.

    Only considers segment 0.

    Args:
        lm: 1D Numpy array containing LM events.
        shape: Tuple containing sinogram unlisted shape.
        n_axials: How many axial positions are in segment 0.
        time_resolution: Number of ms each time point should cover.
        axial_spacing: Spacing between axial positions in mm (ring spacing if
            span == 1, else ring spacing / 2).
        ring_rad: Radius of the detector rings in mm.
        horizontal_a: (0) Angle index where horizontal angles begin.
            0 <= horizontal_a < A // 2 - angle_width.
        angle_width: (10) Number of angle indices to group.
        bar: (optional) A progress bar to update.

    Returns:
        The x-y-z-COM surrogate signal, of shape (3, length) where length is
            the daution // time_resolution.
    """
    cdef:
        CPacket packet, time
        CTimeIdx begin_time = begin(lm)
        CTimeIdx length = <CTimeIdx>ceil((<double>duration(lm) - begin_time)
                                         / time_resolution)
        CTimeIdx time_idx = 0, new_time_idx = 0
        CSurrogateValue[:, :] com = np.zeros(
            (3, length), dtype=NpSurrogateValue)
        CSinoIdxElem E = shape[0]
        CSinoIdxElem A = shape[1]
        CSinoIdxElem S = shape[2]
        CSinoIdxElem T = shape[3]
        CCount[:, :, :] temp_sino = np.zeros((E, A, n_axials), dtype=NpCount)
        CSinoIdxElem a_x_min = horizontal_a
        CSinoIdxElem a_x_max = a_x_min + angle_width
        CSinoIdxElem a_y_min = a_x_min + A // 2
        CSinoIdxElem a_y_max = a_y_min + angle_width

    assert len(shape) == 4
    for d in shape:
        assert d > 0
    assert time_resolution > 0
    assert axial_spacing != 0
    assert ring_rad != 0
    assert 0 < n_axials <= S
    assert horizontal_a < A//2 - angle_width
    assert 0 < angle_width < A
    assert a_y_max <= A

    logger = logging.getLogger(__name__)

    for pidx in range(len(lm)):
        packet = lm[pidx]
        # build a sinogram
        if is_event(packet):
            bin_event_no_tof(temp_sino, packet, E, A, S, T, n_axials)

        # update COM and reset sinogram
        elif is_tag_time(packet):
            time = packet & TAG_UNMASK
            new_time_idx = (time - begin_time) // time_resolution
            assert new_time_idx < length

            if new_time_idx != time_idx: # next time bin
                update_com(com, temp_sino, axial_spacing, ring_rad, time_idx,
                           a_x_min, a_x_max, a_y_min, a_y_max, E, A, n_axials)

                # reset temp variables
                time_idx = new_time_idx
                temp_sino = np.zeros((E, A, n_axials), dtype=NpCount)

                PyErr_CheckSignals()
                if bar:
                    bar.update(pidx)

    # final update
    update_com(com, temp_sino, axial_spacing, ring_rad, time_idx,
               a_x_min, a_x_max, a_y_min, a_y_max, E, A, n_axials)
    logger.debug('surrogate shape: %s.', com.shape)

    return np.array(com, dtype=NpSurrogateValue) # convert back to real Numpy
