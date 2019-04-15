#cython: language_level=3
#cython: linetrace=False, binding=False

include 'common.pxi'

from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport ceil, cos

import logging
from scipy import signal
from sklearn import decomposition

from petlink import interfile
import petlink.helpers.progress


cdef factors(int n):
    """Find the factors of a number n."""
    fs = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            fs.append(i)
            fs.append(n // i)
    return set(fs)


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


def _estimate_body_mask(sino):
    # threshold at 10% of 95th percentile after median filtering
    medfilt = signal.medfilt(sino)
    return medfilt > np.percentile(medfilt, 95) * 0.1


def _generate_pca_features(
        np.ndarray[CPacket, ndim=1] lm not None, shape not None,
        max_shape not None, CSinoIdxElem[:] segments_def,
        np.ndarray static_sino not None,
        CTimeIdx time_resolution, bar=None):
    cdef:
        CPacket packet, time
        CSinoIdxElem E = shape[0]
        CSinoIdxElem A = shape[1]
        CSinoIdxElem S = shape[2]
        CSinoIdxElem T = shape[3]
        CSinoIdxElem n_axials = segments_def[0]

    logger = logging.getLogger(__name__)
    logger.debug('called _generate_pca_features(')
    logger.debug('  shape=%s, time_res..=%s, segments_def=%s)',
                 shape, time_resolution, list(segments_def))

    assert len(shape) == 4
    for d in shape:
        assert d > 0
    assert time_resolution > 0
    assert 0 < n_axials <= S


    sino_series = unlist_series_low_res(
        lm, shape, max_shape, segments_def, time_resolution, bar)
    series_shape = sino_series.shape
    logger.debug('unlisted %d counts, %s', sino_series.sum(), series_shape)
    body_mask = _estimate_body_mask(static_sino)
    logger.debug('masked %f%% body', body_mask.sum() / body_mask.size * 100)
    sino_series[:, ~body_mask] = 0
    # rescale to equal counts per time point
    # sino_series = sino_series.astype(np.float16)
    time_series_counts = sino_series.sum(axis=(1, 2, 3))
    logger.debug('before normalisation: %s %s %s',
                 time_series_counts.shape, time_series_counts.dtype,
                 time_series_counts)
    # sino_series = (sino_series / sino_series.sum(axis=(0, 1, 2))
    #                * sino_series[1, ...].sum())
    logger.debug('normalisation: %s %s',
                 time_series_counts.mean(),
                 time_series_counts.mean() / time_series_counts)
    sino_series = sino_series * (
        time_series_counts.mean(dtype=np.float64)
        / time_series_counts[:, np.newaxis, np.newaxis, np.newaxis])
    logger.debug('normalised over time: %s %s %s',
                 sino_series.shape, sino_series.dtype,
                 sino_series.sum(axis=(1, 2, 3)))

    # Freeman-Tukey variance stabilisation
    sino_series[sino_series < 0] = 0.
    sino_series = np.sqrt(sino_series) + np.sqrt(sino_series + 1)
    # sino_series = sino_series.astype(np.float16)

    # TODO
    # should do some smoothing spatially (not in angle dimension)

    logger.debug('after scaling: %s %s %s',
                 sino_series.shape, sino_series.dtype,
                 sino_series.sum(axis=(1, 2, 3)))
    sino_series = sino_series.reshape(sino_series.shape[0], -1)
    sino_series = sino_series - sino_series.mean(axis=1)[:, np.newaxis]
    return sino_series, series_shape


@cython.binding(True)
def pca_surrogate(np.ndarray[CPacket, ndim=1] lm not None, shape not None,
                  CSinoIdxElem[:] segments_def,
                  CTimeIdx time_resolution, int n_components=2,
                  # double axial_spacing=0, # unused, for compat w/ *_surrogate
                  # double ring_rad=0,      # unused, for compat w/ *_surrogate
                  # unsigned int mash=0,    # unused, for compat w/ *_surrogate
                  # double half_life=0,     # unused, for compat w/ *_surrogate
                  bar=None,
                  save_model=False):
    """Calculate the first num_components principal components of a PCA on the
    sinograms.

    Only considers segment 0.

    Implementation of:
        Thielemans, K., Rathore, S., Engbrant, F., & Razifar, P. (2011,
        October). Device-less gating for PET/CT using PCA. In Nuclear Science
        Symposium and Medical Imaging Conference (NSS/MIC), 2011 IEEE (pp.
        3904-3910). IEEE.

    Additionally scales each component by the ratio of variance explained.

    Args:
        lm: 1D Numpy array containing LM events.
        shape: Tuple containing sinogram unlisted shape.
        n_axials: How many axial positions are in segment 0.
        time_resolution: Number of ms each time point should cover.
        axial_spacing: Spacing between axial positions in mm (ring spacing if
            span == 1, else ring spacing / 2).
        ring_rad: Radius of the detector rings in mm.
        mash: Mash factor of the list mode data.
        horizontal_a: (0) Angle index where horizontal angles begin.
            0 <= horizontal_a < A // 2 - angle_width.
        angle_width: (10) Number of angle indices to group.
        bar: (optional) A progress bar to update.

    Returns:
        The PCA surrogate signal, of shape (n_componenets, length) where length
        is ceil(duration / time_resolution).
    """
    logger = logging.getLogger(__name__)
    # first generate a low res. sinogram series
    # PCA

    # constants for subsampling
    MAX_E = 100
    MAX_A = 10
    n_axials = max(segments_def)
    max_shape = (MAX_E, MAX_A, n_axials)

    static_sino = unlist_series_low_res(
        lm, shape, max_shape, segments_def, time_resolution, bar).sum(axis=0)
    try:
        features, f_shape = _generate_pca_features(
            lm, shape, max_shape, segments_def, static_sino, time_resolution, bar)
        logger.info('Performing PCA')
        pca = decomposition.PCA(n_components)
        # final component is probably noisy, as it was likely cut short in time
        # don't use it to train PCA
        pca.fit(features[:-1, ...])
        surrogate = pca.transform(features)
    except MemoryError:
        logger.warning('Memory Error on PCA. Performing Incremental PCA')
        pca = decomposition.IncrementalPCA(n_components, batch_size=100)
        starts = list(np.arange(begin(lm), duration(lm), time_resolution * 100))
        ends = starts[1:] + [duration(lm)]
        surrogate = np.zeros((0, n_components))
        for start, end in zip(starts, ends):
            start_idx = _find_time_index(lm, lm.size, start)
            end_idx = _find_time_index(lm, lm.size, end)
            features, f_shape = _generate_pca_features(
                lm[start_idx:end_idx], shape, max_shape, segments_def, static_sino,
                time_resolution, bar)
            pca.partial_fit(features)
            surrogate = np.concatenate((surrogate, pca.transform(features)))

    logger.info(
        'PCA complete.\n'
        + '\n'.join('PCA {}: explains {:.1f} ({:.1f}% ratio).'
                    .format(i+1, pca.explained_variance_[i],
                            pca.explained_variance_ratio_[i] * 100)
                    for i in range(n_components)))

    if save_model:
        import pickle
        np.savez('features.npz', features=features, shape=f_shape,
                 transformed=surrogate, static=static_sino)
        with open('pca_model.pkl', 'wb+') as fp:
            pickle.dump(pca, fp)

    # We need a measure of quality of tracking signals.
    # Use the maximum autocorrelation (excluding with no shift).
    # Assumably a realistic tracking signal should have a high maximum.
    weighting = np.array(
        [np.correlate(surrogate[:, c], surrogate[:, c],
                      mode='full')[:surrogate.shape[0]-1].max()
         for c in range(n_components)])
    # weighting = pca.explained_variance_ratio_

    surrogate -= surrogate.mean(0)
    surrogate /= surrogate.std(0) # normalise
    surrogate *= weighting # scale surrogate
    surrogate = surrogate.T
    logger.debug('surroagate shape: %s.', surrogate.shape)

    return surrogate.astype(NpSurrogateValue)
