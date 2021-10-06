"""Testing for listmode submodule."""

import sys
import os
import logging
import pytest
import numpy as np

from petlink import ListMode, Interfile


logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)


HERE = os.path.dirname(__file__)


hoffrock_ptd = os.path.join(HERE, 'data', 'hoffrock', 'LM.ptd')
hoffrock_prompt_hs = os.path.join(HERE, 'data', 'hoffrock', 'lm_prompt.hs')
hoffrock_delay_hs = os.path.join(HERE, 'data', 'hoffrock', 'lm_delay.hs')


@pytest.mark.data
def test_ListMode_load_ptd():
    ListMode.from_file(hoffrock_ptd)


@pytest.mark.data
def test_ListMode_get_ifl():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.ifl
    assert lm.ifl['originating system']


@pytest.mark.data
def test_ListMode_get_data():
    lm = ListMode.from_file(hoffrock_ptd)
    assert isinstance(lm.data, np.ndarray)
    assert lm.data.size > 0


@pytest.mark.data
def test_ListMode_get_properties():
    lm = ListMode.from_file(hoffrock_ptd)
    assert isinstance(lm.unlist_shape, tuple)
    assert lm.duration > 0


@pytest.mark.data
def test_ListMode_time_indexing():
    lm = ListMode.from_file(hoffrock_ptd)
    one_sec_lm = lm.tloc[5000:6000]
    assert one_sec_lm.duration == 1000

    first_sec_lm = lm.tloc[:1000]
    assert first_sec_lm.duration == 1000
    assert first_sec_lm.get_time_at_index(0) == 0

    last_sec_lm = lm.tloc[-1000:]
    assert last_sec_lm.duration == 1000
    assert last_sec_lm.get_time_at_index(0) == 0


@pytest.mark.data
def test_ListMode_time_index_consistency():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.get_time_at_index(lm.get_index_at_time(100)) == 100


@pytest.mark.data
def test_ListMode_time_at_start_is_0():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.get_time_at_index(0) == 0


@pytest.mark.slow
@pytest.mark.data
def test_ListMode_unlist():
    lm = ListMode.from_file(hoffrock_ptd)
    prompt, delay = lm.unlist()
    psino, dsino = prompt.get_data(), delay.get_data()

    # correct no. counts
    assert psino.sum() == lm._make_tag_mask('prompt').sum()
    assert dsino.sum() == lm._make_tag_mask('delay').sum()

    # correct shape
    assert psino.shape == dsino.shape
    assert psino.ndim == 3

    psino_ref = Interfile.from_file(hoffrock_prompt_hs).get_data()
    assert np.all(psino == psino_ref.astype(psino.dtype))
    dsino_ref = Interfile.from_file(hoffrock_delay_hs).get_data()
    assert np.all(dsino == dsino_ref.astype(dsino.dtype))


@pytest.mark.slow
@pytest.mark.data
def test_ListMode_unlist_tof():
    lm = ListMode.from_file(hoffrock_ptd)
    prompt, delay = lm.unlist(keep_tof=True)
    psino, dsino = prompt.get_data(), delay.get_data()

    # correct no. counts
    assert psino.sum() == lm._make_tag_mask('prompt').sum()
    assert dsino.sum() == lm._make_tag_mask('delay').sum()

    # correct shape
    assert psino.shape == dsino.shape
    assert psino.ndim == 4

    # tof bins aren't replicated or 0
    assert np.any(psino[:, :, :, 6] != psino[:, :, :, 7])
    psino_tof_sums = psino.sum(axis=(0, 1, 2))
    dsino_tof_sums = dsino.sum(axis=(0, 1, 2))
    assert np.all(psino_tof_sums[:13] > 0)
    assert np.all(psino_tof_sums[13] == 0)  # final bin is delays
    assert np.all(dsino_tof_sums[:13] == 0)  # no tof for delays
    assert np.all(dsino_tof_sums[13] > 0)


@pytest.mark.data
def test_ListMode_extract():
    lm = ListMode.from_file(hoffrock_ptd)
    time = lm.extract('time')
    # check time counts up
    assert time[0] + 1 == time[1]
    assert time[1] + 1 == time[2]
