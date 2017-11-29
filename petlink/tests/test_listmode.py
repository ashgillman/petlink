"""Testing for listmode submodule."""

import os
import pytest
import numpy as np

from ..listmode import ListMode


HERE = os.path.dirname(__file__)


hoffrock_ptd = os.path.join(HERE, 'data', 'hoffrock', 'LM.ptd')


def test_ListMode_load_ptd():
    lm = ListMode.from_file(hoffrock_ptd)


def test_ListMode_get_ifl():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.ifl
    assert lm.ifl['originating system']


def test_ListMode_get_data():
    lm = ListMode.from_file(hoffrock_ptd)
    assert isinstance(lm.data, np.ndarray)
    assert lm.data.size > 0


def test_ListMode_get_properties():
    lm = ListMode.from_file(hoffrock_ptd)
    assert isinstance(lm.unlist_shape, tuple)
    assert lm.duration > 0


def test_ListMode_time_indexing():
    lm = ListMode.from_file(hoffrock_ptd)
    one_sec_lm = lm.tloc[5000:6000]
    assert one_sec_lm.duration == 1000


def test_ListMode_time_index_consistency():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.get_time_at_index(lm.get_index_at_time(100)) == 100


def test_ListMode_time_at_start_is_0():
    lm = ListMode.from_file(hoffrock_ptd)
    assert lm.get_time_at_index(0) == 0
