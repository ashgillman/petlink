"""Testing for interfile.py"""

import os
from itertools import zip_longest
import numpy as np
import pytest

from ..interfile import Interfile, InvalidInterfileError, PL_DTYPE


test_lines = (
    ('magic'         , ('!INTERFILE:=', 'INTERFILE', '', '!')),
    ('no_value'      , ('no value :=', 'no value', '', '')),
    ('int'           , ('int := 0', 'int', 0, '')),
    ('float'         , ('float := 0.0', 'float', 0., '')),
    ('string'        , ('string := Hello, World!',
                        'string', 'Hello, World!', '')),
    ('list'          , ('list := { 1, 2, 3 }', 'list', [1, 2, 3], '')),
    ('list_unspaced' , ('list unspaced := {1,2,3}',
                        'list unspaced', [1, 2, 3], '')),
    ('path'          , ('path := \\\\path\\to\\file.file',
                        'path', os.path.join(
                            os.path.sep, 'path', 'to', 'file.file'),
                        '')),
    ('blank'         , ('', None, None, '')),
    ('comment'       , ('; comment', None, None, '')),
    ('empty_comment' , (';', None, None, '')),
    ('exclaim'       , ('!exclaim :=',
                        'exclaim', '', '!')),
    ('percent'       , ('%percent :=',
                        'percent', '', '%')),
    ('spaced'        , ('key 1 := 1', 'key 1', 1, '')),
    ('spaced_left'   , ('key 2 :=2', 'key 2', 2, '')),
    ('spaced_right'  , ('key 3:= 3', 'key 3', 3, '')),
    ('unspaced'      , ('key 4:=4', 'key 4', 4, '')),
)


def second(iter):
    next(iter)
    return next(iter)


def check_against_test_lines(parsed):
    for actual, expected in zip_longest(
            parsed.header.keys(), (t[1][1] for t in test_lines
                                   if t[1][1] is not None)):
        assert actual == expected

    for actual, expected in zip_longest(
            parsed.header.values(), (t[1][2] for t in test_lines
                                     if t[1][2] is not None)):
        assert actual == expected

    for actual, expected in zip_longest(
            parsed.key_types.values(), (t[1][3] for t in test_lines
                                        if t[1][1] is not None)):
        assert actual == expected


def test_Interfile_individual():
    """Test individual lines of test_lines"""
    for test, (parsable, key, value, key_type) in test_lines[1:]:
        parsed = Interfile('\n'.join((test_lines[0][1][0],
                                      parsable)))
        if key is not None:
            assert second(iter(parsed.header.keys())) == key
        if value is not None:
            assert second(iter(parsed.header.values())) == value
        if key is not None:
            assert second(iter(parsed.key_types.values())) == key_type


def test_Interfile_together():
    """Test parsing the whole of test_lines"""
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))
    check_against_test_lines(parsed)


def test_Interfile_no_magic():
    """Test parsing with a file not beginning with '!INTERFILE'"""
    with pytest.raises(InvalidInterfileError):
        parsed = Interfile('\n'.join(t[1][0] for t in test_lines[1:]))


def test_Interfile_file(tmpdir):
    content = '\n'.join(t[1][0] for t in test_lines)
    f = tmpdir.join('interfile.h')
    f.write(content)
    parsed = Interfile(str(f))
    check_against_test_lines(parsed)


def test_Interfile_read_data_absolute(tmpdir):
    header_f = tmpdir.join('interfile.h')
    data_f = tmpdir.join('interfile.hx')

    header_content = '\n'.join(t[1][0] for t in test_lines)
    header_content += '\nname of data file := '
    header_content += str(data_f)
    header_content += '\n'

    data_content = np.arange(10, dtype=PL_DTYPE)

    header_f.write(header_content)
    data_content.tofile(str(data_f))

    parsed = Interfile(str(header_f))
    assert (parsed.get_data() == data_content).all()


def test_Interfile_read_data_relative(tmpdir):
    header_f = tmpdir.join('interfile.h')
    data_f = tmpdir.join('interfile.hx')

    header_content = '\n'.join(t[1][0] for t in test_lines)
    header_content += '\nname of data file := interfile.hx\n'

    data_content = np.arange(10, dtype=PL_DTYPE)

    header_f.write(header_content)
    data_content.tofile(str(data_f))

    parsed = Interfile(str(header_f))
    assert (parsed.get_data() == data_content).all()


def test_Interfile_dont_read_data_relative_without_sourcefile(tmpdir):
    header_content = '\n'.join(t[1][0] for t in test_lines)
    header_content += '\nname of data file := interfile.hx\n'

    parsed = Interfile(header_content)
    with pytest.raises(FileNotFoundError) as e:
        parsed.get_data()

    # make sure it was the correct error
    assert 'source file' in str(e)


def test_Interfile_read_data_memmap(tmpdir):
    header_f = tmpdir.join('interfile.h')
    data_f = tmpdir.join('interfile.hx')

    header_content = '\n'.join(t[1][0] for t in test_lines)
    header_content += '\nname of data file := '
    header_content += str(data_f)
    header_content += '\n'

    data_content = np.arange(10, dtype=PL_DTYPE)

    header_f.write(header_content)
    data_content.tofile(str(data_f))

    parsed = Interfile(str(header_f))
    parsed_data = parsed.get_data(memmap=True)
    assert (parsed_data == data_content).all()
    assert isinstance(parsed_data, np.memmap)
