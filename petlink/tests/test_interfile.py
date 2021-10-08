"""Testing for interfile.py"""

import os
import pathlib
from itertools import zip_longest
import numpy as np
import pytest
import pyparsing as pp

from ..interfile import Interfile, Value, InvalidInterfileError
from ..constants import PL_DTYPE


HERE = os.path.dirname(__file__)


class IsNone():
    """Evaluates as equal to None without actually being None.

    >>> IsNone() == None
    True
    >>> IsNone() is None
    False
    """
    def __eq__(self, other):
        return other is None


# ('name' , ('line', 'expected_key', 'expected_value', 'expected_key_type'))
test_lines = (
    ('magic',              ('!INTERFILE:=', 'INTERFILE', '', '!')),
    ('int',                ('int := 0', 'int', 0, '')),
    ('float',              ('float := 0.0', 'float', 0., '')),
    ('scifloat',           ('scifloat := 5.1e2', 'scifloat', 510., '')),
    ('scifloat2',          ('scifloat2 := 3.14e+007',
                            'scifloat2', 3.14e7, '')),
    ('no_value',           ('no value :=', 'no value', '', '')),
    ('none',               ('none := <NONE>', 'none', IsNone(), '')),
    ('string',             ('string := Hello, World!',
                            'string', 'Hello, World!', '')),
    ('list',               ('list := { 1, 2, 3 }',
                            'list', [1, 2, 3], '')),
    ('list_unspaced',      ('list unspaced := {1,2,3}',
                            'list unspaced', [1, 2, 3], '')),
    ('list_weird_spaced',  ('list weird spaced := {1, 2 ,3}',
                            'list weird spaced', [1, 2, 3], '')),
    ('list_floats',        ('list of floats := {1.1,2.2 ,3.3}',
                            'list of floats', [1.1, 2.2, 3.3], '')),
    ('list_text',          ('list of text := {a, b ,c,d, e f }',
                            'list of text', ['a', 'b', 'c', 'd', 'e f'], '')),
    ('path',               ('path := \\\\path\\to\\file.file',
                            'path', os.path.join(
                                os.path.sep, 'path',
                                'to', 'file.file'),
                            '')),
    ('date (yyyy:mm:dd)',  ('date (yyyy:mm:dd) := 2016:11:15',
                            'date', '2016:11:15', '')),
    ('blank',              ('', None, None, '')),
    ('comment',            ('; comment', None, None, '')),
    ('empty_comment',      (';', None, None, '')),
    ('exclaim',            ('!exclaim :=',
                            'exclaim', '', '!')),
    ('percent',            ('%percent :=',
                            'percent', '', '%')),
    ('spaced',             ('key 1 := 1', 'key 1', 1, '')),
    ('spaced_left',        ('key 2 :=2', 'key 2', 2, '')),
    ('spaced_right',       ('key 3:= 3', 'key 3', 3, '')),
    ('unspaced',           ('key 4:=4', 'key 4', 4, '')),
    ('colons:in:key',      ('colons:in:key:=colons:in:value',
                            'colons:in:key', 'colons:in:value', '')),
    ('ignore case',        ('Ignore Case := yes', 'ignore case', 'yes', '')),
    ('units',              ('units (unit):=value', 'units', 'value', '')),
    ('vector',             ('vector [1] := 0', 'vector', [0], '')),
    ('vector2',            ('vector2 [1] := 0\n'
                            'vector2 [2] := 1',
                            'vector2', [0, 1], '')),
    ('vector unordered',   ('vector unordered [2] := 0\n'
                            'vector unordered [1] := 1',
                            'vector unordered', [1, 0], '')),
    ('list in vector',     ('list in vector [1] := 1\n'
                            'list in vector [2] := { 0, 1 }',
                            'list in vector', [1, [0, 1]], '')),
    ('end magic',          ('!END OF INTERFILE:=',
                            'END OF INTERFILE', '', '!')),
)


def second(iter):
    next(iter)
    return next(iter)


def check_against_test_lines(parsed):
    for actual, expected in zip_longest(
            parsed.header.keys(),
            (t[1][1] for t in test_lines[1:-1] if t[1][1] is not None)):
        assert actual == expected

    for actual, expected in zip_longest(
            parsed.header.values(),
            (t[1][2] for t in test_lines[1:-1] if t[1][2] is not None)):
        assert actual.value == expected

    for actual, expected in zip_longest(
            parsed.header.values(),
            (t[1][3] for t in test_lines[1:-1] if t[1][1] is not None)):
        assert actual.key_type == expected


def test_Interfile_individual():
    """Test individual lines of test_lines"""
    # Interfile._initialise_value_parser()
    # Interfile._int_parser.setDebug(True)
    # Interfile._float_parser.setDebug(True)
    # Interfile._list_parser.setDebug(True)
    # Interfile._path_parser.setDebug(True)
    # Interfile._text_parser.setDebug(True)

    start, *middle, end = test_lines
    for test, (parsable, key, value, key_type) in middle:
        print(test)
        parsed = Interfile('\n'.join((start[1][0],
                                      parsable,
                                      end[1][0])))
        if key is not None:
            assert next(iter(parsed.header.keys())) == key
        if value is not None:
            assert next(iter(parsed.header.values())).value == value
        if key is not None:
            assert next(iter(parsed.header.values())).key_type == key_type


def test_Interfile_together():
    """Test parsing the whole of test_lines"""
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))
    check_against_test_lines(parsed)


def test_Interfile_format_line():
    """Test formatting against the parser."""
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))
    reparsed = Interfile(str(parsed))
    for before, after in zip_longest(parsed.header.items(),
                                     reparsed.header.items()):
        assert after == before


def test_Interfile_no_magic():
    """Test parsing with a file not beginning with '!INTERFILE'"""
    # strict
    with pytest.raises(InvalidInterfileError):
        Interfile('\n'.join(t[1][0] for t in test_lines[1:]))

    # not strict
    Interfile('\n'.join(t[1][0] for t in test_lines[1:]), strict=False)


def test_Interfile_file(tmpdir):
    header = '\n'.join(t[1][0] for t in test_lines)
    data = np.arange(50)
    ifl = Interfile(header, data=data)
    f = tmpdir.join('interfile.h')
    ifl.to_filename(str(f))
    parsed = Interfile(source=str(f))
    for k, v in ifl.header.items():
        assert parsed.header[k] == v
    assert parsed.sourcefile == f
    assert np.all(parsed.get_data() == data)
    assert parsed.get_datatype() == data.dtype


def test_Interfile_file_relative(tmpdir):
    header = '\n'.join(t[1][0] for t in test_lines)
    data = np.arange(50)
    ifl = Interfile(header, data=data)
    os.chdir(tmpdir)
    filedir = tmpdir.join('subdir').mkdir()
    f = filedir.join('interfile.h')
    f = f.relto(tmpdir)
    ifl.to_filename(str(f))
    parsed = Interfile(source=str(f))
    for k, v in ifl.header.items():
        assert parsed.header[k] == v
    assert parsed.sourcefile == f
    assert np.all(parsed.get_data() == data)


def test_Interfile_maintains_data_type_key_type(tmpdir):
    # normally saves as:
    #    number format := xx
    #    !number of bytes per pixel := xx
    #    imagedata byte order := xx
    header = '\n'.join(t[1][0] for t in test_lines)
    data = np.arange(50)
    ifl = Interfile(header, data=data)
    f = tmpdir.join('interfile.h')
    ifl.to_filename(str(f))
    parsed = Interfile(source=str(f))

    assert parsed.header['number format'].key_type == ''
    assert parsed.header['number of bytes per pixel'].key_type == '!'
    assert parsed.header['imagedata byte order'].key_type == ''

    # but if we changed that, it will maintain
    #    !number format := xx
    #    number of bytes per pixel := xx
    #    imagedata byte order := xx
    this_lines = list(test_lines)  # clone
    this_lines.insert(1, ('', ('!number format := float', '', '', '')))
    this_lines.insert(2, ('', ('number of bytes per pixel := 4', '', '', '')))
    this_lines.insert(3, ('', ('imagedata byte order := littleendian', '', '', '')))
    header = '\n'.join(t[1][0] for t in this_lines)
    data = np.arange(50)
    ifl = Interfile(header, data=data)
    f = tmpdir.join('interfile.h')
    ifl.to_filename(str(f))
    parsed = Interfile(source=str(f))

    assert parsed.header['number format'].key_type == '!'
    assert parsed.header['number of bytes per pixel'].key_type == ''
    assert parsed.header['imagedata byte order'].key_type == ''


def test_zero_terminated_file(tmpdir):
    """E7 Tools norm files have a zero byte on a line at the end..."""
    header = '\n'.join(t[1][0] for t in test_lines)
    data = np.arange(50)
    ifl = Interfile(header, data=data)
    f = tmpdir.join('interfile.h')

    ifl.to_filename(str(f))

    # add the zero byte line seen in E7 extracted norm files
    with open(str(f), 'a') as fp:
        fp.write('\x00\n')

    parsed = Interfile(source=str(f))
    for k, v in ifl.header.items():
        assert parsed.header[k] == v
    assert parsed.sourcefile == f
    assert np.all(parsed.get_data() == data)


def test_Interfile_caseless_lookup():
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))
    assert parsed['string'] == 'Hello, World!'
    assert parsed['String'] == 'Hello, World!'
    assert parsed['STRING'] == 'Hello, World!'


def test_Interfile_meta_lookup():
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))
    assert parsed.header['string'].value == 'Hello, World!'


def test_Interfile_add_key():
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))

    parsed['new'] = 'unitless, no type'
    assert parsed['new'] == 'unitless, no type'
    assert parsed.header['new'].value == 'unitless, no type'
    assert parsed.header['new'].key_type == ''
    assert parsed.header['new'].units is None

    parsed['another'] = Value(
        value='unit, type', key_type='!', units='units', inline=True)
    assert parsed['another'] == 'unit, type'
    assert parsed.header['another'].value == 'unit, type'
    assert parsed.header['another'].key_type == '!'
    assert parsed.header['another'].units == 'units'
    assert parsed.header['another'].inline


def test_Interfile_add_list():
    parsed = Interfile('\n'.join(t[1][0] for t in test_lines))

    parsed['inline vector'] = Value(
        value=[1, 2, 3], key_type='!', units='units', inline=True)
    parsed['noninline vector'] = Value(
        value=[1, 2, 3], key_type='!', units='units', inline=False)

    line = [l for l in str(parsed).splitlines() if 'inline vector' in l][0]
    parser = (pp.Literal('!')
              + pp.OneOrMore(pp.Word(pp.alphas))
              + pp.Literal('(').suppress()
              + pp.Word(pp.alphas)
              + pp.Literal(')').suppress()
              + pp.Literal(':=').suppress()
              + pp.Literal('{').suppress()
              + pp.delimitedList(pp.Word(pp.nums))
              + pp.Literal('}').suppress())
    tokens = ' '.join(parser.parseString(line))
    assert tokens == '! inline vector units 1 2 3'

    for index_val in (1, 2, 3):
        line = [l for l in str(parsed).splitlines()
                if 'noninline vector' in l][index_val-1]
        parser = (pp.Literal('!')
                  + pp.OneOrMore(pp.Word(pp.alphas))
                  + pp.Literal('(').suppress()
                  + pp.Word(pp.alphas)
                  + pp.Literal(')').suppress()
                  + pp.Literal('[').suppress()
                  + pp.Word(pp.nums)
                  + pp.Literal(']').suppress()
                  + pp.Literal(':=').suppress()
                  + pp.Word(pp.nums))
        tokens = ' '.join(parser.parseString(line))
        assert tokens == ('! noninline vector units ' + str(index_val) + ' '
                          + str(index_val))


def test_Interfile_cleanup():
    ifl = Interfile('\n'.join(t[1][0] for t in test_lines))
    ifl['patient orientation'] = 'HFS'
    ifl['image data byte order'] = 'LITTLEENDIAN'
    ifl['scale factor'] = 2
    ifl['data offset in bytes'] = 2
    ifl['image relative start time'] = 0
    cleaned = Interfile(str(ifl), do_clean=True)

    assert cleaned['patient orientation'] == 'head_in'
    assert cleaned['imagedata byte order'] == 'LITTLEENDIAN'
    assert 'image data byte order' not in cleaned
    assert cleaned['scaling factor'] == 2
    assert 'scale factor' not in cleaned
    assert cleaned.header['data offset in bytes'].key_type == ';'
    assert cleaned['image relative start time'] == [0]


def test_Interfile_read_data_absolute(tmpdir):
    header_f = tmpdir.join('interfile.h')
    data_f = tmpdir.join('interfile.hx')

    header_content = '\n'.join(t[1][0] for t in test_lines[:-1])
    header_content += '\nname of data file := \\'
    header_content += str(data_f).replace('/', '\\')
    header_content += '\n'

    data_content = np.arange(10, dtype=PL_DTYPE)

    header_f.write(header_content)
    data_content.tofile(str(data_f))

    parsed = Interfile(source=str(header_f))
    assert (parsed.get_data() == data_content).all()


def test_Interfile_read_data_relative(tmpdir):
    header_f = tmpdir.join('interfile.h')
    data_f = tmpdir.join('interfile.hx')

    header_content = '\n'.join(t[1][0] for t in test_lines[:-1])
    header_content += '\nname of data file := interfile.hx\n'

    data_content = np.arange(10, dtype=PL_DTYPE)

    header_f.write(header_content)
    data_content.tofile(str(data_f))

    parsed = Interfile(source=str(header_f))
    assert (parsed.get_data() == data_content).all()


def test_Interfile_dont_read_data_relative_without_sourcefile(tmpdir):
    header_content = '\n'.join(t[1][0] for t in test_lines[:-1])
    header_content += '\nname of data file := interfile.hx\n'

    parsed = Interfile(header_content)
    with pytest.raises(FileNotFoundError) as e:
        parsed.get_data()

    # make sure it was the correct error
    assert 'source file' in str(e)


@pytest.mark.data
def test_Interfile_read_data_memmap(tmpdir):
    umap_hv = os.path.join(HERE, 'data', 'hoffrock', 'umap.hv')

    # header_f = tmpdir.join('interfile.h')
    # data_f = tmpdir.join('interfile.hx')

    # header_content = '\n'.join(t[1][0] for t in test_lines[:-1])
    # header_content += '\nname of data file := \\'
    # header_content += str(data_f).replace('/', '\\')
    # header_content += '\n'

    # print(header_content)

    # data_content = np.arange(10, dtype=PL_DTYPE)

    # header_f.write(header_content)
    # data_content.tofile(str(data_f))

    parsed = Interfile(source=str(umap_hv))
    parsed_data = parsed.get_data(memmap=True)
    # assert (parsed_data == data_content).all()
    assert parsed_data.size > 0
    assert isinstance(parsed_data, np.memmap)


def test_Interfile_maintains_inline_noninline():
    inline = [test for test in test_lines if test[0] == 'list'][0]
    noninline = [test for test in test_lines if test[0] == 'vector2'][0]

    inline_lines = len(str(Interfile(
        '\n'.join((test_lines[0][1][0],
                   inline[1][0],
                   test_lines[-1][1][0])))).splitlines())
    noninline_lines = len(str(Interfile(
        '\n'.join((test_lines[0][1][0],
                   noninline[1][0],
                   test_lines[-1][1][0])))).splitlines())
    assert noninline_lines > inline_lines
