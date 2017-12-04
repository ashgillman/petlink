"""Interfile I/O."""

import logging

import os
import sys
import ntpath
import mmap
import datetime
from collections import OrderedDict, namedtuple
import textwrap
from functools import reduce
import pyparsing as pp

try:
    import numpy as np
except ImportError as err:
    print("Warning: Can't import NumPy. Interfile data loading is "
          'unsupported.',
          file=sys.stderr)
    np_err = err

try:
    import dicom
    from dicom.errors import InvalidDicomError
except ImportError as err:
    print("Warning: Can't import PyDICOM. .ptd data loading is unsupported.",
          file=sys.stderr)
    dicom_err = err

    class InvalidDicomError(Exception):
        pass

from petlink import constants


Value = namedtuple('Value', 'value key_type units inline')
Value.__new__.__defaults__ = (None, '', None, False)


def load(filename):
    """Load interfile data, either from a plaintext header file or Siemens'
    .ptd format.

    This mostly exists to be similar in API to Nibabel.
    """
    return Interfile(sourcefile=filename)


def load_plaintext(filename):
    """Load interfile data, from a plaintext header file."""
    return Interfile(**_from_plaintext(filename))


def _from_plaintext(filename):
    """Load interfile creation parameters, from a plaintext header file."""
    try:
        with open(filename, 'rt') as fp:
            source = fp.read(constants.IFL_MAX_HEADER_SIZE)
    except TypeError:
        raise FileNotFoundError(filename)

    return dict(source=source, sourcefile=filename)


def load_ptd(filename):
    """Load interfile data, from Siemens' .ptd format.

    This format consists of the first part as a data file, and the latter part
    as a DICOM file.
    """
    return Interfile(**_from_ptd(filename))


def _from_ptd(filename):
    with open(filename, 'rb') as fp:
        # skip to DICOM header
        try:
            fp.seek(
                mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                .find(constants.DCM_MAGIC))
        except OSError as err:
            raise InvalidDicomError("Can't find DICOM magic") from err

        data_length = fp.tell()

        # read the remaining file as DICOM
        dcm = dicom.filereader.read_file(fp)

    # TODO?: dcm[CSA_IMAGE_HEADER_INFO].value

    source = dcm[constants.DCM_CSA_DATA_INFO].value.decode().rstrip('\0')
    interfile = Interfile(source)
    dtype = interfile.get_datatype()
    try:
        data = np.memmap(
            filename, mode='r', shape=(data_length / dtype.itemsize, ),
            dtype=dtype,
            offset=interfile.header.get(Interfile.OFFSET_KEY, 0).value)
    except TypeError:
        # happens, e.g., with norm .ptd's
        data = None

    return dict(source=source, sourcefile=filename, data=data, dcm=dcm)


class Interfile(object):
    """Read two-part (header + data) interfile images.

    Attributes:
      sourcefile: Filename of header souce.
      source: Header file contents.
      header: Parsed header contents as an OrderedDict.
      dcm: If read from a .ptd file, the DICOM data.

    To access values, index the Interfile object directly. Use the header
    attribute only if access to metadata is required.

    >>> if_src = '''!INTERFILE:=
    !name of data file := <NONE>
    %image orientation:={1,0,0,0,1,0}
    ; ...
    '''
    >>> if = Interfile(if_src)
    >>> if['image orientation']
    [1, 0, 0, 0, 1, 0]
    >>> if.header['image orientation'].inline
    True
    >>> if.header['image orientation'].key_type
    '%'
    """

    key_value_parser = None
    key_parser = None
    value_parser = None

    INTERFILE_MAGIC = '!INTERFILE'
    INTERFILE_MAGIC_END = '!END OF INTERFILE'
    INTERFILE_NONE = '<NONE>'
    INTERFILE_SEP = ':='
    INTERFILE_LIST_START = '{'
    INTERFILE_LIST_END = '}'
    INTERFILE_UNITS_START = '('
    INTERFILE_UNITS_END = ')'
    INTERFILE_INDEX_START = '['
    INTERFILE_INDEX_END = ']'

    DATA_FILE_KEY = 'name of data file'
    OFFSET_KEY = 'data offset in bytes'

    DATA_FORMAT_KEY = 'number format'
    DATA_FORMAT_DEFAULT = 'UNSIGNED INTEGER'
    DATA_SIZE_KEY = 'number of bytes per pixel'
    DATA_SIZE_DEFAULT = 4
    DATA_ORDER_KEY = 'imagedata byte order'
    DATA_ORDER_DEFAULT = 'LITTLEENDIAN'

    def __init__(self, source=None, sourcefile=None, header=None, data=None,
                 dcm=None):
        """Read and/or parse interfile text.

        Inputs:
        - interfile (str or filename): Interfile header
        """
        # parser init: should only be required once
        if not self.key_value_parser:
            self._initialise_key_value_parser()
        if not self.key_parser:
            self._initialise_key_parser()
        if not self.value_parser:
            self._initialise_value_parser()

        self.source = source
        self.sourcefile = sourcefile
        self.header = header
        self._data = data
        self.dcm = dcm

        if sourcefile and not source:
            # Read from sourcefile and set any unset attributes.
            try:
                sourced_attrs = _from_ptd(sourcefile)
            except InvalidDicomError:
                sourced_attrs = _from_plaintext(sourcefile)
            for k, v in sourced_attrs.items():
                if getattr(self, k, None) is None:
                    setattr(self, k, v)

        if self.source:
            # Parse the source.
            try:
                self.header = self._parse(self.source)
            except InvalidInterfileError as err:
                logger = logging.getLogger(__name__)
                logger.error("Couldn't parse:\n%s", repr(self.source))
                raise err
        else:
            logger = logging.getLogger(__name__)
            logger.warn('Interfile has no source, so header is None.')
            logger.warn(self.source)

    def __copy__(self):
        return Interfile(str(self))

    def get_datatype(self):
        if (self._data is not None
                and self.DATA_FORMAT_KEY not in self
                and self.DATA_SIZE_KEY not in self
                and self.DATA_ORDER_KEY not in self):
            # we have a added data but no type information at all
            return self._data.dtype

        else:
            format_ = self.get(self.DATA_FORMAT_KEY, self.DATA_FORMAT_DEFAULT)
            size = self.get(self.DATA_SIZE_KEY, self.DATA_SIZE_DEFAULT)
            order = self.get(self.DATA_ORDER_KEY, self.DATA_ORDER_DEFAULT)

            format_ = {
                'UNSIGNED INTEGER': 'u',
                'SIGNED INTEGER': 'i',
                'INTEGER': 'i',
                'FLOAT': 'f',
            }[format_.upper()]
            order = {
                'LITTLEENDIAN': '<',
                'BIGENDIAN': '>',
            }[order.upper()]

            return np.dtype(order + format_ + str(size))

    def get_data(self, memmap=False):
        """Retrieve the image data. Optionally, may be returned as a
        numpy memmap rather than an array to avoid loading into
        memory.
        """
        if self._data is not None:
            return self._data

        # Don't bother if we never imported NumPy
        try:
            np
        except NameError as err:
            raise Exception(
                'Unable to import NumPy: required for Interfile data reading.'
            ) from err

        # check whether the file is absolute or relative
        data_file = self[self.DATA_FILE_KEY]
        if not os.path.isabs(data_file):
            try:
                data_file = os.path.join(
                    os.path.dirname(self.sourcefile), data_file)
            except AttributeError as err:
                raise FileNotFoundError(
                    'Relative filenames are only supported when '
                    'source file is known.') from err
        dtype = self.get_datatype()

        if memmap:
            return np.memmap(
                data_file, dtype=dtype, mode='r',
                offset=self.header.get(self.OFFSET_KEY, 0))
        else:
            return np.fromfile(data_file, dtype=dtype)

    def get_datetime(self, key):
        logger = logging.getLogger(__name__)
        # load data and time
        date_v = self.header[key.lower() + ' date']

        time_v = self.header[key.lower() + ' time']

        date_fmt, time_fmt = self._get_date_time_formats_from_units(
            date_v, time_v)

        # load time zone
        # e.g., GMT+10:00, gmt+00:00
        tz_idx = time_v.units.lower().find('gmt')
        if tz_idx >= 0:
            tz_str = time_v.units[tz_idx+3:tz_idx+9].replace(':', '')
        else:
            tz_str = '+0000'

        # parse date and time
        logger.debug(
            'Parsing %s', ' '.join((date_v.value, time_v.value, tz_str)))
        datetime_ = datetime.datetime.strptime(
            ' '.join((date_v.value, time_v.value, tz_str)),
            ' '.join((date_fmt,     time_fmt,     '%z')))

        return datetime_

    def set_datetime(self, key, new):
        # load data and time
        date_v = self.header[key.lower() + ' date']

        time_v = self.header[key.lower() + ' time']

        date_fmt, time_fmt = self._get_date_time_formats_from_units(
            date_v, time_v)

        # load time zone
        # e.g., GMT+10:00
        tz_idx = time_v.units.lower().find('gmt')
        if tz_idx >= 0:
            tz_str = time_v.units.lower()[tz_idx:]
            tz_offset = datetime.timedelta(hours=int(tz_str[3:6]),
                                           minutes=int(tz_str[7:9]))
        else:
            tz_offset = datetime.timedelta()
        tz = datetime.timezone(tz_offset)

        # set date and time
        self[key.lower() + ' date'] = new.astimezone(tz).strftime(date_fmt)
        self[key.lower() + ' time'] = new.astimezone(tz).strftime(time_fmt)

    def _get_date_time_formats_from_units(self, date_value, time_value):
        date_fmt = (date_value.units
                    .lower()
                    .strip()
                    .replace('yyyy', '%Y')
                    .replace('yy',   '%y')
                    .replace('mm',   '%m')
                    .replace('dd',   '%d'))
        time_fmt = (time_value.units
                    .lower()
                    .split()[0]
                    .strip()
                    .replace('hh', '%H')
                    .replace('mm', '%M')
                    .replace('ss', '%S'))
        return date_fmt, time_fmt

    def __getitem__(self, key):
        """Indexing is shorthand to get actual value from header. It is also
        caseless."""
        return self.header[key.lower()].value

    def get(self, key, default=None):
        """Indexing with a default."""
        return self[key] if key in self else default

    def __contains__(self, key):
        """Shortcut for the wrapped header dict."""
        return key in self.header

    def __setitem__(self, key, value):
        """Set an item of the header attribute.

        Inputs:
        - key: key
        - value: value or Value object. If value, default Value arguments are
        used.
        """
        if isinstance(value, Value):
            # User has done the work
            self.header[key] = value
        elif key in self.header:
            # replace the value for an existing key, retaining metadata
            meta = self.header[key]._asdict()
            del meta['value']
            self.header[key] = Value(value=value, **meta)
        else:
            # create a new key with default metadata
            self.header[key] = Value(value=value)

    def __str__(self):
        """Serialise to Interfile format."""
        content = '\n'.join(self.format_line(k, v)
                            for k, v in self.header.items())
        string = textwrap.dedent('''\
        {magic}{sep}
        {content}
        {magic_end}{sep}
        ''').format(content=content,
                    magic=self.INTERFILE_MAGIC,
                    magic_end=self.INTERFILE_MAGIC_END,
                    sep=self.INTERFILE_SEP)
        return string

    def __repr__(self):
        return "Interfile('{!s}')".format(str(self).replace('\n', '\\n'))

    def to_filename(self, filename):
        """Write interfile to file."""
        try:
            data = self.get_data()
        except KeyError:
            data = None

        if data is not None:
            if filename[-3:-1] == '.h':  # *.h*
                data_file = filename[:-2] + filename[-1:]
            else:
                data_file = filename + '.dat'

            dtype = self.get_datatype()
            data.astype(dtype).tofile(data_file)

            temp_self = Interfile(str(self))
            temp_self.header['name of data file'] = Value(data_file, '!')
            temp_self.header[self.DATA_FORMAT_KEY] = Value(
                {
                    'i': 'SIGNED INTEGER',
                    'u': 'UNSIGNED INTEGER',
                    'f': 'FLOAT',
                }[dtype.kind])
            temp_self.header[self.DATA_SIZE_KEY] = Value(dtype.itemsize, '!')
            temp_self.header[self.DATA_ORDER_KEY] = Value(
                {
                    '<': 'LITTLEENDIAN',
                    '>': 'BIGENDIAN',
                    '=': sys.byteorder.upper() + 'ENDIAN',
                }[dtype.byteorder])
        else:
            temp_self = self

        with open(filename, 'w+') as fp:
            fp.write(str(temp_self))

    def format_line(self, key, value):
        """Format an interfile line."""
        unit_str = (
            ' {units_start}{units}{units_end}'.format(
                units=value.units,
                units_start=self.INTERFILE_UNITS_START,
                units_end=self.INTERFILE_UNITS_END)
            if value.units is not None
            else '')
        key_str = '{key_type}{key}{units}'.format(
            key_type=value.key_type, key=key, units=unit_str)

        # special cases
        if isinstance(value.value, list):
            if value.inline:
                value_str = '{list_start} {list} {list_end}'.format(
                    list=', '.join(str(v) for v in value.value),
                    list_start=self.INTERFILE_LIST_START,
                    list_end=self.INTERFILE_LIST_END)
            else:
                return self._format_multiline(key_str, value)

        elif value.value is None:
            value_str = self.INTERFILE_NONE

        else:
            value_str = str(value.value)

        return '{key} {sep} {value}'.format(
            key=key_str, value=value_str, sep=self.INTERFILE_SEP)

    def _format_multiline(self, key_str, value):
        """Used by format_line to format non-inline vectors."""
        return '\n'.join(
            '{key} {index_start}{idx}{index_end} {sep} {value!s}'.format(
                key=key_str, idx=idx+1, value=val, sep=self.INTERFILE_SEP,
                index_start=self.INTERFILE_INDEX_START,
                index_end=self.INTERFILE_INDEX_END)
            for idx, val in enumerate(value.value))

    @classmethod
    def _parse(cls, source):
        """Parse an interfile source.

        Inputs:
        - source (str): Interfile header

        Returns:
        - header (OrderedDict): Parsed source.
        - key_types (OrderedDict): If a key has a prefix (e.g., '%'),
                                   it will be stored here.
        """
        # Check magic
        magic = source[:len(cls.INTERFILE_MAGIC)].upper()
        if not magic == cls.INTERFILE_MAGIC:
            raise InvalidInterfileError('Interfile magic number missing')

        # Top-level parse
        try:
            tokens = cls.key_value_parser.parseString(source, parseAll=True)
        except pp.ParseException as err:
            error_message = str(err) + '\n'
            error_message += get_parse_exception_context(err, source)
            raise InvalidInterfileError(error_message) from err

        # Now parse each individual key and value
        header = OrderedDict()
        for token in tokens:
            try:
                key_token = cls.key_parser.parseString(token.key)
                value_token = cls.value_parser.parseString(token.value)
            except pp.ParseException as err:
                error_message = str(err) + '\n'
                error_message += token.key + ' : ' + token.value
                raise InvalidInterfileError(error_message) from err

            # Special cases
            if key_token.key == cls.INTERFILE_MAGIC_END[1:].lower():
                break
            if value_token.value == cls.INTERFILE_NONE:
                value_token.value = None

            try:
                value = value_token.value.asList()
            except AttributeError:
                # not a list
                value = value_token.value

            # If we have a non-inline vector, store it as a 2-tuple with index
            # first. We will parse the index out later.
            if 'index' in key_token:
                # TODO, this should fail since we now parse index as ZeroOrMore
                indexed = (key_token.index, value_token.value)
                try:
                    value = header[key_token.key].value + [indexed]
                except KeyError:
                    value = [indexed]

            header[key_token.key] = Value(
                key_type=key_token.key_type, value=value,
                units=(key_token.units if 'units' in key_token else None),
                inline=('index' not in key_token))

        # any non-inline vectors need to be de-indexed
        for key, value in header.items():
            if (isinstance(value.value, list)
                    and isinstance(value.value[0], tuple)):
                val_dict = value._asdict()
                del val_dict['value']
                header[key] = Value(
                    value=[v[1] for v in sorted(value.value)],
                    **val_dict)

        return header

    @classmethod
    def _initialise_key_parser(cls):
        """Initialise a class-level parser, `key_parser`, to parse Interfile
        keys.
        """
        important_key = pp.Literal('!')
        custom_key = pp.Literal('%')
        units_start = pp.Literal(cls.INTERFILE_UNITS_START).suppress()
        units_end = pp.Literal(cls.INTERFILE_UNITS_END).suppress()
        index_start = pp.Literal(cls.INTERFILE_INDEX_START).suppress()
        index_end = pp.Literal(cls.INTERFILE_INDEX_END).suppress()
        chars = ''.join(c for c in pp.printables
                        if c not in (cls.INTERFILE_UNITS_START
                                     + cls.INTERFILE_INDEX_START))
        endl = pp.LineEnd().suppress()

        key_type = (pp.Optional(important_key | custom_key, default='')
                    .setResultsName('key_type'))
        key = (pp.OneOrMore(pp.Word(chars))
               .setParseAction(' '.join)
               .addParseAction(pp.downcaseTokens)
               .addParseAction(lambda t: t[0].strip())
               .setResultsName('key'))
        units = pp.Optional(
            (units_start + pp.SkipTo(units_end, include=True))
            .setParseAction(lambda t: t[0])
            .setResultsName('units'))
        index = pp.ZeroOrMore(
            (index_start + pp.Word(pp.nums) + index_end)
            .setParseAction(lambda t: int(t[0]))
            .setResultsName('index'))

        key_parser = key_type + key + units + index + endl
        cls.key_parser = key_parser

    @classmethod
    def _initialise_value_parser(cls):
        """Initialise a class-level parser, `value_parser`, to parse Interfile
        keys.
        """
        list_start = pp.Literal(cls.INTERFILE_LIST_START).suppress()
        list_end = pp.Literal(cls.INTERFILE_LIST_END).suppress()
        path_sep = pp.Literal(ntpath.sep)
        endl = pp.LineEnd().suppress()

        # types
        int_ = pp.Regex(r'[+-]?\d+').setParseAction(lambda t: int(t[0]))
        float_ = (pp.Regex(r'[+-]?\d+\.\d*([eE][+-]?\d+)?')
                  .setParseAction(lambda t: float(t[0])))

        # values
        int_value = int_ + endl
        float_value = float_ + endl
        list_value = ((list_start +
                       pp.Group(pp.delimitedList(int_ | float_)) +
                       list_end + endl)
                      .setParseAction(lambda s, l, t: t))
        path_value = (
            (path_sep + pp.Word(pp.printables) + endl)
            .setParseAction(parse_interfile_path))
        text_value = pp.restOfLine + endl

        value_parser = (
            pp.Optional(
                float_value | int_value | list_value | path_value |
                text_value,
                default='')
            .setResultsName('value'))
        cls.value_parser = value_parser

    @classmethod
    def _initialise_key_value_parser(cls):
        """Initialise a class-level parser, `key_value_parser`, to parse
        Interfile lines into key-value pairs. Comments are ignored.
        """
        # newline is significant
        pp.ParserElement.setDefaultWhitespaceChars(' \t\r')

        # constants
        semi = pp.Literal(';')
        # equals = (pp.Optional(pp.Word(' ')).suppress() +
        #           pp.Literal(cls.INTERFILE_SEP).suppress() +
        #           pp.Optional(pp.Word(' ')).suppress())
        equals = pp.Literal(cls.INTERFILE_SEP).suppress()
        endl = pp.LineEnd().suppress()
        magic = pp.CaselessLiteral(cls.INTERFILE_MAGIC)

        key = (pp.SkipTo(equals, include=True)
               .setParseAction(pp.downcaseTokens)
               .addParseAction(lambda t: t[0].strip())
               .setResultsName('key'))
        value = (pp.SkipTo(endl, include=True)
                 .setParseAction(lambda t: t[0].strip())
                 .setResultsName('value'))

        sof = (magic + equals + endl).suppress()
        key_values = pp.ZeroOrMore(pp.Group(key + value))
        comment = semi + pp.restOfLine + endl
        empty = pp.LineStart().leaveWhitespace() + endl

        key_value_parser = sof + key_values
        key_value_parser.ignore(comment)
        key_value_parser.ignore(empty)

        cls.key_value_parser = key_value_parser

        # Undo previous state change
        pp.ParserElement.setDefaultWhitespaceChars(
            pp.ParserElement.DEFAULT_WHITE_CHARS)


class InvalidInterfileError(Exception):
    pass


def parse_interfile_path(tokens):
    """Helper for pyparsing. Paths are saved in Windows format, make
    native.
    """
    path = ''.join(tokens)
    path = path.replace(ntpath.sep + ntpath.sep, ntpath.sep)
    folders = split_ntpath(path)
    if folders[0] == ntpath.sep:
        folders[0] = os.path.sep
    return reduce(os.path.join, folders)


def split_ntpath(path):
    """Split a windows path into a list of folders (and a file)."""
    folders = []
    path, folder = ntpath.split(path)

    while folder != '':
        folders.append(folder)
        path, folder = ntpath.split(path)

    if path != '':
        folders.append(path)

    folders.reverse()
    return folders


def get_parse_exception_context(error, source):
    """Parse the exception and print the erronous lines of source."""
    line_key = pp.Literal('line:').suppress()
    line_no = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
    line = pp.SkipTo(line_key).suppress() + line_key + line_no

    error_line_number = line.parseString(str(error))[0] - 1
    error_message = ''

    sourcelines = source.splitlines()
    if error_line_number > 1:
        error_message += '   ' + sourcelines[error_line_number - 1]
        error_message += '\n'

    error_message += '-> ' + sourcelines[error_line_number] + '\n'
    try:
        error_message += '   ' + sourcelines[error_line_number + 1]
        error_message += '\n'
    except IndexError:
        pass

    return error_message
