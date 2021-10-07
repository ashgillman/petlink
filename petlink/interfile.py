"""Interfile I/O."""

import logging
import os
import sys
import ntpath
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
    from pydicom.errors import InvalidDicomError
except ImportError as err:
    class InvalidDicomError(Exception):
        pass

from petlink import constants, ptd


Value = namedtuple('Value', 'value key_type units inline')
Value.__new__.__defaults__ = (None, '', None, True)


def load(filename):
    """Load interfile data, either from a plaintext header file or Siemens'
    .ptd format.

    This mostly exists to be similar in API to Nibabel.
    """
    return Interfile(source=filename)


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
    dcm = ptd.read_dcm(filename)

    source = dcm[constants.DCM_CSA_DATA_INFO].value.decode().rstrip('\0')
    interfile = Interfile(source)
    dtype = interfile.get_datatype()
    try:
        data = ptd.read_data(filename, dtype=dtype)
    except TypeError:
        # happens, e.g., with norm .ptd's
        data = None

    return dict(source=source, sourcefile=filename, data=data)


class Interfile(object):
    """Read two-part (header + data) interfile images.

    Attributes:
        sourcefile: Filename of header souce.
        source: Header file contents.
        header: Parsed header contents as an OrderedDict.
        data: A mutable data associated with the Interfile, contrast with
            get_data() which is always from file.

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

    # only need one class level parser, delay initialisation to first instance
    key_value_parser = None
    key_parser = None
    nonstrict_key_parser = None
    value_parser = None

    def __init__(self, source=None, header=None, data=None,
                 strict=True, do_clean=False):
        """Read and/or parse interfile text.

        Args:
            source: Source Intefile string to parse, or file to load and parse.
            header: An existing header OrderedDict.
            data: Interfile data, a numpy array.
            strict: Whether to parse as an Interfile (.hv, etc.) or
                Interfile-like (.par, etc.).
            do_clean: Whether  to apply  a cleanup  to the  Interfile contents.
        """
        # parser init: should only be required once
        if not self.key_value_parser:
            self._initialise_key_value_parser()
        if not self.key_parser:
            self._initialise_key_parser()
        if not self.nonstrict_key_parser:
            self._initialise_nonstrict_key_parser()
        if not self.value_parser:
            self._initialise_value_parser()

        self.header = header
        self._data = data
        self.strict = strict

        if os.path.exists(source):
            # load file and update attributes
            try:
                sourced_attrs = _from_ptd(source)
            except InvalidDicomError:
                sourced_attrs = _from_plaintext(source)
            for k, v in sourced_attrs.items():
                if getattr(self, k, None) is None:
                    setattr(self, k, v)

        else:
            self.source = source
            self.sourcefile = None

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

        if do_clean:
            self._cleanup()

    @property
    def data(self):
        if self._data is None:
            self._data = self.get_data()
        
        return self._data

    from_file = load

    def _cleanup(self):
        """Fix some common Interfile issues."""
        wrong = 'image data byte order'
        right = 'imagedata byte order'
        if wrong in self:
            tmp = self.header[wrong]
            self.insert_before(wrong, right, *tmp)
            del self.header[wrong]

        wrong = 'scale factor'
        right = 'scaling factor'
        if wrong in self:
            tmp = self.header[wrong]
            self.insert_before(wrong, right, *tmp)
            del self.header[wrong]

        wrong = 'data offset in bytes'
        if wrong in self:
            tmp = self.header[wrong]
            self[wrong] = Value(tmp.value, ';', tmp.units, tmp.inline)

        wrong = 'image relative start time'
        if (wrong in self
                and not isinstance(self[wrong], list)):
            tmp = self.header[wrong]
            self[wrong] = Value([tmp.value], tmp.key_type, tmp.units, False)

        wrong = 'patient orientation'
        if self.get(wrong, '').upper() == 'HFS':
            self[wrong] = 'head_in'
        elif self.get(wrong, '').upper() == 'FFS':
            self[wrong] = 'feet_in'

    def insert_before(self, existing_key, new_key, new_val,
                      new_key_type='', new_units=None, new_inline=False):
        """Insert a new_key and new_val before existing_key."""
        inserted_yet = False
        for key in list(self.header.keys()):
            if key == existing_key:
                self.header[new_key] = Value(
                    new_val, new_key_type, new_units, new_inline)
                inserted_yet = True

            if inserted_yet:
                self.header.move_to_end(key)

    def __copy__(self):
        return Interfile(str(self), data=self.get_data(), strict=self.strict)

    def get_datatype(self):
        if (self._data is not None
                and constants.IFL_DATA_FORMAT_KEY not in self
                and constants.IFL_DATA_SIZE_KEY not in self
                and constants.IFL_DATA_ORDER_KEY not in self):
            # we have a added data but have no type information at all
            return self._data.dtype

        else:
            format_ = self.get(constants.IFL_DATA_FORMAT_KEY,
                               constants.IFL_DATA_FORMAT_DEFAULT)
            size = self.get(constants.IFL_DATA_SIZE_KEY,
                            constants.IFL_DATA_SIZE_DEFAULT)
            order = self.get(constants.IFL_DATA_ORDER_KEY,
                             constants.IFL_DATA_ORDER_DEFAULT)

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

    def get_data_file(self):
        """Return an absolute path to the data file."""
        data_file = self[constants.IFL_DATA_FILE_KEY]
        if not os.path.isabs(data_file):
            if self.sourcefile is not None:
                data_file = os.path.join(
                    os.path.dirname(self.sourcefile), data_file)
            else:
                raise FileNotFoundError(
                    'Relative filenames are only supported when '
                    'source file is known.')
        return data_file

    def get_data(self, memmap=False, flat=False):
        """Retrieve the image data. Optionally, may be returned as a
        numpy memmap rather than an array to avoid loading into
        memory. Optionally may be flat, with shape ignored.
        Return is either a numpy array, or for norm files, a dict of arrays by
        normalization component.
        """
        # if self._data is not None:
        #     return self._data

        # Don't bother if we never imported NumPy
        try:
            np
        except NameError as err:
            raise Exception(
                'Unable to import NumPy: required for Interfile data reading.'
            ) from err

        # check whether the file is absolute or relative
        data_file = self.get_data_file()
        dtype = self.get_datatype()

        if memmap:
            data = np.memmap(
                data_file, dtype=dtype, mode='r',
                offset=self.header.get(constants.IFL_OFFSET_KEY, 0))
        else:
            data = np.fromfile(data_file, dtype=dtype)

        shape = self.get_shape(flat=flat)
        logger = logging.getLogger(__name__)
        logger.debug('Reshaping from {} to {}'.format(data.shape, shape))

        if isinstance(shape, tuple):
            data = data.reshape(shape, order='F')
        else:  # list of tuple, assume norm
            # Pull out and reshape each component, storing in a dict
            data, data_all = {}, data
            offset = 0
            for comp, comp_shape in zip(self['normalization component'], shape):
                end = offset + np.prod(comp_shape)
                data[comp] = data_all[offset:end].reshape(comp_shape, order='F')
                offset += np.prod(comp_shape)
            if offset != np.prod(data_all.shape):
                raise InvalidInterfileError(
                    'Data size differs from Interfile matrix size (%s, %s).',
                    offset, data_all.shape)

        return data

    def get_shape(self, flat=False):
        """Retrieve the data shape. Optionally may be flat, with shape (-1).
        """
        if 'matrix size' in self and not flat:
            depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

            if depth(self['matrix size']) == 1:
                shape = tuple(self['matrix size'])
            elif depth(self['matrix size']) == 2:
                shape = map(tuple, self['matrix size'])
            else:
                raise InvalidInterfileError(
                    'Unsure how to process shape %s', self['matrix size'])

        else:
            shape = (-1, )

        return shape

    # Time

    def get_datetime(self, key):
        """Get a Python datetime object for a give key of date and time."""
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

    # Accessing

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

    # Serialising

    def __str__(self):
        """Serialise to Interfile format."""
        content = '\n'.join(self.format_line(k, v)
                            for k, v in self.header.items())
        string = textwrap.dedent('''\
        {magic}{sep}
        {content}
        {magic_end}{sep}
        ''').format(content=content,
                    magic=constants.IFL_MAGIC,
                    magic_end=constants.IFL_MAGIC_END,
                    sep=constants.IFL_SEP)
        return string

    def __repr__(self):
        return "Interfile('{!s}')".format(str(self).replace('\n', '\\n'))

    def to_filename(self, filename):
        """Write interfile to file."""
        try:
            data = self.data
        except KeyError:
            data = None

        if isinstance(data, dict):
            data = np.concatenate([d.flatten(order='F') for d in data.values()])

        if data is not None:
            if filename[-3:-1] == '.h':  # *.h*
                data_file = filename[:-2] + filename[-1:]
            else:
                data_file = filename + '.dat'

            dtype = self.get_datatype()
            data.astype(dtype).flatten(order='F').tofile(data_file)

            data_file_relative = os.path.basename(data_file)

            temp_self = Interfile(str(self))
            temp_self.header['name of data file'] = Value(data_file_relative, '!')
            temp_self.header[constants.IFL_DATA_FORMAT_KEY] = Value(
                {
                    'i': 'signed integer',
                    'u': 'unsigned integer',
                    'f': 'float',
                }[dtype.kind])
            temp_self.header[constants.IFL_DATA_SIZE_KEY] = Value(
                dtype.itemsize, '!')
            temp_self.header[constants.IFL_DATA_ORDER_KEY] = Value(
                {
                    '<': 'LITTLEENDIAN',
                    '>': 'BIGENDIAN',
                    '=': sys.byteorder.upper() + 'ENDIAN',
                }[dtype.byteorder])
        else:
            temp_self = self

        with open(filename, 'w+') as fp:
            fp.write(str(temp_self))

    def format_line(self, key, value, idx=None):
        """Format an interfile line."""
        unit_str = (
            ' {units_start}{units}{units_end}'.format(
                units=value.units,
                units_start=constants.IFL_UNITS_START,
                units_end=constants.IFL_UNITS_END)
            if value.units is not None
            else '')
        idx_str = (
            ' {idx_start}{idx}{idx_end}'.format(
                idx_start=constants.IFL_INDEX_START, idx=idx,
                idx_end=constants.IFL_INDEX_END)
            if idx is not None
            else '')

        # special cases
        if isinstance(value.value, list):
            if value.inline:
                value_str = '{list_start} {list} {list_end}'.format(
                    list=', '.join(str(v) for v in value.value),
                    list_start=constants.IFL_LIST_START,
                    list_end=constants.IFL_LIST_END)
            else:
                return self._format_multiline(key, value)

        elif value.value is None:
            value_str = constants.IFL_NONE

        else:
            value_str = str(value.value)

        return '{key_type}{key}{units}{idx} {sep} {value}'.format(
            key_type=value.key_type, key=key, units=unit_str, idx=idx_str,
            value=value_str, sep=constants.IFL_SEP)

    def _format_multiline(self, key, value):
        """Used by format_line to format non-inline vectors."""
        return '\n'.join(
            self.format_line(
                key,
                Value(value=val, key_type=value.key_type, units=value.units,
                      inline=True),
                idx=idx+1)
            for idx, val in enumerate(value.value))

    # Parsing

    def _parse(self, source):
        """Parse an interfile source.

        Inputs:
        - source (str): Interfile header

        Returns:
        - header (OrderedDict): Parsed source.
        """
        # Check magic
        magic = source[:len(constants.IFL_MAGIC)].upper()
        if not magic == constants.IFL_MAGIC and self.strict:
            raise InvalidInterfileError('Interfile magic number missing')

        # Top-level parse
        try:
            tokens = self.key_value_parser.parseString(source, parseAll=True)
        except pp.ParseException as err:
            error_message = str(err) + '\n'
            error_message += _get_parse_exception_context(err, source)
            raise InvalidInterfileError(error_message) from err

        # Now parse each individual key and value
        header = OrderedDict()
        for token in tokens:
            try:
                if self.strict:
                    key_token = self.key_parser.parseString(token.key)
                else:
                    key_token = self.nonstrict_key_parser.parseString(
                        token.key)
                value_token = self.value_parser.parseString(token.value)
            except pp.ParseException as err:
                error_message = str(err) + '\n'
                error_message += 'Error on: ' + token.key + ' : ' + token.value
                raise InvalidInterfileError(error_message) from err

            # Special cases
            if key_token.key == constants.IFL_MAGIC_END[1:].lower():
                break
            if value_token.value == constants.IFL_NONE:
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
                indexed_value = (key_token.index, value)
                try:
                    value = header[key_token.key].value + [indexed_value]
                except KeyError:
                    value = [indexed_value]

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
        units_start = pp.Literal(constants.IFL_UNITS_START).suppress()
        units_end = pp.Literal(constants.IFL_UNITS_END).suppress()
        index_start = pp.Literal(constants.IFL_INDEX_START).suppress()
        index_end = pp.Literal(constants.IFL_INDEX_END).suppress()
        chars = ''.join(c for c in pp.printables
                        if c not in (constants.IFL_UNITS_START
                                     + constants.IFL_INDEX_START))
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
    def _initialise_nonstrict_key_parser(cls):
        """Initialise a class-level parser, `nonstrict_key_parser`, to parse
        Interfile keys without units, keys, indices.
        """
        chars = pp.printables
        endl = pp.LineEnd().suppress()

        key = (pp.OneOrMore(pp.Word(chars))
               .setParseAction(' '.join)
               .addParseAction(pp.downcaseTokens)
               .addParseAction(lambda t: t[0].strip())
               .setResultsName('key'))

        nonstrict_key_parser = key + endl
        cls.nonstrict_key_parser = nonstrict_key_parser

    @classmethod
    def _initialise_value_parser(cls):
        """Initialise a class-level parser, `value_parser`, to parse Interfile
        keys.
        """
        list_start = pp.Literal(constants.IFL_LIST_START).suppress()
        list_end = pp.Literal(constants.IFL_LIST_END).suppress()
        path_sep = pp.Literal(ntpath.sep)
        endl = pp.LineEnd().suppress()

        # types
        int_ = pp.Regex(r'[+-]?\d+').setParseAction(lambda t: int(t[0]))
        float_ = (pp.Regex(r'[+-]?\d+\.\d*([eE][+-]?\d+)?')
                  .setParseAction(lambda t: float(t[0])))
        text_ = pp.Regex(r'[^;]+')
        list_text_ = pp.Regex(r'[^;,}]+').setParseAction(lambda t: t[0].strip())

        # value parsers
        cls._int_parser = int_ + endl
        cls._float_parser = float_ + endl
        cls._list_parser = \
            list_start \
            + pp.Group(pp.delimitedList(float_ | int_ | list_text_)) \
            + list_end + endl
        cls._path_parser = \
            (path_sep + pp.Word(pp.printables) + endl) \
            .setParseAction(_parse_interfile_path)
        cls._text_parser = text_ + endl

        value_parser = (
            pp.Optional(
                cls._float_parser
                | cls._int_parser
                | cls._list_parser
                | cls._path_parser
                | cls._text_parser,
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
        #           pp.Literal(constants.IFL_SEP).suppress() +
        #           pp.Optional(pp.Word(' ')).suppress())
        equals = pp.Literal(constants.IFL_SEP).suppress()
        endl = pp.LineEnd().suppress()
        magic = pp.CaselessLiteral(constants.IFL_MAGIC)

        key = (pp.SkipTo(equals, include=True)
               .setParseAction(pp.downcaseTokens)
               .addParseAction(lambda t: t[0].strip())
               .setResultsName('key'))
        value = (pp.SkipTo(endl, include=True)
                 .setParseAction(lambda t: t[0].strip())
                 .setResultsName('value'))

        start_of_file = (magic + equals + endl).suppress()
        key_values = pp.ZeroOrMore(pp.Group(key + value))
        comment = semi + pp.restOfLine + endl
        empty = pp.LineStart().leaveWhitespace() + endl

        # E7 Tools norm files
        zero_termination = (pp.Literal('\x00') + pp.LineEnd()).suppress()

        key_value_parser = pp.Optional(start_of_file) \
            + key_values \
            + pp.Optional(zero_termination)
        key_value_parser.ignore(comment)
        key_value_parser.ignore(empty)

        cls.key_value_parser = key_value_parser

        # Undo previous state change
        pp.ParserElement.setDefaultWhitespaceChars(
            pp.ParserElement.DEFAULT_WHITE_CHARS)


class InvalidInterfileError(Exception):
    pass


def _parse_interfile_path(tokens):
    """Helper for pyparsing. Paths are saved in Windows format, make
    native.
    """
    path = ''.join(tokens)
    path = path.replace(ntpath.sep + ntpath.sep, ntpath.sep)
    folders = _split_ntpath(path)
    if folders[0] == ntpath.sep:
        folders[0] = os.path.sep
    return reduce(os.path.join, folders)


def _split_ntpath(path):
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


def _get_parse_exception_context(error, source):
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
