"""Interfile I/O."""

import os
import sys
import ntpath
from collections import OrderedDict
from functools import reduce
import pyparsing as pp
try:
    import numpy as np
    from .petlink32 import DTYPE as PL_DTYPE
except ImportError:
    print("Warning: Can't import NumPy. Interfile data loading is "
          'unsupported.',
          file=sys.stderr)


class Interfile():
    """Read two-part (header + data) interfile images.

    Attributes:
    - sourcefile: filename of header souce
    - source: header file contents
    - header: parsed header contents as an OrderedDict
    - key_types: if keys have a prefix (e.g., '!', '%') they are
                 stored separately here.
    """

    bnf = None
    interfile_magic = '!INTERFILE'

    data_file_key = 'name of data file'
    offset_key = 'data offset in bytes'

    def __init__(self, interfile):
        """Read and/or parse interfile text.

        Inputs:
        - interfile (str or filename): Interfile header
        """
        # parser init: should only be required once
        if not self.bnf:
            self._initialise_bnf()

        try:
            # First, assume `interfile` is a filename.
            self.sourcefile = interfile
            if not os.path.exists(self.sourcefile):
                raise FileNotFoundError(self.sourcefile)

            with open(self.sourcefile, 'rt') as fp:
                self.source = fp.read()

        except (FileNotFoundError, OSError):
            # Otherwise, assume `interfile` is the header text.
            self.sourcefile = None
            self.source = interfile

        self.header, self.key_types = self.parse(self.source)

    def get_data(self, memmap=False):
        """Retrieve the image data. Optionally, may be returned as a
        numpy memmap rather than an array to avoid loading into
        memory.
        """
        # Don't bother if we never imported NumPy
        try:
            np
        except NameError:
            raise ImportError(
                'Unable to import NumPy: required for Interfile data '
                'reading.')

        # check whether the file is absolute or relative
        data_file = self.header[self.data_file_key]
        if not os.path.isabs(data_file):
            try:
                data_file = os.path.join(
                    os.path.dirname(self.sourcefile), data_file)
            except AttributeError:
                raise FileNotFoundError(
                    'Relative filenames are only supported when '
                    'source file is known.')

        if memmap:
            return np.memmap(
                data_file, dtype=PL_DTYPE, mode='r',
                offset=self.header.get(self.offset_key, 0))
        else:
            return np.fromfile(data_file,
                               #offset = self.header[self.offset_key],
                               dtype=PL_DTYPE)

    @classmethod
    def parse(cls, source):
        """Parse an interfile source.

        Inputs:
        - source (str): Interfile header

        Returns:
        - header (OrderedDict): Parsed source.
        - key_types (OrderedDict): If a key has a prefix (e.g., '%'),
                                   it will be stored here.
        """
        # Check magic
        magic = source[:len(cls.interfile_magic)].upper()
        if not magic == cls.interfile_magic:
            raise InvalidInterfileError('Interfile magic number missing')

        # Parse
        try:
            tokens = cls.bnf.parseString(source, parseAll=True)

        except pp.ParseException as e:
            # parse the exception and get the erronous line of source
            line_key = pp.Literal('line:').suppress()
            line_no = pp.Word(pp.nums).setParseAction(
                lambda t: int(t[0]))
            line = pp.SkipTo(line_key).suppress() + line_key + line_no
            error_line = line.parseString(str(e))[0] - 1

            sourcelines = source.splitlines()
            error_message = str(e) + '\n'
            if error_line > 1:
                error_message += '   ' + sourcelines[error_line - 1]
                error_message += '\n'

            error_message += '-> ' + sourcelines[error_line] + '\n'
            try:
                error_message += '   ' + sourcelines[error_line + 1]
                error_message += '\n'
            except IndexError: pass

            raise InvalidInterfileError(error_message)

        # Reformat
        header = OrderedDict()
        key_types = OrderedDict()
        for token in tokens:
            try:
                header[token['key'][0]] = token['value'].asList()

            except AttributeError:
                # not a list
                header[token['key'][0]] = token['value']

            key_types[token['key'][0]] = token['key_type']

        return header, key_types

    @classmethod
    def _initialise_bnf(cls):
        """Initialise a class-level parser, `bnf`, to parse Interfile.
        """
        pp.ParserElement.setDefaultWhitespaceChars(' \t')

        # constants
        semi       = pp.Literal(';')
        equals     = (pp.Optional(pp.Word(' ')).suppress() +
                      pp.Literal(':=').suppress() +
                      pp.Optional(pp.Word(' ')).suppress())
        list_start = pp.Literal('{').suppress()
        list_end   = pp.Literal('}').suppress()
        list_sep   = pp.Literal(',').suppress()
        path_sep   = pp.Literal(ntpath.sep)
        endl       = pp.LineEnd().suppress()

        # types
        int_ = (
            pp.Regex(r'[+-]?\d+')
            .setParseAction(lambda s, l, t: int(t[0])))
        float_ = (
            pp.Regex(r'[+-]?\d+\.\d*([eE]\d+)?')
            .setParseAction(lambda s, l, t: float(t[0])))

        # keys
        key_type = (pp.Optional(pp.oneOf('! %'), default='')
                    .setResultsName('key_type'))
        key = pp.SkipTo(equals).setResultsName('key')

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
        value = (
            pp.Optional(
                float_value | int_value | list_value | path_value |
                text_value,
                default='')
            .setResultsName('value'))

        keyDef = key_type + key + equals + value

        bnf =  pp.ZeroOrMore(pp.Group(keyDef))

        comment = semi + pp.restOfLine + pp.LineEnd().suppress()
        empty = pp.LineStart() + pp.LineEnd().suppress()
        bnf.ignore(comment)
        bnf.ignore(empty)

        cls.bnf = bnf

        # Undo previous state change
        pp.ParserElement.setDefaultWhitespaceChars(
            pp.ParserElement.DEFAULT_WHITE_CHARS)


class InvalidInterfileError(Exception): pass


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
