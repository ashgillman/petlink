"""Convenient accessors for CSA DICOM fields.

Authored by Ashley Gillman. CSA header reading adapted from Parnesh Raniga.
"""


import logging
import os
import struct
import io
import collections
import numpy as np
import dicom

from petlink import interfile, constants
from petlink.helpers import dicomhelper


class InterfileCSA(object):
    """Read CSA DICOM files. These may be .IMA, .dcm and .bf pairs, or .ptd
    files.

    Attributes:
        dcm: DICOM header, a pydicom Dataset object.
        ifl: Interfile header, an Interfile object (only if .dcm+.bf or .ptd).
        data: Interfile data, a Numpy ndarray (only if .dcm+.bf or .ptd).
        csa_header: CSA header ad a dict (only if .IMA).

    Supported are:
      - Image files with Siemens CSA headers that need to be read (.IMA).
      - DICOM files that wrap Interfile objects (i.e., PET listmode, sinogram
        and norm files, .dcm+.bf or .ptd).
    """

    @property
    def data(self):
        return self._data

    def __init__(self, dcm, data=None):
        """Create a InterfileCSA instance. If loading from a file, can also use
        InterfileCSA.from_file().

        Args:
            dcm: DICOM header for data or a path pointing to DICOM file.
        """
        logger = logging.getLogger(__name__)

        # Parse given dcm to a Dataset

        if isinstance(dcm, dicom.dataset.Dataset):
            self.dcm = dcm

        elif isinstance(dcm, str) and os.path.exists(dcm):
            self.dcm = dicom.read_file(dcm)

        else:
            raise ValueError("Can't parse DICOM input.")

        # Save data, or check dcm for data

        if isinstance(data, str) and os.path.exists(data):
            # load data from file
            self._data = np.memmap(data, mode='r',
                                   dtype=self.ifl.dtype)
        elif data is not None:
            # data was passed
            self._data = data
        elif constants.DCM_CSA_DATA in self.dcm:
            # load data from dcm
            self._data = np.fromstring(self.dcm[constants.DCM_CSA_DATA].value,
                                       dtype=self.ifl.dtype)

    # IO

    @staticmethod
    def from_ptd(ptd_file):
        """Load a InterfileCSA object from a .ptd file."""
        csa = InterfileCSA(ptd.read_dcm(ptd_file))
        return InterfileCSA(data=ptd.read_data(ptd_file, dtype=csa.ifl.dtype),
                            dcm=csa.dcm)

    @staticmethod
    def from_file(filename, force_type=None):
        """Load a InterfileCSA object from a .ptd file. Optionally force filetype
        reading with force_type to 'ptd' or 'dcm'.
        """
        if force_type == 'ptd' or filename.lower().endswith('.ptd'):
            return InterfileCSA.from_ptd(filename)
        if force_type == 'dcm' or (filename.lower().endswith('.dcm')
                                   or filename.lower().endswith('.ima')):
            bf_file = filename[:-4] + '.bf'
            if os.path.exists(bf_file):
                return InterfileCSA(dcm=filename, data=bf_file)
            else:
                return InterfileCSA(dcm=filename)
        else:
            raise RuntimeError('Unkown filetype for {}.'.format(
                force_type or filename))

    def to_ptd(self, filename):
        ptd.write_ptd(self, filename)

    # Interfile

    @property
    def ifl(self):
        """read Siemens CSA header as interfile."""
        if not hasattr(self, '_ifl'):
            # Parse or extract ifl as an Interfile
            if constants.DCM_CSA_DATA_INFO in self.dcm:
                ifl_source = dicomhelper.decode_ob_header(
                    self.dcm[constants.DCM_CSA_DATA_INFO].value)
                self._ifl = interfile.Interfile(source=ifl_source)

            else:
                raise ValueError("No CSA header to find interfile in.")

        return self._ifl

    def to_interfile(self, basename):
        img_type = self.dcm.ImageType[-1]
        if 'LISTMODE' in img_type:
            header_ext = '.hl'
            data_ext = '.l'
        elif 'NORM' in img_type:
            header_ext = '.hn'
            data_ext = '.n'
        elif 'SINO' in img_type:
            header_ext = '.hs'
            data_ext = '.s'

        # correct name of data file tag
        new_ifl = copy.copy(self.ifl)

        old_data_file_full = header['name of data file']
        old_data_file_short = (old_data_file_full
                               .replace('/', '\\')
                               .split('\\'))[-1]
        new_ifl = Interfile(
            source=(str(self.ifl)
                    .replace(old_data_file_full, os.path.abspath(data_file))
                    .replace(old_data_file_short, os.path.abspath(data_file))))

        new_ifl.to_filename(basename + header_ext)
        self.data.tofile(basename + data_ext)
        return basename + header_ext

    # CSA Header

    @property
    def csa_header(self):
        """Read the Siemens CSA Header from the DICOM as a dict."""
        # cache
        if not hasattr(self, '_csa_header'):
            # TODO: check (0029,0010) is 'SIEMENS CSA HEADER'?
            header_data = self.dcm[constants.DCM_CSA_DATA_INFO].value
            self._csa_header = self._read_csa_header(header_data)

        return self._csa_header

    def _read_csa_header(self, dcm_value):
        csa_raw = io.BytesIO(dcm_value)
        id, _, ntags, _ = struct.unpack('<4s4sII', csa_raw.read(4+4+4+4))
        assert id == b'SV10', 'Only know how to unpack SV10.'

        header = collections.OrderedDict()
        for _ in range(ntags):
            tag = self._read_csa_tag(csa_raw)
            header[tag['name']] = tag

        return header

    def _read_csa_tag(self, csa_raw):
        """Read SIEMENS CSA HEADER from raw bytes in DICOM, return a dict."""
        # unpack
        raw_name, vm, raw_vr, syngodt, nitems, etag = struct.unpack(
            '<64si4siii', csa_raw.read(64+4+4+4+4+4))

        # decode strings
        name = raw_name.split(b'\x00')[0].decode('ascii')
        vr = raw_vr.strip(b'\x00').decode('ascii')

        # form tags
        tag = dict(name=name, vm=vm, vr=vr, syngodt=syngodt, nitems=nitems,
                   etag=etag, items=[])
        for item in range(tag['nitems']):
            tag['items'].append(self._read_csa_tag_item(csa_raw))

        return tag

    def _read_csa_tag_item(self, csa_raw):
        """Read an individual item from a CSA Header."""
        len_bound = struct.unpack('<'+4*'i', csa_raw.read(16))
        len_item = len_bound[1]

        # short circuit
        if len_item == 0:
            return ''

        # find end
        cur = csa_raw.tell()
        csa_raw.seek(0, os.SEEK_END)
        end = csa_raw.tell()
        csa_raw.seek(cur, os.SEEK_SET)

        # read out value
        if len_item > end - cur:
            ## read in however many bytes we have
            val = struct.unpack(
                '<'+str(end-cur)+'s', csa_raw.read(len_item))[0]
        else:
            val = struct.unpack(
                '<'+str(len_item)+'s', csa_raw.read(len_item))[0]

        # ensure a valid position
        cur = csa_raw.tell()
        if cur%4 != 0:
            csa_raw.seek(4-(cur%4), os.SEEK_CUR)

        return val.strip(b'\x00').decode('ascii')

    # DICOM helpers

    def get_datetime(self, which_time='acquisition'):
        """Return a datatime object of a given DICOM time. E.g., Acquisition
        for Acquition Date and Acquisition Time.
        """
        return dicomhelper.get_datetime(self.dcm, which_time.capitalize())
