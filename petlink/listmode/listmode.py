"""Present list mode functionality in a convenient object."""


import logging
import os
import dicom
import datetime
import numpy as np

from petlink.listmode import unlisting
from petlink import ptd, interfile, constants
from petlink.helpers import dicomhelper


HERE = os.path.dirname(__file__)


# TODO: inherit from InterfileCSA
class ListMode:
    """List mode functionality.

    Attributes:
        data: List mode raw data.
        dcm: DICOM header, a pydicom Dataset object.
        ifl: Interfile header, an Interfile object.
        unlist_shape: The shape of the resulting unlisted sinogram.
    """
    tags = constants.TAGS

    #
    # Read-only attributes
    #

    @property
    def unlist_shape(self):
        return self._unlist_shape

    @property
    def data(self):
        return self._data

    #
    # Calculated attributes
    #

    # @property
    # def scanner(self):
    #     return scanner.load(str(self.ifl['originating system']))

    # @property
    # def mash(self):
    #     return self.scanner['n_crystals'] // 2 // self.ifl['number of views']

    @property
    def duration(self):
        """Get the duration of a scan based on the data time tags."""
        return self.get_time_at_index(self.data.size-1)

    def __init__(self, data=None, dcm=None, ifl=None, #scanner=None,
                 unlist_shape=None):
        """Create a ListMode instance. If loading from a file, use
        ListMode.from_file().

        The absolute value of time tags in the list stream seems to be
        arbitrary. As such, all time tags are treated only relative to
        one-another. Therefore:
          - Time-based indexing treats the first time tag as t=0.
          - The user should be careful if using time tags in the data.

        Indexing may be performed pandas-style using the .tloc and .iloc
        attributes. .tloc exposes time-based indexing, in ms since scan
        commencement, and .iloc exposes integer-based indexing.

        Args:
            data: A numpy (or similar) Array with raw list data.
            dcm: (optional) DICOM header for data.
            ifl: (optional) Interfile header for data. If dcm has a CSA Data
                Info tag, this will be extracted from here.
            unlist_shape: (optional) Shape of unlisted sinogram. If ifl is
                defined it will be calculated automatically. It is not
                recommended the user calculate this manually.
        """
        logger = logging.getLogger(__name__)

        # Parse given dcm to a Dataset

        if isinstance(dcm, dicom.dataset.Dataset):
            self.dcm = dcm

        elif isinstance(dcm, str) and os.path.exists(dcm):
            self.dcm = dicom.read_file(dcm)

        elif dcm is not None:
            raise ValueError(
                "Can't parse DICOM input: %s. Does file exist?" % dcm)

        else:
            self.dcm = None

        # Save data, or check dcm for data

        if isinstance(data, str) and os.path.exists(data):
            self._data = np.memmap(data, mode='r',
                                   dtype=constants.PL_DTYPE)
        elif data is not None:
            self._data = data

        elif self.dcm and constants.DCM_CSA_DATA in self.dcm:
            self._data = np.fromstring(self.dcm[constants.DCM_CSA_DATA].value,
                                       dtype=constants.PL_DTYPE)

        else:
            raise ValueError('No value given or parsable for ListMode data.')

        # Parse or extract ifl as an Interfile

        if isinstance(ifl, interfile.Interfile):
            self.ifl = ifl

        elif isinstance(ifl, str) and os.path.exists(ifl):
            raise NotImplementedError('Interfile file input is TODO.')

        elif ifl is not None:
            raise ValueError("Can't parse Interfile input.")

        elif self.dcm and constants.DCM_CSA_DATA_INFO in self.dcm:
            logger.debug('Extracting interfile header from DICOM')
            ifl_source = dicomhelper.decode_ob_header(
                self.dcm[constants.DCM_CSA_DATA_INFO].value)
            self.ifl = interfile.Interfile(source=ifl_source)

        else:
            self.ifl = None

        # Extract unlisting shape

        if isinstance(unlist_shape, tuple):
            self._unlist_shape = unlist_shape

        elif unlist_shape is not None:
            raise ValueError("Can't interpret unlist_shape input.")

        elif self.ifl:
            self._unlist_shape = self._get_unlist_shape_from_interfile(
                self.ifl)
            logger.debug('Calculated unlisting shape as %s',
                         str(self.unlist_shape))

        # elif scanner:
        #     raise NotImplementedError(
        #         'Calculating unlist shape from scanner is TODO.')

        else:
            logger.warning('ListMode object instantiated but unlist shape '
                           "couldn't be determined")
            self._unlist_shape = None

    #
    # I/O
    #

    @staticmethod
    def from_ptd(ptd_file):
        """Load a ListMode object from a .ptd file."""
        return ListMode(data=ptd.read_data(ptd_file),
                        dcm=ptd.read_dcm(ptd_file))

    @staticmethod
    def from_file(filename, force_type=None):
        """Load a ListMode object from a .ptd file. Optionally force filetype
        reading with force_type to 'ptd' or 'dcm'.
        """
        if force_type == 'ptd' or filename.lower().endswith('.ptd'):
            return ListMode.from_ptd(filename)
        if force_type == 'dcm' or (filename.lower().endswith('.dcm')
                                   or filename.lower().endswith('.ima')):
            bf_file = filename[:-4] + '.bf'
            if os.path.exists(bf_file):
                return ListMode(dcm=filename, data=bf_file)
            else:
                return ListMode(dcm=filename)
        else:
            raise RuntimeError('Unkown filetype for {}.'.format(
                force_type or filename))

    def to_ptd(self, filename):
        ptd.write_ptd(self.data, self.dcm, filename)

    #
    # Functionality
    #

    def unlist(self, template=None, keep_tof=False):
        """Unlist/histogram list mode events into a prompt and delay sinogram.

        If a template for an interfile header is passed (path, string or
        Interfile), two Interfile objects are returned. If template is None
        (default), automatically find a template. If template is False, two
        numpy arrays are returned.
        """
        psino, dsino = unlisting.unlist(
            self.data, self.unlist_shape, tof=keep_tof)

        if template is False:
            return psino, dsino
        elif template is None:
            orig_sys = str(self.ifl['originating system'])
            template = os.path.join(HERE, 'templates', orig_sys + '_span1.hs')

        if isinstance(template, interfile.Interfile):
            pass
        elif isinstance(template, str) and os.path.exists(template):
            template = interfile.Interfile(sourcefile=template)
        elif isinstance(template, str):
            template = interfile.Interfile(source=template)
        else:
            raise TypeError('Unexpected type for template')

        psino = interfile.Interfile(source=str(template), data=psino)
        dsino = interfile.Interfile(source=str(template), data=dsino)
        return psino, dsino

    def get_time_at_index(self, index):
        """Get the time in ms since the start of the scan at a given list
        data index.

        The first time tag is assigned t=0.
        """
        if index == -1:
            duration_raw = unlisting.duration(self.data)
        else:
            duration_raw = unlisting.duration(self.data[:index+1])
        if duration_raw == 0:
            # special case, no previous time tags
            return 0
        return (duration_raw - unlisting.begin(self.data))

    def get_index_at_time(self, time):
        """Get the data index for a time tag in ms since the start of the
        scan.
        """
        if time < 0:
            time = self.duration + time
        assert 0 <= time <= self.duration, 'Chosen time out of range.'
        return unlisting.find_time_index(
            self.data, unlisting.begin(self.data) + time)

    #
    # Indexing
    #

    def _slice(self, start_idx, end_idx):
        """Slice list data, adjusting header information accordingly."""
        # make the new data.
        new_ifl = self.ifl
        new_dcm = self.dcm
        new_data = self.data[start_idx:end_idx]

        # calculate new meta data
        start_delta = datetime.timedelta(
            milliseconds=self.get_time_at_index(start_idx))
        end_delta = datetime.timedelta(
            milliseconds=self.get_time_at_index(end_idx))

        assert start_delta < end_delta, 'Issue extracting times.'

        if self.ifl:
            # Update Interfile start time
            original_ifl_study_time = self.ifl.get_datetime('study')
            new_ifl_study_time = original_ifl_study_time + start_delta
            new_ifl.set_datetime('study', new_ifl_study_time)

            # Update Interfile duration
            new_duration = end_delta - start_delta
            assert self.ifl.header['image duration'].units == 'sec'
            new_ifl['image duration'] = new_duration.seconds

        if self.dcm:
            # Update DICOM start time
            original_dcm_acq_time = dicomhelper.get_datetime(
                self.dcm, 'Acquisition')
            new_dcm_acq_time = original_dcm_acq_time + start_delta
            dicomhelper.set_datetime(new_dcm, 'Acquisition',
                                      new_dcm_acq_time)

            if self.ifl:
                # stuff new interfile header into DICOM
                new_dcm[constants.DCM_CSA_DATA_INFO].value = \
                    dicomhelper.encode_ob_header(str(new_ifl))

        return ListMode(data=new_data, dcm=new_dcm, ifl=new_ifl,
                        unlist_shape=self.unlist_shape)

    @property
    def tloc(self):
        """Indexing based on time (ms since beginning).

        e.g.,
            one_sec = listmode.tloc[5000:6000]
        Here, one_sec is a new ListMode object containing the sixth second
        of acquisition. The acqusition time in the DICOM and Interfiles
        headers of one_sec are five seconds later than the original
        listmode, and the durations are one second.
        """
        return _TimeIndexer(self)

    #
    # Internal calculations
    #

    @staticmethod
    def _get_unlist_shape_from_interfile(ifl):
        """Determine the unlisted sinogram shape from Siemens
        interfile data.
        """
        n_tof = int(ifl['number of tof time bins'])
        if n_tof > 1:
            n_tof += 1 # PET/CTs seem to pop delays in last bin

        unlist_shape = (int(ifl['number of projections']),
                        int(ifl['number of views']),
                        int(np.sum(ifl['segment table'])),
                        n_tof)
        return unlist_shape


class _TimeIndexer:
    """Implements ListMode.tloc[...]."""
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, slice_):
        assert isinstance(slice_, slice), \
            'ListMode.tloc only supports slicing.'
        assert slice_.step is None, 'Step slicing of ListMode not supported.'

        # get start and stop in indices rather than times
        start_idx = self.owner.get_index_at_time(slice_.start or 0)
        if slice_.stop is None:
            stop_idx = self.owner.get_index_at_time(self.owner.duration)
        else:
            # add 1 so that we keep the final time tag
            stop_idx = self.owner.get_index_at_time(slice_.stop) + 1

        return self.owner._slice(start_idx, stop_idx)
