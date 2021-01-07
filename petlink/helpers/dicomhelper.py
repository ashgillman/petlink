"""Routines for helping with the pydicom module."""

import logging
import time
import datetime


# If loading many DICOMs, we can tend to "spew"
class DuplicateFilter(object):
    """A class with a filter method that remembers if its seen a value before."""
    def __init__(self, max_elem=1000):
        self.msgs = set()
        self.max_elem = max_elem

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)

        # Prevent a memory leak
        # dodgy, but just forget a random one...
        # This ought to be a FIFO queue...
        if len(self.msgs) > self.max_elem:
            self.msgs.pop()  
        return rv


dup_filter = DuplicateFilter()


DCM_DATE_FMT = '%Y%m%d'
DCM_TIME_FMT = '%H%M%S.%f'


def get_local_timezone():
    return datetime.timezone(datetime.timedelta(seconds=-time.timezone))


def get_datetime(dcm, type_):
    """Get a `type` of date/time (e.g. Acquisition for Acqusition Date and
    Acquisition Time) as a datetime.datetime object.
    """
    # unsure first letter is capital, e.g. acquisition->Acquisition
    type_ = type_.capitalize()

    # Get the strings
    dcm_date_str = getattr(dcm, type_ + 'Date')
    dcm_time_str = getattr(dcm, type_ + 'Time')

    return date_time_str_to_datetime(dcm_date_str, dcm_time_str, dcm)


def date_time_str_to_datetime(dcm_date_str, dcm_time_str, dcm=None):
    """Get a Python datatime object."""
    logger = logging.getLogger(__name__)
    logger.addFilter(dup_filter)

    # Get time zone, {+,-}HHMM
    # NB: I haven't actually ever seen a TimezoneOffsetFromUTC tag...
    if dcm is not None:
        tz = getattr(dcm, 'TimezoneOffsetFromUTC', None)
    else:
        tz = None

    # convert time zone to timezone object
    # if omitted assume local time zone
    if tz is not None:
        tz = datetime.timezone(datetime.timedelta(hours=int(tz[:3]),
                                                  minutes=int(tz[3:5])))
    else:
        tz = get_local_timezone()
        logger.warn(
            'No time zone in DICOM. Will be incorrect if acquired in another '
            'time zone to your PC')

    dt = datetime.datetime.strptime(dcm_date_str + dcm_time_str,
                                    DCM_DATE_FMT + DCM_TIME_FMT)
    # Now add in the timezone...
    dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                           dt.second, dt.microsecond, tzinfo=tz)
    return dt


def set_datetime(dcm, type_, new_datetime):
    """Set a `type` of date/time (e.g. Acquisition for Acqusition Date and
    Acquisition Time) from a datetime.datetime object.

    Assumes dcm's timezone is already defined., or if omitted uses local.
    """
    # Get time zone, {+,-}HHMM
    # NB: I haven't actually seen one of these tags...
    # if omitted assume local time zone
    tz = getattr(dcm, 'TimezoneOffsetFromUTC', None)
    if tz:
        tz = datetime.timezone(datetime.timedelta(hours=int(tz[:3]),
                                                  minutes=int(tz[3:5])))
    else:
        tz = get_local_timezone()

    # ensure that we have a time zone defined for the new datetime
    # if omitted assume local time zone
    if not new_datetime.tzinfo:
        new_datetime.tzinfo = get_local_timezone()

    setattr(dcm, type_ + 'Date',
            new_datetime.astimezone(tz).strftime(DCM_DATE_FMT))
    setattr(dcm, type_ + 'Time',
            new_datetime.astimezone(tz).strftime(DCM_TIME_FMT))


def decode_ob_header(bytes_):
    """Decode long string stored as bytes to a string.

    Use this method instead of manually decoding to avoid forgetting the null
    terminator.
    """
    return bytes_.strip(b'\x00').decode('ascii')


def encode_ob_header(string):
    """Encode long string to bestored as bytes.

    Use this method instead of manually encoding to avoid forgetting the null
    terminator.
    """
    return string.encode('ascii') + b'\x00'

def dicom_time_to_daily_ms(dcm_time_str):
    """'hhmmss.xxxxxx' to no. ms since midnight."""
    dcm_datetime = datetime.datetime.strptime(dcm_time_str, DCM_TIME_FMT)
    midnight = dcm_datetime.replace(hour=0, minute=0, second=1, microsecond=0)
    delta = dcm_datetime - midnight
    return delta.total_seconds() * 1000 + delta.microseconds / 1000
