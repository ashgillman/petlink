import os
import shutil
import logging
import inspect
import numpy as np
from scipy import signal
import click

from petlink import listmode
from petlink.listmode import tracking
from petlink.helpers import dicomhelper, progress

# pydicom changes the import name
try:
    import pydicom
except ImportError:
    import dicom as pydicom


CM2MM = 10
S2MS = 1000


@click.group()
@click.version_option()
def cli():
    """Tools for PET data."""


@cli.group()
def track():
    """Track motion from list mode data."""


@track.command('com')
@click.argument('listmode', type=click.Path(exists=True))
@click.argument('time_res', type=int)
@click.option(
    '--output', '-o', type=click.Path(), default='com.csv',
    help='Output file (.csv).')
@click.option(
    '--lpf-hz', '-l', type=int, default=0,
    help='Cutoff frequecy in hz for low pass filter. (Default, no filter)')
def extract_com(listmode, time_res, lpf_hz=0, output='com.csv'):
    """Extract a centre-of-mass signal from PET data, LISTMODE, at a given time
    resolution, TIME_RES (given in ms).
    """
    return extract_surrogate_from_pet(
        listmode, tracking.com_surrogate, time_res, lpf_hz=lpf_hz,
        out_csv=output)


@track.command('pca')
@click.argument('listmode', type=click.Path(exists=True))
@click.argument('time_res', type=int)
@click.option(
    '--output', '-o', type=click.Path(), default='pca.csv',
    help='Output file (.csv).')
@click.option(
    '--lpf-hz', '-l', type=int, default=0,
    help='Cutoff frequecy in hz for low pass filter. (Default, no filter)')
def extract_com(listmode, time_res, lpf_hz=0, output='com.csv'):
    """Extract a centre-of-mass signal from PET data, LISTMODE, at a given time
    resolution, TIME_RES (given in ms).
    """
    return extract_surrogate_from_pet(
        listmode, tracking.pca_surrogate, time_res, lpf_hz=lpf_hz,
        out_csv=output)


def extract_surrogate_from_pet(surrogate_acq, extract_function, time_res,
                               lpf_hz=0, extract_function_args={},
                               out_csv='surr.csv'):
    """Extract a surrogate derived from the center-of-mass in z, y, and z for
    the PET counts.

    Args:
        surrogate_acq: PET list mode file (.ptd) acquisition.
        extract_funtion: Function to extract a surrogate.
        time_res: Time resolution (in ms) of surrogate.
        lpf_hz: (optional) Low pass filter frequency (in Hz), 0 to disable.
        extract_function_args: (optional) Default args to pass to
            extract_function.
        out_csv: (optional) Output surrogate file.
        progress: (optional) Whether to show a progress bar.
    Returns:
        A file name where the surrogate has been saved to.
    """
    logger = logging.getLogger(__name__)

    out_csv = os.path.abspath(out_csv)

    # Load the input files,
    lm = listmode.ListMode.from_file(surrogate_acq)

    # Actually pull out the sinograms,
    bar = progress.ProgressBar(max_value=lm.data.size)
    available_args = dict(
        lm=lm.data, shape=lm.unlist_shape, n_axials=lm.ifl['segment table'][0],
        segments_def=np.array(lm.ifl['segment table'], dtype=np.uint16),
        time_resolution=time_res,
        axial_spacing=lm.ifl['distance between rings'] * CM2MM,
        ring_rad=lm.ifl['gantry crystal radius'] * CM2MM,
        half_life=lm.ifl['isotope gamma halflife'], bar=bar, save_model=True)
    accepted_args = inspect.signature(extract_function).parameters.keys()
    args = {k: v for k, v in available_args.items() if k in accepted_args}
    args.update(extract_function_args)
    surrogate = extract_function(**args)
    bar.finish()
    logger.debug('Extracted %s surrogate', surrogate.shape)

    # Calculate time points
    start = dicomhelper.dicom_time_to_daily_ms(lm.dcm.AcquisitionTime)
    time = start + np.arange(surrogate.shape[-1]) * time_res

    # write the data
    tracking.save_tracking_csv(time, surrogate, out_csv)

    if lpf_hz:
        logger.info('Filtering surrogate with Butter LPF Fc=%s Hz', lpf_hz)

        def butter_lowpass(cutoff, fs, order=5):
            # http://stackoverflow.com/a/25192640/3903368
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low',
                                 analog=False)
            return b, a

        # set up filter
        fs = S2MS / time_res
        b, a = butter_lowpass(lpf_hz, fs)
        logger.debug('Fs=%s Hz, a=%s, b=%s', fs, a, b)

        # apply filter
        surrogate = signal.filtfilt(b, a, surrogate)

        # update output and save
        out_csv = out_csv.replace('.csv', '_filtered.csv')
        tracking.save_tracking_csv(time, surrogate, out_csv)

    return out_csv


@cli.group()
def dicom():
    """Tools for handling DICOMs."""


@dicom.command()
@click.argument('in_directory', type=click.Path(exists=True), default='.')
@click.argument('out_directory', type=click.Path(exists=False), default='.')
@click.option(
    '--pattern', '-p', type=str,
    default='{Patient ID}/{Study Description}/{Series Number}-{Series Description}/'
    '{Instance Number}{ext}',
    help='Output path format, from out_directory. Python str.format() style.')
def stage(in_directory, out_directory, pattern):
    """Reformat a DICOM dump into a usable format.

    \b
    Pattern is a full path and filename to the output from out_directory as
    root. Format variables are pydicom's key names (DICOM names without
    spaces), or
    - `filename` (input file),
    - `ext` (input filename extension, includes the dot).
    """
    logger = logging.getLogger(__name__)

    for root, _, files in os.walk(in_directory):
        for f in files:
            f = os.path.join(root, f)
            logger.info(f"Parsing {f}.")
            try:
                dcm = pydicom.read_file(f)
                fmt_dict = {i.name: i.value for i in dcm}
                fmt_dict['filename'] = f
                fmt_dict['ext'] = os.path.splitext(f)[-1]
                out_f = os.path.join(out_directory, pattern.format(**fmt_dict))
                out_d = os.path.dirname(out_f)

                os.makedirs(out_d, exist_ok=True)
                shutil.copyfile(f, out_f)

            except pydicom.errors.InvalidDicomError:
                logger.info(f"Couldn't parse {f}.")


if __name__ == '__main__':
    cli()
