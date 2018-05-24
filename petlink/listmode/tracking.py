from .com_tracking import com_surrogate
from .sens_tracking import sens_surrogate
from .pca_tracking import pca_surrogate

import numpy as np


def load_tracking_csv(csv_file, is_gates=False):
    """Load CSV file with time and tracking signal. If is_gates, coerces gates
    to int type.
    """
    time, *tracking = np.loadtxt(csv_file, delimiter=',').T
    tracking = np.vstack(tracking).T.squeeze()
    time = time.astype(np.int)
    if is_gates:
        tracking = tracking.astype(np.int)
    return time, tracking


def save_tracking_csv(time, tracking, csv_file, is_gates=False):
    """Save CSV file with time and tracking signal."""
    extra_args = {}
    if is_gates:
        extra_args['fmt'] = '%d'
    np.savetxt(csv_file, np.vstack((time, tracking)).T, delimiter=',',
               **extra_args)
