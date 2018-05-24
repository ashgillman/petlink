"""Abstracts Progress Bar functionality.

Allow ProgressBar-2 to be an optional dependency. If ProgressBar-2 is
available, use it, else provide a dummy.
"""

import logging
try:
    import progressbar
except ImportError:
    progressbar = None


class DummyBar:
    def update(*args, **kwargs): pass
    def finish(*args, **kwargs): pass
    def __enter__(self, *args, **kwargs): return self
    def __exit__(self, *args, **kwargs): pass


def ProgressBar(*args, **kwargs):
    logger = logging.getLogger(__name__)

    if progressbar:
        return progressbar.ProgressBar(*args, **kwargs)
    else:
        logger.warning('`progressbar` could not be imported.')
        return DummyBar()
