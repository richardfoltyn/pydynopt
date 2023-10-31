"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""
from logging import FileHandler

import logging
import sys
import datetime
import os.path
from typing import Optional


def configure_logging():
    """
    Configure logging framework with default console handler.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    # Set default log level to INFO, otherwise we'll be flooded by MPL, Numba,
    # etc. log messages
    ch.setLevel(logging.INFO)
    fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    # Format used the (asctime) field
    datefmt = '%H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    # Turn of DEBUG messages for Numba
    logger = logging.getLogger('numba')
    logger.setLevel(logging.INFO)

    logger = logging.getLogger('numexpr.utils')
    logger.setLevel(logging.WARNING)

    # Disable JAX compilation debug info
    logger = logging.getLogger("jax")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("jaxlib")
    logger.setLevel(logging.WARNING)


def add_logfile(
        file: str,
        *,
        logdir: Optional[str] = None,
        file_timestamp: bool = False,
        date: bool = True,
        time: bool = True,
        append: bool = False
) -> FileHandler:
    """
    Add file handler to current logger.

    Parameters
    ----------
    file : str
        Log file name or path
    logdir : str, optional
        Log directory
    file_timestamp : bool
        If true, append time stamp to log file
    date : bool
        Add date to log output.
    time : bool
        Add time stamp to log output.
    append : bool
        If true, append to existing log file
    """

    timestamp = datetime.datetime.now()

    if file_timestamp:
        suffix = timestamp.strftime('%Y%m%d-%Hh%Mm')
        root, ext = os.path.splitext(file)
        if not ext:
            ext = '.log'
        file = f'{root}-{suffix}{ext}'

    if logdir:
        file = os.path.join(logdir, file)

    logger = logging.getLogger()

    mode = 'a' if append else 'w'
    fh = logging.FileHandler(file, mode=mode)
    fh.setLevel(logging.DEBUG)
    if date or time:
        fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    else:
        fmt = '%(name)s %(levelname)s: %(message)s'

    # Format used the (asctime) field
    tokens = []
    if date:
        tokens.append('%Y-%m-%d')
    if time:
        tokens.append('%H:%M:%S')

    datefmt = ' '.join(tokens)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.info(f'Log started on {timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Logging to {file}')

    return fh
