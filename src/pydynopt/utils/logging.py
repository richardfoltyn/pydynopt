"""
Logging utilities for the pydynopt package.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from contextlib import suppress
import datetime
from importlib.metadata import PackageNotFoundError, version
import logging
from logging import FileHandler, Logger
import os
from pathlib import Path
import platform
import sys
from typing import Any

old_factory = logging.getLogRecordFactory()


def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """
    Record factory to attach relative time attributes.

    Parameters
    ----------
    *args
        Positional arguments.
    **kwargs
        Keyword arguments.

    Returns
    -------
    The created log record with custom relative time attributes.
    """
    record = old_factory(*args, **kwargs)

    # Created timestamp in seconds
    created = record.relativeCreated / 1000.0

    rdays = int(created / 60 / 60 / 24)
    rem = created % (60 * 60 * 24)
    rhours = int(rem / 60 / 60)
    rem = rem % (60 * 60)
    rminutes = int(rem / 60)
    rseconds = rem % 60

    record.rday = rdays  # type: ignore[attr-defined]
    record.rhrs = rhours  # type: ignore[attr-defined]
    record.rmin = rminutes  # type: ignore[attr-defined]
    record.rsec = rseconds  # type: ignore[attr-defined]

    return record


def configure_logging(reltime: bool = True, stdout: bool = True) -> None:
    """
    Configure a logging framework with the default console handler.

    Parameters
    ----------
    reltime
        Print time stamp as relative time since logging start.
    stdout
        Print log messages to stdout.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console
    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        # Set the default log level to INFO, otherwise we'll be flooded by MPL, Numba,
        # etc. log messages
        ch.setLevel(logging.INFO)

        if reltime:
            # Set custom RecordFactory to attach relative time attributes
            logging.setLogRecordFactory(record_factory)
            fmt = (
                '[{rday:d}d {rhrs:02d}:{rmin:02d}:{rsec:04.1f}] {name} {levelname}: '
                '{message}'
            )
            formatter = logging.Formatter(fmt=fmt, style='{')
        else:
            fmt = '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
            # Format used the (asctime) field
            datefmt = '%H:%M:%S'
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    # Turn off DEBUG messages for Numba
    logger = logging.getLogger('numba')
    logger.setLevel(logging.INFO)

    logger = logging.getLogger('matplotlib')
    logger.setLevel(logging.INFO)

    logger = logging.getLogger('numexpr.utils')
    logger.setLevel(logging.WARNING)

    logger = logging.getLogger('fontTools')
    logger.setLevel(logging.WARNING)

    # Disable JAX compilation debug info
    logger = logging.getLogger('jax')
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger('jaxlib')
    logger.setLevel(logging.WARNING)


def add_logfile(
    file: Path | str,
    *,
    logdir: Path | str | None = None,
    file_timestamp: bool = False,
    date: bool = False,
    time: bool = False,
    reltime: bool = False,
    append: bool = False,
) -> FileHandler:
    """
    Add file handler to current logger.

    Parameters
    ----------
    file
        Log file name or path.
    logdir
        Log directory.
    file_timestamp
        If true, append time stamp to log file.
    date
        Add date to log output.
    time
        Add time stamp to log output.
    reltime
        Add relative time stamp since logging start. Ignores `date` and `time`
        arguments.
    append
        If true, append to existing log file.

    Returns
    -------
    The added file handler.
    """
    timestamp = datetime.datetime.now()
    path = Path(file)

    if file_timestamp:
        suffix = timestamp.strftime('%Y%m%d-%Hh%Mm')
        root = path.stem
        ext = path.suffix or '.log'
        path = path.with_name(f'{root}-{suffix}{ext}')

    if logdir:
        path = Path(logdir) / path

    logger = logging.getLogger()

    mode = 'a' if append else 'w'
    fh = logging.FileHandler(path, mode=mode)
    fh.setLevel(logging.DEBUG)

    if date or time:
        fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
        # Format used the (asctime) field
        tokens = []
        if date:
            tokens.append('%Y-%m-%d')
        if time:
            tokens.append('%H:%M:%S')
        datefmt = ' '.join(tokens)
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    elif reltime:
        # Set custom RecordFactory to attach relative time attributes
        logging.setLogRecordFactory(record_factory)
        fmt = (
            '[{rday:d}d {rhrs:02d}:{rmin:02d}:{rsec:04.1f}] {name} {levelname}: '
            '{message}'
        )
        formatter = logging.Formatter(fmt=fmt, style='{')
    else:
        fmt = '%(name)s %(levelname)s: %(message)s'
        formatter = logging.Formatter(fmt=fmt)

    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.info(f'Log started on {timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Logging to {path}')

    info = platform.uname()
    tokens = []
    if info.node:
        tokens.append(f'host {info.node}')
    tokens.append(
        ' '.join(tok for tok in (info.system, info.release, info.version) if tok)
    )
    logger.info(f'Running on {", ".join(tok.strip() for tok in tokens)}')

    return fh


def log_cmd_args(logger: Logger | None = None, level: int = logging.DEBUG) -> None:
    """
    Log command line arguments.

    Parameters
    ----------
    logger
        Logger instance. If None, the root logger is used.
    level
        Logging level.
    """
    if logger is None:
        logger = logging.getLogger()

    logger.log(level, 'Script:')
    logger.log(level, f'  {sys.argv[0]}')

    args = ' '.join(arg.strip() for arg in sys.argv[1:])
    if args:
        logger.log(level, 'Command line arguments:')
        logger.log(level, f'  {args}')


def log_python_env(logger: Logger | None = None, level: int = logging.DEBUG) -> None:
    """
    Log version info for Python and important packages.

    Parameters
    ----------
    logger
        Logger instance. If None, the root logger is used.
    level
        Logging level.
    """
    if logger is None:
        logger = logging.getLogger()

    logger.log(level, 'Python environment:')
    logger.log(level, f'  Interpreter: {sys.executable}')

    logger.log(level, f'  Python: {platform.python_version()}')

    # List of distribution and display names for relevant packages. Note that the
    # distribution name is used for version lookup and NEED NOT be the same as
    # the import name (e.g., scikit-learn = distribution name)
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('numba', 'numba'),
        ('patsy', 'patsy'),
        ('statsmodels', 'statsmodels'),
        ('scikit-learn', 'sklearn'),
        ('jax', 'JAX'),
    ]

    for dist_name, display_name in packages:
        with suppress(PackageNotFoundError):
            logger.log(level, f'  {display_name}: {version(dist_name)}')


def log_env_vars(logger: Logger | None = None, level: int = logging.DEBUG) -> None:
    """
    Log set environment variables related to OMP, MKL, or thread/CPU control.

    Parameters
    ----------
    logger
        Logger instance. If None, the root logger is used.
    level
        Logging level.
    """
    if logger is None:
        logger = logging.getLogger()

    # Prefixes for wildcard matching (case-insensitive)
    prefixes = ('OMP_', 'MKL_', 'NUMBA_')
    # Specific other environment variables related to thread/CPU control
    specific_vars = {
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'TBB_NUM_THREADS',
        'BLAS_NUM_THREADS',
        'TF_NUM_INTEROP_THREADS',
        'TF_NUM_INTRAOP_THREADS',
    }

    env_vars = {}
    for key, value in os.environ.items():
        key_upper = key.upper()
        if any(key_upper.startswith(p) for p in prefixes) or key_upper in specific_vars:
            env_vars[key] = value

    if env_vars:
        logger.log(level, 'Environment variables:')
        for key in sorted(env_vars.keys()):
            logger.log(level, f'  {key}: {env_vars[key]}')
