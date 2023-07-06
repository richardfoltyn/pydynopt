"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import gzip
import logging
import os
import pickle
import re
from os.path import join
from typing import Any, Optional

__all__ = [
    'dump',
    'load',
    'get_cached_object'
]


def dump(
        path: str,
        obj: Any,
        directory: Optional[str] = None,
        compress: bool = True,
        overwrite: bool = True
):
    """
    Pickle object and dump it to a file, optionally using GZIP or LZ4
    compression.

    Parameters
    ----------
    path : str
        File name or path
    obj : object
    directory : str or None, optional
        Base directory
    compress : bool
        If true, apply gzip or lz4 compression to pickled objects.
    overwrite : bool
        If true, overwrite existing file. Otherwise, append a unique number
        before the extension to create a unique file name.
    """

    logger = logging.getLogger('IO')

    if not os.path.isabs(path):
        if directory:
            path = join(directory, path)

    path = os.path.normpath(path)

    if compress:
        has_lz4 = False
        try:
            import lz4.frame
            has_lz4 = True
        except ImportError:
            pass

        if not any(path.endswith(ext) for ext in ('.gz', '.lz4')):
            path += '.lz4' if has_lz4 else '.gz'

        if path.endswith('.gz'):
            lopen = gzip.open
        elif path.endswith('.lz4') and has_lz4:
            lopen = lz4.frame.open
        else:
            raise RuntimeError('Unsupported compression format')
    else:
        lopen = open

    if os.path.isfile(path) and not overwrite:

        # Use non-greedy match to get multiple extensions, if present
        pattern = r'(?P<root>.*?)(?P<ext>\.[^.]+)(?P<compress>\.[^.]+)?$'
        m = re.match(pattern, path)

        root = m.group('root')
        ext = m.group('ext')
        ext_compress = m.group('compress')
        if ext_compress:
            ext += ext_compress

        i = 0
        while True:
            fn_try = '{:s}_{:03d}{:s}'.format(root, i, ext)
            if not os.path.isfile(fn_try):
                path = fn_try
                break
            else:
                i += 1

    with lopen(path, 'wb') as f:
        pickle.dump(obj, f)

    msg = 'Saved to {:s}'.format(path)
    logger.info(msg)


def load(path: str, directory: Optional[str] = None):
    """
    Load pickled object from a given file, optionally decompressing it if
    required.

    Parameters
    ----------
    path : str
    directory : str or None

    Returns
    -------
    obj :
        Unpickled object
    """

    logger = logging.getLogger('IO')

    if not os.path.isfile(path):
        if directory:
            path = join(directory, path)

    path = os.path.normpath(path)

    logger.info('Loading from {:s}'.format(path))

    if path.endswith('.gz'):
        lopen = gzip.open
    elif path.endswith('.lz4'):
        try:
            import lz4.frame
            lopen = lz4.frame.open
        except ImportError:
            raise IOError('LZ4 library not installed')
    else:
        lopen = open

    with lopen(path, 'rb') as f:
        obj = pickle.load(f)

    return obj


def get_cached_object(
        fcn: callable,
        *args,
        cache_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
):
    """
    Load object from cache file, if present. Otherwise, call given function
    to compute object and store it in given cache file.

    Parameters
    ----------
    fcn : callable
        Function used to compute object if cache file is not found.
    args
         Positional arguments passed to `fcn`
    cache_file : str, optional
        Cache file name or path.
    cache_dir : str, optional
        Cache directory
    kwargs
        Keyword arguments passed to `fcn`

    Returns
    -------

    """

    path = None
    if cache_file is not None:
        if cache_dir is not None:
            path = os.path.join(cache_dir, cache_file)
        else:
            path = cache_file

    if path:
        extensions = ('', '.lz4', '.gz')
        for ext in extensions:
            p = f'{path}{ext}'
            if os.path.isfile(p):
                obj = load(p)
                return obj

    # Cached result does not exist, compute it
    logging.info(f'Cached result not found, calling {fcn}')

    obj = fcn(*args, **kwargs)

    if path:
        dump(path, obj, compress=True, overwrite=True)

    return obj
