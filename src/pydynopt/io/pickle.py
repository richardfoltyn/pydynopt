"""
Utility functions for pickling and unpickling objects with compression.

This work is licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable
import gzip
import logging
import os
from pathlib import Path
import pickle
from typing import Any

__all__ = ['dump', 'get_cached_object', 'get_hash_value', 'load']


def dump(
    path: Path | str,
    obj: Any,
    directory: Path | str | None = None,
    compress: bool = True,
    overwrite: bool = True,
    nthreads: int | None = -1,
    **kwargs: Any,
) -> Path:
    """
    Pickle an object and dump it to a file.

    Optionally use GZIP, LZ4, XZ, or Zstandard compression.

    Parameters
    ----------
    path
        File name or path.
    obj
        Object to pickle.
    directory
        Base directory.
    compress
        If true, compress the pickled object. If the path has no recognized
        compression suffix, append ``.zstd``.
    overwrite
        If true, overwrite an existing file. Otherwise, append a unique number
        before the extension to create a unique file name.
    nthreads
        Number of threads to use for decompression (if applicable). A value of -1 uses
        all available logical cores.
    kwargs
        Keyword arguments passed to the respective ``open()`` function of the
        chosen compression library.

    Returns
    -------
    Path actually written. This may differ from the requested path because of
    the ``directory`` argument, compression suffix selection, or collision
    numbering when ``overwrite`` is false.
    """
    logger = logging.getLogger('IO')

    path = Path(path)
    if not path.is_absolute() and directory:
        path = Path(directory) / path

    if nthreads is None:
        nthreads = int((os.cpu_count() or 2) / 2)
    elif nthreads == -1:
        nthreads = os.cpu_count() or 1

    kw = {}

    if compress:
        valid_suffixes = {'.gz', '.lz4', '.xz', '.zstd'}
        if path.suffix.lower() not in valid_suffixes:
            path = path.with_name(path.name + '.zstd')

        suffix = path.suffix.lower()
        if suffix == '.gz':
            lopen = gzip.open
        elif suffix == '.xz':
            import lzma

            lopen = lzma.open
        elif suffix == '.lz4':
            try:
                import lz4.frame

                lopen = lz4.frame.open
            except ImportError:
                raise ValueError('lz4 package is not installed') from None
        elif suffix == '.zstd':
            try:
                # Built-in zstd support added in Python 3.14
                from compression import zstd  # ty: ignore[unresolved-import]

                lopen = zstd.open
                kw = {
                    'options': {
                        zstd.CompressionParameter.nb_workers: nthreads,
                        zstd.CompressionParameter.compression_level: 19,
                    }
                }
            except ImportError:
                try:
                    import pyzstd
                    from pyzstd import CParameter

                    lopen = pyzstd.open
                    kw = {
                        'level_or_option': {
                            CParameter.nbWorkers: nthreads,
                            CParameter.compressionLevel: 19,
                        }
                    }
                except ImportError:
                    raise ValueError(
                        'Cannot use zstd compression, neither zstd nor pyzstd library is installed'
                    ) from None
        else:
            raise RuntimeError('Unsupported compression format')
    else:
        lopen = open

    if path.is_file() and not overwrite:
        suffixes = path.suffixes
        if suffixes:
            # Keep up to two extensions (e.g., .pkl.gz)
            ext_parts = suffixes[-2:]
            ext = ''.join(ext_parts)
            root = path.name.removesuffix(ext)

            i = 0
            while True:
                fn_try = path.with_name(f'{root}_{i:03d}{ext}')
                if not fn_try.is_file():
                    path = fn_try
                    break
                else:
                    i += 1

    kw.update(kwargs)

    with lopen(path, 'wb', **kw) as f:  # ty: ignore[no-matching-overload]
        pickle.dump(obj, f)

    msg = f'Saved to {path}'
    logger.info(msg)

    return path


def load(path: Path | str, directory: Path | str | None = None, **kwargs: Any) -> Any:
    """
    Load a pickled object from a given file.

    Optionally decompress the file if required.

    Parameters
    ----------
    path
        File name or path.
    directory
        Base directory.
    kwargs
        Keyword arguments passed to respective open() function of the chosen
        compression library.

    Returns
    -------
    Unpickled object.
    """
    logger = logging.getLogger('IO')

    if not path:
        raise ValueError(f"Invalid path '{path}'")

    path = Path(path)
    if not path.is_file() and directory:
        path = Path(directory) / path

    logger.info(f'Loading from {path}')

    kw = {}

    suffix = path.suffix.lower()
    if suffix in ('.gz', '.gzip'):
        lopen = gzip.open
    elif suffix == '.lz4':
        try:
            import lz4.frame

            lopen = lz4.frame.open
        except ImportError:
            raise ValueError(f'LZ4 library not installed, cannot load {path}') from None
    elif suffix in ('.xz', '.lzma'):
        import lzma

        lopen = lzma.open
    elif suffix == '.zstd':
        try:
            # Built-in zstd support added in Python 3.14
            from compression import zstd  # ty: ignore[unresolved-import]

            lopen = zstd.open
        except ImportError:
            try:
                import pyzstd

                lopen = pyzstd.open
            except ImportError:
                raise ImportError(
                    'neither zstd nor pyzstd library is installed'
                ) from None
    else:
        lopen = open

    kw.update(kwargs)

    with lopen(path, 'rb', **kw) as f:
        obj = pickle.load(f)

    return obj


def get_cached_object(
    fcn: Callable[..., Any],
    *args: Any,
    cache_file: Path | str | None = None,
    cache_dir: Path | str | None = None,
    compress: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Load object from cache file, if present.

    Otherwise, call given function to compute object and store it in given
    cache file.

    Parameters
    ----------
    fcn
        Function used to compute object if cache file is not found.
    args
        Positional arguments passed to `fcn`.
    cache_file
        Cache file name or path.
    cache_dir
        Cache directory.
    compress
        Use compression when storing the cache file.
    kwargs
        Keyword arguments passed to `fcn`.

    Returns
    -------
    Computed or cached object.
    """
    path = None
    if cache_file is not None:
        if cache_dir is not None:
            path = Path(cache_dir) / cache_file
        else:
            path = Path(cache_file)

    if path:
        extensions = ('', '.xz', '.lz4', '.gz', '.zstd')
        for ext in extensions:
            p = path.with_name(path.name + ext) if ext else path
            if p.is_file():
                obj = load(p)
                return obj

    # Cached result does not exist, compute it
    fcn_name = getattr(fcn, '__name__', 'callable')
    logging.info(f'Cached result not found, calling {fcn_name}()')

    obj = fcn(*args, **kwargs)

    if path:
        dump(path, obj, compress=compress, overwrite=True)

    return obj


def get_hash_value(*args: Any, **kwargs: Any) -> str:
    """
    Convert sequence of objects to a hash value.

    Can be used as a filename component.

    Parameters
    ----------
    args
        Positional arguments to hash.
    kwargs
        Keyword arguments to hash.

    Returns
    -------
    Hash value.
    """
    import hashlib

    hashes = []
    for obj in args:
        try:
            h = hashlib.sha256(obj).hexdigest()
        except TypeError:
            h = hashlib.sha3_256(f'{obj}'.encode()).hexdigest()
        hashes.append(h)

    for key, value in kwargs.items():
        for obj in (key, value):
            try:
                h = hashlib.sha256(obj).hexdigest()  # ty: ignore[invalid-argument-type]
            except TypeError:
                h = hashlib.sha3_256(f'{obj}'.encode()).hexdigest()
            hashes.append(h)

    s = '_'.join(hashes)
    h = hashlib.sha256(s.encode())

    return h.hexdigest()
