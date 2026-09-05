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
import struct
import tempfile
from typing import Any
import zlib

__all__ = ['CorruptFileError', 'dump', 'get_cached_object', 'get_hash_value', 'load']


class CorruptFileError(Exception):
    """A pickle file could not be decoded."""


def _corrupt_file_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [
        pickle.UnpicklingError,
        EOFError,
        struct.error,
        gzip.BadGzipFile,
        zlib.error,
    ]

    import lzma

    errors.append(lzma.LZMAError)
    try:
        import lz4.frame

        errors.append(lz4.frame.LZ4FrameError)
    except (ImportError, AttributeError):
        pass
    try:
        import pyzstd

        errors.append(pyzstd.ZstdError)
    except (ImportError, AttributeError):
        pass
    return tuple(errors)


def dump(
    path: Path | str,
    obj: Any,
    directory: Path | str | None = None,
    compress: bool = True,
    overwrite: bool = True,
    nthreads: int | None = -1,
    atomic: bool = True,
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
    atomic
        If true, write to a temporary file and atomically replace the destination.
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
        valid_suffixes = {'.gz', '.lz4', '.xz', '.zstd', '.zst'}
        if path.suffix.lower() not in valid_suffixes:
            path = path.with_name(path.name + '.zst')

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
                raise ValueError(
                    'lz4 package is not installed. '
                    'Install pydynopt[compression] to enable this compression format.'
                ) from None
        elif suffix in ('.zstd', '.zst'):
            try:
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
                        'Cannot use zstd compression, neither zstd nor pyzstd library is installed. '
                        'Install pydynopt[compression] to enable this compression format.'
                    ) from None
        else:
            raise RuntimeError('Unsupported compression format')
    else:
        lopen = open

    if path.is_file() and not overwrite:
        suffixes = path.suffixes
        if suffixes:
            ext = ''.join(suffixes[-2:])
            root = path.name.removesuffix(ext)
            i = 0
            while True:
                fn_try = path.with_name(f'{root}_{i:03d}{ext}')
                if not fn_try.is_file():
                    path = fn_try
                    break
                i += 1

    kw.update(kwargs)

    tmp_path: Path | None = None
    try:
        if atomic:
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            write_path = tmp_path
        else:
            write_path = path

        with lopen(write_path, 'wb', **kw) as f:  # ty: ignore[no-matching-overload]
            pickle.dump(obj, f)

        if tmp_path is not None:
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    logger.info(f'Saved to {path}')
    return path


def load(path: Path | str, directory: Path | str | None = None, **kwargs: Any) -> Any:
    """Load a pickled object from a given file, optionally decompressing it."""
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
            raise ValueError(
                f'LZ4 library not installed, cannot load {path}. '
                'Install pydynopt[compression] to enable this compression format.'
            ) from None
    elif suffix in ('.xz', '.lzma'):
        import lzma

        lopen = lzma.open
    elif suffix in ('.zstd', '.zst'):
        try:
            from compression import zstd  # ty: ignore[unresolved-import]

            lopen = zstd.open
        except ImportError:
            try:
                import pyzstd

                lopen = pyzstd.open
            except ImportError:
                raise ImportError(
                    'neither zstd nor pyzstd library is installed. '
                    'Install pydynopt[compression] to enable this compression format.'
                ) from None
    else:
        lopen = open

    kw.update(kwargs)
    try:
        with lopen(path, 'rb', **kw) as f:
            obj = pickle.load(f)
    except _corrupt_file_errors() as err:
        raise CorruptFileError(f'Could not load pickle file: {path}') from err

    return obj


def get_cached_object(
    fcn: Callable[..., Any],
    *args: Any,
    cache_file: Path | str | None = None,
    cache_dir: Path | str | None = None,
    compress: bool = True,
    **kwargs: Any,
) -> Any:
    """Load an object from cache or compute and persist it."""
    path = None
    if cache_file is not None:
        path = Path(cache_dir) / cache_file if cache_dir is not None else Path(cache_file)

    if path:
        extensions = ('', '.xz', '.lz4', '.gz', '.zstd', '.zst')
        for ext in extensions:
            candidate = path.with_name(path.name + ext) if ext else path
            if candidate.is_file():
                try:
                    return load(candidate)
                except CorruptFileError:
                    logging.warning(
                        'Cache file %s is corrupt; ignoring it and recomputing',
                        candidate,
                    )

    fcn_name = getattr(fcn, '__name__', 'callable')
    logging.info(f'Cached result not found, calling {fcn_name}()')
    obj = fcn(*args, **kwargs)
    if path:
        dump(path, obj, compress=compress, overwrite=True)
    return obj


def get_hash_value(*args: Any, **kwargs: Any) -> str:
    """Convert sequence of objects to a hash value."""
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

    return hashlib.sha256('_'.join(hashes).encode()).hexdigest()
