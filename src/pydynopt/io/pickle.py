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
import struct
import tempfile
import zlib
from os.path import join
from typing import Any, Optional

__all__ = [
    "CorruptFileError",
    "dump",
    "load",
    "get_cached_object",
    "get_hash_value",
]


class CorruptFileError(Exception):
    """A pickle file could not be decoded."""


def _corrupt_file_errors():
    errors = [
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
    path: str,
    obj: Any,
    directory: Optional[str] = None,
    compress: bool = True,
    overwrite: bool = True,
    nthreads: Optional[int] = -1,
    atomic: bool = True,
    **kwargs,
):
    """
    Pickle an object and dump it to a file, optionally using GZIP or LZ4
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
        If true, overwrite an existing file. Otherwise, append a unique number
        before the extension to create a unique file name.
    atomic : bool
        If true, write to a temporary file and atomically replace the destination.
    nthreads : int or None, optional
        Number of threads to use for decompression (if applicable). A value of -1 uses
        all available logical cores.
    kwargs :
        Keyword arguments passed to respective open() function of the chosen
        compression library.
    """

    logger = logging.getLogger("IO")

    if not os.path.isabs(path):
        if directory:
            path = join(directory, path)

    path = os.path.normpath(path)

    if nthreads is None:
        nthreads = int(os.cpu_count() / 2)
    elif nthreads == -1:
        nthreads = os.cpu_count()

    kw = {}

    if compress:
        has_lz4 = False
        try:
            import lz4.frame

            has_lz4 = True
        except ImportError:
            pass

        if not re.match(".*((gz)|(lz4)|(xz)|(zst)|(zstd))$", path, re.IGNORECASE):
            path += ".xz"

        if re.match(r".*\.gz$", path, re.IGNORECASE):
            lopen = gzip.open
        elif re.match(r".*\.xz$", path, re.IGNORECASE):
            import lzma

            lopen = lzma.open
        elif re.match(r".*\.lz4$", path, re.IGNORECASE) and has_lz4:
            lopen = lz4.frame.open
        elif re.match(r".*\.(zst|zstd)$", path, re.IGNORECASE):
            try:
                import pyzstd
                from pyzstd import CParameter

                lopen = pyzstd.open
                kw = {
                    "level_or_option": {
                        CParameter.nbWorkers: nthreads,
                        CParameter.compressionLevel: 19,
                    }
                }
            except ImportError:
                raise ValueError(
                    "Cannot use zstd compression, pyzstd library not installed"
                )
        else:
            raise RuntimeError("Unsupported compression format")
    else:
        lopen = open

    if os.path.isfile(path) and not overwrite:
        # Use non-greedy match to get multiple extensions, if present
        pattern = r"(?P<root>.*?)(?P<ext>\.[^.]+)(?P<compress>\.[^.]+)?$"
        m = re.match(pattern, path)

        root = m.group("root")
        ext = m.group("ext")
        ext_compress = m.group("compress")
        if ext_compress:
            ext += ext_compress

        i = 0
        while True:
            fn_try = "{:s}_{:03d}{:s}".format(root, i, ext)
            if not os.path.isfile(fn_try):
                path = fn_try
                break
            else:
                i += 1

    kw.update(kwargs)

    tmp_path = None
    try:
        if atomic:
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(path) or ".",
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
            )
            os.close(fd)
            write_path = tmp_path
        else:
            write_path = path

        with lopen(write_path, "wb", **kw) as f:
            pickle.dump(obj, f)

        if tmp_path is not None:
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    msg = "Saved to {:s}".format(path)
    logger.info(msg)


def load(
    path: str, directory: Optional[str] = None, **kwargs
):
    """
    Load a pickled object from a given file, optionally decompressing it if
    required.

    Parameters
    ----------
    path : str
    directory : str or None
    kwargs : dict
        Keyword arguments passed to respective open() function of the chosen
        compression library.

    Returns
    -------
    obj :
        Unpickled object
    """

    logger = logging.getLogger("IO")

    if not path:
        raise ValueError(f'Invalid path \'{path}\'')

    if not os.path.isfile(path):
        if directory:
            path = join(directory, path)

    path = os.path.normpath(path)

    logger.info("Loading from {:s}".format(path))

    kw = {}

    if m := re.match(r".*\.(?P<ext>[^.]+)$", path):
        ext = m.group("ext").lower()

        if ext in ("gz", "gzip"):
            lopen = gzip.open
        elif ext in ("lz4",):
            try:
                import lz4.frame

                lopen = lz4.frame.open
            except ImportError:
                raise IOError("LZ4 library not installed")
        elif ext in ("xz", "lzma"):
            import lzma

            lopen = lzma.open
        elif ext in ("zst", "zstd"):
            try:
                import pyzstd
                from pyzstd import CParameter

                lopen = pyzstd.open
            except ImportError:
                raise ImportError("pyzstd library not installed")
        else:
            lopen = open
    else:
        lopen = open

    kw.update(kwargs)

    try:
        with lopen(path, "rb", **kw) as f:
            obj = pickle.load(f)
    except _corrupt_file_errors() as err:
        raise CorruptFileError(f"Could not load pickle file: {path}") from err

    return obj


def get_cached_object(
    fcn: callable,
    *args,
    cache_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    compress: bool = True,
    **kwargs,
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
    compress : bool
        Use compression when storing the cache file
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
        extensions = ("", ".xz", ".lz4", ".gz")
        for ext in extensions:
            p = f"{path}{ext}"
            if os.path.isfile(p):
                try:
                    obj = load(p)
                except CorruptFileError:
                    logging.warning(
                        "Cache file %s is corrupt; ignoring it and recomputing", p
                    )
                else:
                    return obj

    # Cached result does not exist, compute it
    logging.info(f"Cached result not found, calling {fcn.__name__}()")

    obj = fcn(*args, **kwargs)

    if path:
        dump(path, obj, compress=compress, overwrite=True)

    return obj


def get_hash_value(*args, **kwargs) -> str:
    """
    Convert sequence of objets to a hash value that can be used as a filename component.

    Parameters
    ----------
    args

    Returns
    -------
    str
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
                h = hashlib.sha256(obj).hexdigest()
            except TypeError:
                h = hashlib.sha3_256(f'{obj}'.encode()).hexdigest()
            hashes.append(h)

    s = '_'.join(hashes)
    h = hashlib.sha256(s.encode())

    return h.hexdigest()
