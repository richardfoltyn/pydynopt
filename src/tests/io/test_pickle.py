"""Unit tests for compressed and uncompressed pickle persistence."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import pydynopt.io.pickle as pickle_io

_OBJECT = {'name': 'test-object', 'values': [1, 2, 3]}


def test_dump_uncompressed_returns_requested_path(tmp_path: Path) -> None:
    """Return the requested path for uncompressed output and round-trip it."""
    path = tmp_path / 'model.pkl'

    result = pickle_io.dump(path, _OBJECT, compress=False)

    assert result == path
    assert pickle_io.load(result) == _OBJECT


@pytest.mark.parametrize('suffix', ('.gz', '.xz', '.lz4', '.zstd'))
def test_dump_compressed_returns_requested_path(tmp_path: Path, suffix: str) -> None:
    """Return and round-trip paths for every supported compression format."""
    path = tmp_path / f'model.pkl{suffix}'

    result = pickle_io.dump(path, _OBJECT, nthreads=1)

    assert result == path
    assert pickle_io.load(result) == _OBJECT


def test_dump_appends_default_zstd_suffix(tmp_path: Path) -> None:
    """Return the appended Zstandard path used by default compression."""
    requested_path = tmp_path / 'model.pkl'
    expected_path = tmp_path / 'model.pkl.zstd'

    result = pickle_io.dump(requested_path, _OBJECT, nthreads=1)

    assert result == expected_path
    assert not requested_path.exists()
    assert pickle_io.load(result) == _OBJECT


def test_dump_combines_relative_path_and_directory(tmp_path: Path) -> None:
    """Return a relative filename combined with the requested directory."""
    expected_path = tmp_path / 'model.pkl'

    result = pickle_io.dump(
        'model.pkl',
        _OBJECT,
        directory=str(tmp_path),
        compress=False,
    )

    assert result == expected_path
    assert pickle_io.load(result) == _OBJECT


def test_dump_overwrite_true_returns_original_path(tmp_path: Path) -> None:
    """Overwrite an existing file and return its original path."""
    path = tmp_path / 'model.pkl'
    pickle_io.dump(path, 'old', compress=False)

    result = pickle_io.dump(path, 'new', compress=False, overwrite=True)

    assert result == path
    assert pickle_io.load(path) == 'new'


def test_dump_overwrite_false_returns_numbered_path(tmp_path: Path) -> None:
    """Preserve an existing file and return an independently loadable path."""
    path = tmp_path / 'model.pkl'
    pickle_io.dump(path, 'old', compress=False)

    result = pickle_io.dump(path, 'new', compress=False, overwrite=False)

    assert result == tmp_path / 'model_000.pkl'
    assert pickle_io.load(path) == 'old'
    assert pickle_io.load(result) == 'new'


def test_dump_numbered_path_preserves_compound_suffix(tmp_path: Path) -> None:
    """Retain pickle and compression suffixes when numbering a collision."""
    path = tmp_path / 'model.pkl.xz'
    pickle_io.dump(path, 'old')

    result = pickle_io.dump(path, 'new', overwrite=False)

    assert result == tmp_path / 'model_000.pkl.xz'
    assert pickle_io.load(path) == 'old'
    assert pickle_io.load(result) == 'new'


def test_get_cached_object_discovers_default_zstd_cache(tmp_path: Path) -> None:
    """Discover a cache written with the default Zstandard compression."""
    cache_path = tmp_path / 'cache.pkl'
    written_path = pickle_io.dump(cache_path, _OBJECT, nthreads=1)
    compute = Mock(side_effect=AssertionError('cache function should not be called'))

    result = pickle_io.get_cached_object(compute, cache_file=cache_path)

    assert written_path == tmp_path / 'cache.pkl.zstd'
    assert result == _OBJECT
    compute.assert_not_called()


def test_dump_forwards_compression_open_kwargs(tmp_path: Path) -> None:
    """Forward caller-supplied keywords to the compression open function."""
    path = tmp_path / 'model.pkl.gz'
    gzip_open = pickle_io.gzip.open

    with patch.object(pickle_io.gzip, 'open', wraps=gzip_open) as mock_open:
        result = pickle_io.dump(path, _OBJECT, compresslevel=1)

    mock_open.assert_called_once_with(path, 'wb', compresslevel=1)
    assert pickle_io.load(result) == _OBJECT
