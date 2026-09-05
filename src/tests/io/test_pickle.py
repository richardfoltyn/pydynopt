"""Unit tests for compressed and uncompressed pickle persistence."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

import pydynopt.io.pickle as pickle_io

_OBJECT = {'name': 'test-object', 'values': [1, 2, 3]}


class _PickleableObject:
    def __init__(self, value: int) -> None:
        self.value = value


class _UnpickleableObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError('objects of this type cannot be pickled')

    def __repr__(self) -> str:
        return f'_UnpickleableObject({self.value})'


def test_get_hash_value_is_invariant_to_mapping_order() -> None:
    """Produce identical hashes for mappings regardless of their insertion order."""
    assert pickle_io.get_hash_value({'a': 1, 'b': 2}) == pickle_io.get_hash_value(
        {'b': 2, 'a': 1}
    )
    assert pickle_io.get_hash_value(a=1, b=2) == pickle_io.get_hash_value(b=2, a=1)
    assert pickle_io.get_hash_value({'a': {'c': 3, 'd': 4}}) == (
        pickle_io.get_hash_value({'a': {'d': 4, 'c': 3}})
    )


def test_get_hash_value_includes_all_array_elements() -> None:
    """Distinguish arrays that differ only at an interior element."""
    arr1 = np.zeros(2_000)
    arr2 = arr1.copy()
    arr2[1_000] = 1

    assert pickle_io.get_hash_value(arr1) != pickle_io.get_hash_value(arr2)


def test_get_hash_value_preserves_float_precision() -> None:
    """Distinguish floats that differ beyond their usual display precision."""
    assert pickle_io.get_hash_value(1.0) != pickle_io.get_hash_value(1.0 + 1e-15)


def test_get_hash_value_ignores_array_print_options() -> None:
    """Produce the same array hash regardless of NumPy display settings."""
    arr = np.linspace(0, 1, 2_000)
    expected = pickle_io.get_hash_value(arr)

    with np.printoptions(precision=2, threshold=10):
        actual = pickle_io.get_hash_value(arr)

    assert actual == expected


@pytest.mark.parametrize(
    'value',
    (
        None,
        True,
        False,
        0,
        1.5,
        'text',
        b'bytes',
        bytearray(b'bytes'),
        memoryview(b'bytes'),
    ),
)
def test_get_hash_value_supports_scalar_types(value: object) -> None:
    """Hash supported scalar and bytes-like values consistently."""
    expected = pickle_io.get_hash_value(value)
    assert pickle_io.get_hash_value(value) == expected


def test_get_hash_value_distinguishes_bytes_like_types() -> None:
    """Distinguish bytes-like values with different type tags."""
    values = (b'value', bytearray(b'value'), memoryview(b'value'))
    hashes = {pickle_io.get_hash_value(value) for value in values}
    assert len(hashes) == len(values)


def test_get_hash_value_distinguishes_sequence_types() -> None:
    """Distinguish lists, tuples, and generic sequences with equal contents."""
    values = ([1, 2], (1, 2), range(1, 3))
    hashes = {pickle_io.get_hash_value(value) for value in values}
    assert len(hashes) == len(values)


def test_get_hash_value_handles_nan() -> None:
    """Canonicalize scalar NaN values."""
    assert pickle_io.get_hash_value(float('nan')) == pickle_io.get_hash_value(
        float('nan')
    )


def test_get_hash_value_includes_array_dtype_shape_and_endianness() -> None:
    """Distinguish arrays with equal logical values but different metadata."""
    arrays = (
        np.array([1, 2], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([[1, 2]], dtype=np.int32),
        np.array([1, 2], dtype='>i4'),
    )
    hashes = {pickle_io.get_hash_value(arr) for arr in arrays}
    assert len(hashes) == len(arrays)


def test_get_hash_value_ignores_array_memory_layout() -> None:
    """Produce the same hash for arrays with equal values and different layouts."""
    arr_c = np.array([[1, 2], [3, 4]], order='C')
    arr_f = np.array(arr_c, order='F')
    assert pickle_io.get_hash_value(arr_c) == pickle_io.get_hash_value(arr_f)


def test_get_hash_value_supports_numpy_scalars_and_object_arrays() -> None:
    """Hash NumPy scalars and object arrays based on their type and contents."""
    assert pickle_io.get_hash_value(np.int32(1)) != pickle_io.get_hash_value(
        np.int64(1)
    )

    arr1 = np.array([{'key': 'value'}, 1], dtype=object)
    arr2 = np.array([{'key': 'value'}, 1], dtype=object)
    arr3 = np.array([1, {'key': 'value'}], dtype=object)
    assert pickle_io.get_hash_value(arr1) == pickle_io.get_hash_value(arr2)
    assert pickle_io.get_hash_value(arr1) != pickle_io.get_hash_value(arr3)


def test_get_hash_value_handles_nested_mixed_structures() -> None:
    """Remain mapping-order invariant while detecting nested value changes."""
    value1 = {'items': [np.array([1, 2]), ('key', b'value')], 'enabled': True}
    value2 = {'enabled': True, 'items': [np.array([1, 2]), ('key', b'value')]}
    value3 = {'enabled': True, 'items': [np.array([1, 3]), ('key', b'value')]}
    assert pickle_io.get_hash_value(value1) == pickle_io.get_hash_value(value2)
    assert pickle_io.get_hash_value(value1) != pickle_io.get_hash_value(value3)


def test_get_hash_value_handles_pickleable_and_unpickleable_objects() -> None:
    """Hash pickled objects and fall back to stable representations on failure."""
    assert pickle_io.get_hash_value(_PickleableObject(1)) == pickle_io.get_hash_value(
        _PickleableObject(1)
    )
    assert pickle_io.get_hash_value(_PickleableObject(1)) != pickle_io.get_hash_value(
        _PickleableObject(2)
    )
    assert pickle_io.get_hash_value(_UnpickleableObject(1)) == pickle_io.get_hash_value(
        _UnpickleableObject(1)
    )
    assert pickle_io.get_hash_value(_UnpickleableObject(1)) != pickle_io.get_hash_value(
        _UnpickleableObject(2)
    )


def test_get_hash_value_is_deterministic_across_processes() -> None:
    """Produce the same hash in a separate Python process."""
    value = {'items': [1, 2.5, 'text'], 'metadata': {'enabled': True}}
    script = """
from pydynopt.io.pickle import get_hash_value
print(get_hash_value({'items': [1, 2.5, 'text'], 'metadata': {'enabled': True}}))
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == pickle_io.get_hash_value(value)


def test_atomic_dump_preserves_existing_file_on_failure(tmp_path: Path) -> None:
    """Preserve existing file and clean up temporary files if dump fails."""
    path = tmp_path / 'cache'
    pickle_io.dump(path, {'value': 'old'}, compress=False)

    with (
        patch.object(pickle_io.pickle, 'dump', side_effect=RuntimeError('failed')),
        pytest.raises(RuntimeError, match='failed'),
    ):
        pickle_io.dump(path, {'value': 'new'}, compress=False, atomic=True)

    assert pickle_io.load(path) == {'value': 'old'}
    assert list(tmp_path.glob('.cache.*.tmp')) == []


@pytest.mark.parametrize(
    'suffix',
    (
        '.gz',
        '.xz',
        '',
        pytest.param(
            '.lz4',
            marks=pytest.mark.skipif(
                importlib.util.find_spec('lz4') is None, reason='lz4 is not installed'
            ),
        ),
        pytest.param(
            '.zst',
            marks=pytest.mark.skipif(
                importlib.util.find_spec('pyzstd') is None,
                reason='pyzstd is not installed',
            ),
        ),
        pytest.param(
            '.zstd',
            marks=pytest.mark.skipif(
                importlib.util.find_spec('pyzstd') is None,
                reason='pyzstd is not installed',
            ),
        ),
    ),
)
def test_atomic_dump_roundtrip(tmp_path: Path, suffix: str) -> None:
    """Round-trip pickle data using atomic writes across compression formats."""
    path = tmp_path / f'cache{suffix}'
    obj = {'value': [1, 2, 3]}

    pickle_io.dump(path, obj, compress=bool(suffix), atomic=True)

    assert pickle_io.load(path) == obj


@pytest.mark.parametrize(
    'suffix',
    (
        '.xz',
        '.gz',
        pytest.param(
            '.lz4',
            marks=pytest.mark.skipif(
                importlib.util.find_spec('lz4') is None, reason='lz4 is not installed'
            ),
        ),
        pytest.param(
            '.zst',
            marks=pytest.mark.skipif(
                importlib.util.find_spec('pyzstd') is None,
                reason='pyzstd is not installed',
            ),
        ),
    ),
)
def test_corrupt_cache_is_recomputed(
    tmp_path: Path, suffix: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Recompute and replace cache when existing cache file is corrupt."""
    path = tmp_path / f'cache{suffix}'
    path.write_bytes(b'not a pickle')
    compute = Mock(return_value={'value': 'new'})

    assert pickle_io.get_cached_object(compute, cache_file=path) == {'value': 'new'}
    compute.assert_called_once_with()
    assert pickle_io.load(path) == {'value': 'new'}
    assert 'corrupt' in caplog.text


def test_dump_uncompressed_returns_requested_path(tmp_path: Path) -> None:
    """Return the requested path for uncompressed output and round-trip it."""
    path = tmp_path / 'model.pkl'
    result = pickle_io.dump(path, _OBJECT, compress=False)
    assert result == path
    assert pickle_io.load(result) == _OBJECT


@pytest.mark.parametrize('suffix', ('.gz', '.xz', '.lz4', '.zstd', '.zst'))
def test_dump_compressed_returns_requested_path(tmp_path: Path, suffix: str) -> None:
    """Return and round-trip paths for every supported compression format."""
    path = tmp_path / f'model.pkl{suffix}'
    result = pickle_io.dump(path, _OBJECT, nthreads=1)
    assert result == path
    assert pickle_io.load(result) == _OBJECT


def test_dump_appends_default_zst_suffix(tmp_path: Path) -> None:
    """Return the appended Zstandard path used by default compression."""
    requested_path = tmp_path / 'model.pkl'
    expected_path = tmp_path / 'model.pkl.zst'
    result = pickle_io.dump(requested_path, _OBJECT, nthreads=1)
    assert result == expected_path
    assert not requested_path.exists()
    assert pickle_io.load(result) == _OBJECT


def test_dump_combines_relative_path_and_directory(tmp_path: Path) -> None:
    """Return a relative filename combined with the requested directory."""
    expected_path = tmp_path / 'model.pkl'
    result = pickle_io.dump(
        'model.pkl', _OBJECT, directory=str(tmp_path), compress=False
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


def test_get_cached_object_discovers_default_zst_cache(tmp_path: Path) -> None:
    """Discover a cache written with the default Zstandard compression."""
    cache_path = tmp_path / 'cache.pkl'
    written_path = pickle_io.dump(cache_path, _OBJECT, nthreads=1)
    compute = Mock(side_effect=AssertionError('cache function should not be called'))
    result = pickle_io.get_cached_object(compute, cache_file=cache_path)
    assert written_path == tmp_path / 'cache.pkl.zst'
    assert result == _OBJECT
    compute.assert_not_called()


def test_get_cached_object_discovers_zstd_cache(tmp_path: Path) -> None:
    """Discover a cache written with the .zstd compression format."""
    cache_path = tmp_path / 'cache.pkl.zstd'
    written_path = pickle_io.dump(cache_path, _OBJECT, nthreads=1)
    compute = Mock(side_effect=AssertionError('cache function should not be called'))
    result = pickle_io.get_cached_object(compute, cache_file=tmp_path / 'cache.pkl')
    assert written_path == tmp_path / 'cache.pkl.zstd'
    assert result == _OBJECT
    compute.assert_not_called()


def test_dump_forwards_compression_open_kwargs(tmp_path: Path) -> None:
    """Forward caller-supplied keywords to the compression open function."""
    path = tmp_path / 'model.pkl.gz'
    gzip_open = pickle_io.gzip.open
    with patch.object(pickle_io.gzip, 'open', wraps=gzip_open) as mock_open:
        result = pickle_io.dump(path, _OBJECT, compresslevel=1)
    mock_open.assert_called_once()
    assert pickle_io.load(result) == _OBJECT
