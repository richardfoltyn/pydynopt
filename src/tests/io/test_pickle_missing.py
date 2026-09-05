"""
Unit tests for pickle persistence when optional compression libraries are missing.

Author: Richard Foltyn
"""

import sys
from unittest.mock import patch

import pytest

from pydynopt.io.pickle import dump, load


def test_missing_lz4_dump(tmp_path) -> None:
    """Verify that dump raises ValueError with install hint when lz4 is missing."""
    path = tmp_path / 'model.pkl.lz4'
    with (
        patch.dict(sys.modules, {'lz4': None, 'lz4.frame': None}),
        pytest.raises(ValueError, match='Install pydynopt\\[compression\\]'),
    ):
        dump(path, {'test': 1})


def test_missing_lz4_load(tmp_path) -> None:
    """Verify that load raises ValueError with install hint when lz4 is missing."""
    path = tmp_path / 'model.pkl.lz4'
    path.write_bytes(b'dummy content')
    with (
        patch.dict(sys.modules, {'lz4': None, 'lz4.frame': None}),
        pytest.raises(ValueError, match='Install pydynopt\\[compression\\]'),
    ):
        load(path)


def test_missing_zstd_dump(tmp_path) -> None:
    """Verify that dump raises ValueError with install hint when zstd is missing."""
    path = tmp_path / 'model.pkl.zstd'
    with (
        patch.dict(
            sys.modules, {'compression': None, 'compression.zstd': None, 'pyzstd': None}
        ),
        pytest.raises(ValueError, match='Install pydynopt\\[compression\\]'),
    ):
        dump(path, {'test': 1})


def test_missing_zstd_load(tmp_path) -> None:
    """Verify that load raises ImportError with install hint when zstd is missing."""
    path = tmp_path / 'model.pkl.zstd'
    path.write_bytes(b'dummy content')
    with (
        patch.dict(
            sys.modules, {'compression': None, 'compression.zstd': None, 'pyzstd': None}
        ),
        pytest.raises(ImportError, match='Install pydynopt\\[compression\\]'),
    ):
        load(path)
