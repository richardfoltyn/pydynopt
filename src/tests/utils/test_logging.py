"""Unit tests for the logging utility functions in pydynopt.utils.logging."""

import logging
import os
from unittest.mock import MagicMock

import pytest

from pydynopt.utils.logging import log_env_vars


def test_log_env_vars_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_env_vars when no relevant environment variables are set."""
    logger = MagicMock()
    monkeypatch.setattr(os, 'environ', {})

    log_env_vars(logger=logger)
    logger.log.assert_not_called()


def test_log_env_vars_with_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_env_vars when relevant environment variables are set."""
    logger = MagicMock()
    fake_env = {
        'OMP_NUM_THREADS': '4',
        'mkl_num_threads': '2',
        'OPENBLAS_NUM_THREADS': '8',
        'SOME_OTHER_VAR': 'ignore_me',
    }
    monkeypatch.setattr(os, 'environ', fake_env)

    log_env_vars(logger=logger, level=logging.INFO)

    logger.log.assert_any_call(logging.INFO, 'Environment variables:')
    logger.log.assert_any_call(logging.INFO, '  OMP_NUM_THREADS: 4')
    logger.log.assert_any_call(logging.INFO, '  OPENBLAS_NUM_THREADS: 8')
    logger.log.assert_any_call(logging.INFO, '  mkl_num_threads: 2')
    assert logger.log.call_count == 4
