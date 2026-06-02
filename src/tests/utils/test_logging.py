"""Unit tests for the logging utility functions in pydynopt.utils.logging."""

import logging
import os
import unittest as ut
from unittest.mock import MagicMock, patch

from pydynopt.utils.logging import log_env_vars


class TestLoggingUtils(ut.TestCase):
    """Test suite for the logging utility functions."""

    def test_log_env_vars_empty(self):
        """Test log_env_vars when no relevant environment variables are set."""
        logger = MagicMock()

        # Patch os.environ to have absolutely no relevant keys
        with patch.dict(os.environ, {}, clear=True):
            log_env_vars(logger=logger)

        logger.log.assert_not_called()

    def test_log_env_vars_with_vars(self):
        """Test log_env_vars when relevant environment variables are set."""
        logger = MagicMock()

        fake_env = {
            'OMP_NUM_THREADS': '4',
            'mkl_num_threads': '2',
            'OPENBLAS_NUM_THREADS': '8',
            'SOME_OTHER_VAR': 'ignore_me',
        }

        with patch.dict(os.environ, fake_env, clear=True):
            log_env_vars(logger=logger, level=logging.INFO)

        logger.log.assert_any_call(logging.INFO, 'Environment variables:')
        logger.log.assert_any_call(logging.INFO, '  OMP_NUM_THREADS: 4')
        logger.log.assert_any_call(logging.INFO, '  OPENBLAS_NUM_THREADS: 8')
        logger.log.assert_any_call(logging.INFO, '  mkl_num_threads: 2')
        self.assertEqual(logger.log.call_count, 4)


if __name__ == '__main__':
    ut.main()
