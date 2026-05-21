"""
Unit tests for the Stata utility functions in pydynopt.utils.stata.

This module verifies:
1. Stata executable lookup paths (in system PATH, custom paths, and standard locations).
2. The batch Stata runner including option flags, macro configurations, log deletion,
   and working directory configurations under mocked execution.
"""

from pathlib import Path
import shutil
import tempfile
import unittest as ut
from unittest.mock import MagicMock, patch

from pydynopt.utils.stata import _check_stata_works, find_stata, run_stata


class TestStataUtils(ut.TestCase):
    def setUp(self):
        self.temp_dirs = []

    def tearDown(self):
        for d in self.temp_dirs:
            if Path(d).exists():
                shutil.rmtree(d)

    def create_temp_dir(self):
        d = tempfile.mkdtemp()
        self.temp_dirs.append(d)
        return Path(d)

    @patch('subprocess.run')
    def test_check_stata_works(self, mock_run):
        # Setup mock subprocess to simulate Stata successfully writing '1' to output.csv
        def side_effect(args, **kwargs):
            cwd = kwargs.get('cwd')
            if cwd:
                out_file = Path(cwd) / 'output.csv'
                out_file.write_text('1')
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Test with str path
        self.assertTrue(_check_stata_works('/usr/bin/stata'))

        # Test with Path path
        self.assertTrue(_check_stata_works(Path('/usr/bin/stata')))

    @patch('subprocess.run')
    def test_check_stata_works_failure(self, mock_run):
        # Simulate subprocess run that does not write '1' (e.g. license error or crash)
        mock_run.return_value = MagicMock(returncode=1)
        self.assertFalse(_check_stata_works('/usr/bin/stata'))

    @patch('shutil.which')
    @patch('pydynopt.utils.stata._check_stata_works')
    def test_find_stata_in_path(self, mock_check, mock_which):
        mock_which.side_effect = lambda cmd: (
            f'/usr/bin/{cmd}' if 'stata' in cmd else None
        )
        mock_check.return_value = True

        res = find_stata()
        self.assertIsInstance(res, Path)
        self.assertEqual(res, Path('/usr/bin/stata-mp'))

    @patch('shutil.which')
    @patch('glob.glob')
    @patch('pydynopt.utils.stata._check_stata_works')
    def test_find_stata_in_candidate_dirs(self, mock_check, mock_glob, mock_which):
        mock_which.return_value = None
        mock_glob.side_effect = lambda pat: (
            ['/opt/stata/stata-mp'] if 'stata-mp' in pat else []
        )
        mock_check.return_value = True

        # Test with str candidate dir
        res = find_stata(dirs='/opt/stata')
        self.assertEqual(res, Path('/opt/stata/stata-mp'))

        # Test with Path candidate dir
        res = find_stata(dirs=Path('/opt/stata'))
        self.assertEqual(res, Path('/opt/stata/stata-mp'))

        # Test with sequence of dirs
        res = find_stata(dirs=[Path('/fake'), '/opt/stata'])
        self.assertEqual(res, Path('/opt/stata/stata-mp'))

    @patch('shutil.which')
    @patch('pydynopt.utils.stata._check_stata_works')
    @patch('subprocess.run')
    def test_run_stata_success(self, mock_run, mock_check, mock_which):
        mock_which.return_value = '/usr/bin/stata'
        mock_check.return_value = True

        tmp_dir = self.create_temp_dir()
        do_file = tmp_dir / 'test.do'
        do_file.touch()

        # Simulate Stata creating a log file in the cwd
        def side_effect(args, **kwargs):
            cwd = kwargs.get('cwd')
            log_file = Path(cwd) / 'test.log'
            log_file.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Test run_stata with Path arguments and deleting log
        code = run_stata(
            do_file=do_file, exe=Path('/usr/bin/stata'), cwd=tmp_dir, delete_log=True
        )
        self.assertEqual(code, 0)
        # Log should be deleted
        self.assertFalse((tmp_dir / 'test.log').exists())

    @patch('shutil.which')
    @patch('pydynopt.utils.stata._check_stata_works')
    @patch('subprocess.run')
    def test_run_stata_keep_log(self, mock_run, mock_check, mock_which):
        mock_which.return_value = '/usr/bin/stata'
        mock_check.return_value = True

        tmp_dir = self.create_temp_dir()
        do_file = tmp_dir / 'test.do'
        do_file.touch()

        # Simulate Stata creating a log file in the cwd
        def side_effect(args, **kwargs):
            cwd = kwargs.get('cwd')
            log_file = Path(cwd) / 'test.log'
            log_file.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Test run_stata with str arguments and keeping log
        code = run_stata(
            do_file=str(do_file),
            exe='/usr/bin/stata',
            cwd=str(tmp_dir),
            delete_log=False,
        )
        self.assertEqual(code, 0)
        # Log should still exist
        self.assertTrue((tmp_dir / 'test.log').exists())

    @patch('shutil.which')
    @patch('pydynopt.utils.stata._check_stata_works')
    @patch('subprocess.run')
    def test_run_stata_with_macros(self, mock_run, mock_check, mock_which):
        mock_which.return_value = '/usr/bin/stata'
        mock_check.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        tmp_dir = self.create_temp_dir()
        do_file = tmp_dir / 'test.do'
        do_file.touch()
        conf_file = tmp_dir / 'config.do'

        macros = {'myvar': 42}

        # Test running with config file and macros
        code = run_stata(
            do_file=do_file,
            conf_file=conf_file,
            macros=macros,
            global_macros=True,
            cwd=tmp_dir,
        )
        self.assertEqual(code, 0)
        # Config file should be automatically deleted after execution
        self.assertFalse(conf_file.exists())


if __name__ == '__main__':
    ut.main()
