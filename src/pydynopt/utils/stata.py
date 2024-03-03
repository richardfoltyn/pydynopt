"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import logging
import os
from typing import Optional, Sequence, Mapping


def _check_stata_works(path_exe: str) -> bool:
    """
    Check that the given path to Stata executable works, i.e. Stata can actuall run
    stuff. Even an existing executable might not work if the corresponding license is
    not valid.
    E.g., stata-mp is present on Linux even irrespective of whether the user has a
    license or not, and it will refuse to run if no license was obtained.

    Parameters
    ----------
    path_exe : str

    Returns
    -------
    bool
    """
    import sys
    import subprocess
    import tempfile
    import shutil

    dtmp = tempfile.mkdtemp()

    # Result file
    out_file = os.path.join(dtmp, 'output.csv')
    # Do file
    do_file = os.path.join(dtmp, 'batch.do')

    contents = f"""
        set obs 1
        gen var = 1
        export delimited using `"{out_file}"', replace novarnames
    """

    with open(do_file, 'wt') as f:
        f.write(contents)

    # Batch mode: use /e on Windows, otherwise Stata shows a dialog box that the
    # script completed.
    batch = '/e' if sys.platform.lower().startswith('win') else '-b'

    # Execute Stata
    result = subprocess.run([path_exe, batch, do_file], cwd=dtmp)

    # Read output file
    works = False
    if os.path.isfile(out_file):
        with open(out_file, 'rt') as f:
            contents = f.readline().strip()
            works = contents == "1"

    # Delete temp directory
    shutil.rmtree(dtmp)

    return works


def find_stata(dirs: Optional[str | Sequence[str]] = None) -> str:
    """
    Try to locate a working Stata executable in the system PATH, in known locations
    are in the optionally given additional directories.

    Parameters
    ----------
    dirs : str or Sequence of str

    Returns
    -------
    str
        Absolute path of Stata executable, if found.
    """
    import sys
    import shutil
    import glob

    is_win = sys.platform.lower().startswith('win')

    # List of candidate executable files
    if is_win:
        exes = [
            f'Stata{v}{bits}.exe'
            for bits in ('-64', '')
            for v in ('MP', 'SE', 'IC', '')
        ]
    else:
        exes = [f'stata{v}' for v in ('-mp', '-se', '')]

    for exe in exes:
        if path_exe := shutil.which(exe):
            # Try to execute so code since exe might be present, but user might not
            # have the license for MP
            if _check_stata_works(path_exe):
                return path_exe

    # executable was not found in PATH, try other candidate directories
    user_dirs = dirs

    if is_win:
        sysdrive = os.environ['SystemDrive']
        if sysdrive.endswith(':'):
            sysdrive += '\\'
        dirs = [
            os.environ['PROGRAMFILES'],
            os.environ['PROGRAMFILES(x86)'],
            sysdrive,
        ]
        dirs = [os.path.join(d, 'Stata*') for d in dirs if d]
    else:
        dirs = ['/opt/stata/**', '/usr/local/**']

    # Append any additional candidate strings
    if isinstance(user_dirs, str):
        dirs.insert(0, user_dirs)
    elif isinstance(user_dirs, Sequence):
        dirs = list(user_dirs) + dirs

    for exe in exes:
        for d in dirs:
            pattern = os.path.join(d, exe)

            candidates = glob.glob(pattern)
            for path_exe in candidates:
                if _check_stata_works(path_exe):
                    return path_exe

    raise FileNotFoundError(f'Stata executable not in found.')


def run_stata(
    do_file: str,
    exe: Optional[str] = None,
    conf_file: Optional[str] = None,
    macros: Optional[Mapping] = None,
    global_macros: bool = False,
    cwd: Optional[str] = None,
    env: Optional[Mapping] = None,
    delete_log: bool = True,
    **kwargs,
) -> int:
    """
    Run Stata in batch mode, executing the given do-file. Optionally
    creates a configuration do-file with given macros and adds
    given variables to the environment.

    Parameters
    ----------
    do_file : str
        Do-file to run
    exe : str
        Stata exectuable. If not an absolute path, tries to locate
        the executable in the PATH, and if that files, checks standard
        locations on Linux.
    conf_file : str, optional
        If present, creates a configuration do-file at the given location
        and populates it with variables from `macros`.
    macros : Mapping, optional
        If present, writes key-value pairs as macros to do-file
        given in `conf_file`.
    global_macros : bool
        If True, write `macros` as global macros, and as local macros otherwise.
    cwd : str, optional
        If present, sets the current working directory for the Stata process.
    env : Mapping, optional
        If present, augments the environment for the Stata process with
        the given name-value pairs.
    delete_log : bool
        If true, delete the log file that Stata automatically generates in batch mode.
    kwargs
        Keyword arguments passed to subprocess.run()

    Returns
    -------
    int
        Stata return code
    """

    import sys
    import subprocess

    logger = logging.getLogger('IO')

    # --- Locate Stata executable ---

    path_exe = exe
    if not path_exe:
        path_exe = find_stata()

    logger.info(f'Using Stata executable {path_exe}')

    # --- Create (optional) config file ---

    if conf_file and macros:
        logger.info(f'Creating config file {conf_file}')

        prefix = 'global' if global_macros else 'local'

        with open(conf_file, 'wt') as f:
            for key, value in macros.items():
                print(f'{prefix} {key} = {value}', file=f)

    # --- Run Stata ---

    # Merge inherited environment with additional variables (note that
    # if env is passed, subprocess.run() replaces the inherited env variables)
    env_all = dict(os.environ)
    env_all.update(env)

    kw = kwargs.copy()
    kw.update({'cwd': cwd, 'env': env_all})

    # Batch mode: use /e on Windows, otherwise Stata shows a dialog box that the
    # script completed.
    batch = '/e' if sys.platform.lower().startswith('win') else '-b'

    # Execute Stata
    result = subprocess.run([path_exe, batch, do_file], **kw)

    if result.returncode:
        raise RuntimeWarning(f'Stata exited with return code {result.returncode}')

    # Delete config file
    if conf_file and macros:
        try:
            os.remove(conf_file)
        except PermissionError:
            # On Windows this might not be possible because some other process still
            # is accessing the file
            pass

    # Delete the log file that Stata automatically generates, assuming that logging
    # is explicitly handled within the do-file
    if delete_log:
        base, ext = os.path.splitext(do_file)
        fn = os.path.join(cwd, f'{base}.log')
        if os.path.isfile(fn):
            try:
                os.remove(fn)
            except PermissionError:
                pass

    return result.returncode
