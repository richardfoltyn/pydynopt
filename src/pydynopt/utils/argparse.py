"""
Author: Richard Foltyn
"""

from argparse import ArgumentParser
from typing import Optional

import re


def add_toggle_arg(
        parser: ArgumentParser,
        name: str,
        dest: Optional[str]=None,
        default: bool = True,
        required: bool = False
) -> ArgumentParser:
    """
    Add a CLI argument that can toggle a certain feature (ON/OFF) by
    specifing either --name or --no-name.

    Parameters
    ----------
    parser : ArgumentParser
    name : str
        Name of the option to add.
    dest : str
        Attribute where option value should be stored (default: option name in lower
        case)
    default : bool
        Default value
    required : bool
        If true, mark as required option.ll
    """


    # strip initial dashes
    pattern = re.compile('-*(.*)')
    mtch = pattern.match(name)
    if mtch:
        name = mtch.group(1)

    if dest is None:
        dest = name.lower()
        dest = re.sub(r'[^a-z_]+', '_', dest)

    grp = parser.add_mutually_exclusive_group(required=required)
    grp.add_argument('--{:s}'.format(name), action='store_true', dest=dest)
    grp.add_argument('--no-{:s}'.format(name), action='store_false', dest=dest)
    kwargs = {dest: default}
    parser.set_defaults(**kwargs)

    return parser