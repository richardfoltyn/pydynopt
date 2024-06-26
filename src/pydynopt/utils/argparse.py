"""
Author: Richard Foltyn
"""

from argparse import ArgumentParser
from typing import Optional, Any, Sequence

import re


def parse_bool(s: Any) -> bool:
    """
    Parse string as bool, mapping strings that contain a numerical value 0 or 0.0
    to False. Otherwise, the usual conversion rules apply.

    Parameters
    ----------
    s : object

    Returns
    -------
    value : bool
    """

    if isinstance(s, bool):
        return s

    try:
        value = bool(float(s))
    except ValueError:
        value = bool(s)

    return value


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
    # We want to support the following arguments:
    #       --name (set to True)
    #       --name=1 (set to True)
    #       --name=0 (set to False)
    #       --no-name (set to False)
    #   Set to default if no argument was specified.
    # With nargs='?' for optional arguments, when --name was given then the value
    # from const is used.
    # If neither --name nor --no-name is given, the default value is used.
    # We use a custom parser as otherwise '0' is set to true with type=bool.
    grp.add_argument(f'--{name}', action='store', dest=dest, nargs='?',
                     default=default, const=True, type=parse_bool)
    grp.add_argument(f'--no-{name}', action='store_false', dest=dest)
    kwargs = {dest: default}
    parser.set_defaults(**kwargs)

    return parser


def flatten_list_args(value: str | Sequence[str] | None) -> list[str]:
    """
    Flatten a list of string values.

    Parameters
    ----------
    value : str or list of str or None
        List of (multiple) option arguments

    Returns
    -------
    list
    """
    from itertools import chain

    if value is None:
        return []

    if isinstance(value, str):
        value = list(value.split(","))
    else:
        value = list(chain(*tuple(v.split(",") for v in value)))

    # Remove any surrounding spaces
    value = [v.strip() for v in value]

    return value


def strip_quotes(value: str | Sequence[str] | None) -> str | list[str] | None:
    """
    String any single or double quotes from a string or list of strings
    option value.

    Parameters
    ----------
    value : str or Sequence of str, optional

    Returns
    -------
    str or list of str or None
    """

    if not value:
        return value

    if isinstance(value, list):
        value = [v.strip('"\' ') for v in value]
    else:
        value = value.strip('"\' ') if value else None

    return value