"""
Helper functions for parsing CLI arguments using the argparse module.

Author: Richard Foltyn
"""

from argparse import ArgumentParser
from collections.abc import Sequence
import re
from typing import Any


def parse_bool(s: Any) -> bool:
    """
    Parse a string as a boolean value.

    This maps strings that contain a numerical value 0 or 0.0 to False.
    Otherwise, the usual conversion rules apply.

    Parameters
    ----------
    s

    Returns
    -------
    Parsed boolean value.
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
    dest: str | None = None,
    default: bool = True,
    required: bool = False,
) -> ArgumentParser:
    """
    Add a CLI argument that can toggle a certain feature (ON/OFF).

    This specifies either --name or --no-name.

    Parameters
    ----------
    parser
    name
        Name of the option to add.
    dest
        Attribute where option value should be stored (default: option name in lower
        case)
    default
        Default value
    required
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
    grp.add_argument(
        f'--{name}',
        action='store',
        dest=dest,
        nargs='?',
        default=default,
        const=True,
        type=parse_bool,
    )
    grp.add_argument(f'--no-{name}', action='store_false', dest=dest)
    kwargs = {dest: default}
    parser.set_defaults(**kwargs)

    return parser


def flatten_list_args(value: str | Sequence[str] | None) -> list[str]:
    """
    Flatten a list of string values.

    Parameters
    ----------
    value
        List of (multiple) option arguments

    Returns
    -------
    Flattened list of string values.
    """
    from itertools import chain

    if value is None:
        return []

    if isinstance(value, str):
        value = list(value.split(','))
    else:
        value = list(chain(*tuple(v.split(',') for v in value)))

    # Remove any surrounding spaces
    value = [v.strip() for v in value]

    return value


def strip_quotes(value: str | Sequence[str] | None) -> str | list[str] | None:
    """
    Strip any single or double quotes from an option value.

    This handles a string or list of strings.

    Parameters
    ----------
    value

    Returns
    -------
    The stripped string or sequence of strings, or None.
    """
    if isinstance(value, str):
        return value.strip('"\' ') if value else None
    elif isinstance(value, Sequence):
        return [v.strip('"\' ') for v in value]
    else:
        return value
