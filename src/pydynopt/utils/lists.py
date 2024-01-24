"""
Utility functions for working with list-like collections.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from typing import Any
from collections.abc import Iterable, Mapping

__all__ = ['anything_to_list', 'anything_to_tuple', 'anything_to_dict']


def anything_to_list(value, force: bool = False) -> list | None:
    """
    Covert a given value to a list (with potentially only one element).

    Parameters
    ----------
    value : object
    force : bool
        If true, return empty list even if input object is None

    Returns
    -------
    list or None
        Input data converted to a list
    """

    # Quick exit
    if isinstance(value, list):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame, Series
        has_pandas = True
    except ImportError:
        DataFrame = None
        Series = None

    items = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = [value]
        elif has_pandas and isinstance(value, (DataFrame, Series)):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = [value]
        elif isinstance(value, Iterable):
            items = list()
            items.extend(value)
        else:
            items = [value]

    if force and items is None:
        items = []

    return items


def anything_to_tuple(value, force: bool = False) -> tuple | None:
    """
    Covert a given value to a tuple (with potentially only one element).

    Parameters
    ----------
    value : object
    force : bool
        If true, return empty tuple even if input object is None

    Returns
    -------
    tuple or None
        Input data converted to a tuple
    """

    # quick exit
    if isinstance(value, tuple):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame, Series
        has_pandas = True
    except ImportError:
        DataFrame = None
        Series = None

    items = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = (value, )
        elif has_pandas and isinstance(value, (DataFrame, Series)):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = (value, )
        elif isinstance(value, Iterable):
            items = tuple(value)
        else:
            items = (value, )

    if force and items is None:
        items = tuple()

    return items


def anything_to_dict(value: Any, force: bool = False) -> dict | None:
    """
    Convert given object to a dictionary using common-sense rules.

    Parameters
    ----------
    value : object
        Anything that can be reasonably converted to a dictionary
    force : bool
        If true, return an empty dictionary even if no meaningful conversion is
        possible.

    Returns
    -------
    dict or None
    """

    if isinstance(value, dict):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame, Series
        has_pandas = True
    except ImportError:
        DataFrame = None

    items = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = {value: None}
        elif isinstance(value, Mapping):
            items = dict(value)
        elif has_pandas and isinstance(value, DataFrame):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = {name: column for name, column in value.items()}
        elif isinstance(value, Iterable):
            # Any iterable other than the ones covered above: create dict with all
            # values set to None
            items = {k: None for k in value}
        else:
            items = {value: None}

    if force and items is None:
        items = dict()

    return items