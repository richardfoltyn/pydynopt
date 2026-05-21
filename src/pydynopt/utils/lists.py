"""
Utility functions for working with collections.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

__all__ = ['anything_to_dict', 'anything_to_list', 'anything_to_tuple']


@overload
def anything_to_list(value: Any, force: Literal[True]) -> list[Any]: ...


@overload
def anything_to_list(value: Any, force: Literal[False] = False) -> list[Any] | None: ...


@overload
def anything_to_list(value: Any, force: bool) -> list[Any] | None: ...


def anything_to_list(value: Any, force: bool = False) -> list[Any] | None:
    """
    Covert a given value to a list (with potentially only one element).

    Parameters
    ----------
    value
    force
        If true, return empty list even if input object is None

    Returns
    -------
    Input data converted to a list.
    """
    # Quick exit
    if isinstance(value, list):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame, Series

        has_pandas = True
    except ImportError:
        DataFrame: Any = None
        Series: Any = None

    items = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = [value]
        elif (
            has_pandas
            and DataFrame is not None
            and Series is not None
            and isinstance(value, (DataFrame, Series))
        ):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = [value]
        elif isinstance(value, Iterable):
            items = []
            items.extend(value)
        else:
            items = [value]

    if force and items is None:
        items = []

    return items


@overload
def anything_to_tuple(value: Any, force: Literal[True]) -> tuple[Any, ...]: ...


@overload
def anything_to_tuple(
    value: Any, force: Literal[False] = False
) -> tuple[Any, ...] | None: ...


@overload
def anything_to_tuple(value: Any, force: bool) -> tuple[Any, ...] | None: ...


def anything_to_tuple(value: Any, force: bool = False) -> tuple[Any, ...] | None:
    """
    Covert a given value to a tuple (with potentially only one element).

    Parameters
    ----------
    value
    force
        If true, return empty tuple even if input object is None

    Returns
    -------
    Input data converted to a tuple.
    """
    # quick exit
    if isinstance(value, tuple):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame, Series

        has_pandas = True
    except ImportError:
        DataFrame: Any = None
        Series: Any = None

    items = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = (value,)
        elif (
            has_pandas
            and DataFrame is not None
            and Series is not None
            and isinstance(value, (DataFrame, Series))
        ):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = (value,)
        elif isinstance(value, Iterable):
            items = tuple(value)
        else:
            items = (value,)

    if force and items is None:
        items = ()

    return items


@overload
def anything_to_dict(value: Any, force: Literal[True]) -> dict[Any, Any]: ...


@overload
def anything_to_dict(
    value: Any, force: Literal[False] = False
) -> dict[Any, Any] | None: ...


@overload
def anything_to_dict(value: Any, force: bool) -> dict[Any, Any] | None: ...


def anything_to_dict(value: Any, force: bool = False) -> dict[Any, Any] | None:
    """
    Convert given object to a dictionary using common-sense rules.

    Parameters
    ----------
    value
        Anything that can be reasonably converted to a dictionary.
    force
        If true, return an empty dictionary even if no meaningful conversion is
        possible.

    Returns
    -------
    The converted dictionary, or None.
    """
    if isinstance(value, dict):
        return value

    has_pandas = False

    try:
        from pandas import DataFrame

        has_pandas = True
    except ImportError:
        DataFrame: Any = None

    items: dict[Any, Any] | None = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            items = {value: None}
        elif isinstance(value, Mapping):
            items = dict(value)
        elif has_pandas and DataFrame is not None and isinstance(value, DataFrame):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            items = dict(value.items())
        elif isinstance(value, Iterable):
            # Any iterable other than the ones covered above: create dict with all
            # values set to None
            items = dict.fromkeys(value)
        else:
            items = {value: None}

    if force and items is None:
        items = {}

    return items
