"""
Utility functions for working with list-like collections.

Author: Richard Foltyn
"""

__all__ = ['anything_to_list', 'anything_to_tuple']


def anything_to_list(value, force=False):
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

    from collections import Iterable

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


def anything_to_tuple(value, force=False):
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

    from collections import Iterable

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
