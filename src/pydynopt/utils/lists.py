"""
Utility functions for working with lists.

Author: Richard Foltyn
"""


def anything_to_list(value, force_list=False):
    """
    Covert a given value to a list (with potentially only one element).

    Parameters
    ----------
    value : object
    force_list : bool
        If true, return empty list even if input object is None

    Returns
    -------
    list or None
        Input data converted to a list
    """

    from collections import Iterable

    try:
        from pandas import DataFrame
    except ImportError:
        DataFrame = None

    lst = None
    if value is not None:
        if isinstance(value, str):
            # Treat string separately to prevent it being split into separate
            # characters, as a string is also Iterable
            lst = [value]
        elif DataFrame is not None and isinstance(value, DataFrame):
            # Treat pandas DataFrame separately, as these are iterable,
            # but iteration is over column index, which is not what we want.
            lst = [value]
        elif isinstance(value, Iterable):
            lst = list()
            lst.extend(value)
        else:
            lst = [value]

    if force_list and lst is None:
        lst = []

    return lst
