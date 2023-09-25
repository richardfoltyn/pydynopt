"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Iterable, Mapping

from typing import Any, Optional

import pandas as pd
import numpy as np

from pydynopt.utils import anything_to_list

__all__ = ["anything_to_dataframe"]


def anything_to_dataframe(
    data: Any, names: Optional[str | Iterable[str]] = None, copy: bool = False
) -> pd.DataFrame:
    """
    Create DataFrame from (almost) any type of data.

    Parameters
    ----------
    data
    names
    copy

    Returns
    -------
    pd.DataFrame
    """

    names = anything_to_list(names)

    if isinstance(data, pd.DataFrame):
        df = data
        if copy:
            df = df.copy(deep=True)
    elif isinstance(data, pd.Series):
        name = names[0] if names else None
        df = pd.DataFrame(data, columns=[name], copy=copy)
    elif isinstance(data, Mapping):
        # Initialization from dict always copies data
        dct = {k: np.asarray(v) for k, v in data.items()}
        df = pd.DataFrame(dct)
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        df = pd.DataFrame(data, columns=names, copy=copy)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        name = names[0] if names else None
        df = pd.DataFrame(data[:, None], columns=[name], copy=copy)
    elif np.isscalar(data) and len(names) == 1:
        df = pd.DataFrame([data], columns=names)
    elif names and len(names) == 1:
        dct = {names[0]: data}
        df = pd.DataFrame(dct)
    elif names:
        # both names and data are iterables of column-specific data
        dct = {k: np.asarray(v) for k, v in zip(names, data)}
        df = pd.DataFrame(dct)
    else:
        df = pd.DataFrame(data, copy=copy)

    return df
