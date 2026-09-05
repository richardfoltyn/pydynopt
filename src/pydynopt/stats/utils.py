"""Provide statistical utility functions.

- Construct Chebyshev polynomial design matrices.
"""

from typing import Any, Literal, overload

import numpy as np
from numpy.polynomial import chebyshev
from numpy.typing import ArrayLike, NDArray
import pandas as pd

__all__ = ['chebyshev_polynomial']


@overload
def chebyshev_polynomial(
    x: ArrayLike,
    deg: int,
    intercept: bool = False,
    return_type: Literal['ndarray'] = 'ndarray',
) -> NDArray[Any]: ...


@overload
def chebyshev_polynomial(
    x: ArrayLike,
    deg: int,
    intercept: bool = False,
    return_type: Literal['dataframe'] = 'dataframe',
) -> pd.DataFrame: ...


@overload
def chebyshev_polynomial(
    x: ArrayLike,
    deg: int,
    intercept: bool = False,
    return_type: str = 'ndarray',
) -> NDArray[Any] | pd.DataFrame: ...


def chebyshev_polynomial(
    x: ArrayLike,
    deg: int,
    intercept: bool = False,
    return_type: str = 'ndarray',
) -> NDArray[Any] | pd.DataFrame:
    """Compute a Chebyshev pseudo-Vandermonde matrix.

    Parameters
    ----------
    x
        Values at which to evaluate the Chebyshev polynomials.
    deg
        Polynomial degree.
    intercept
        If true, include the constant polynomial as the first column.
    return_type
        Return either an ``ndarray`` or a ``dataframe``.

    Returns
    -------
    Chebyshev pseudo-Vandermonde matrix in the requested format.
    """
    return_type = return_type.lower()
    if return_type not in ('ndarray', 'dataframe'):
        raise ValueError('Invalid return_type argument')

    x_arr = np.atleast_1d(x).flatten()
    xmin, xmax = np.nanmin(x_arr), np.nanmax(x_arr)

    x_scaled = 2 * (x_arr - xmin) / (xmax - xmin) - 1
    # Create the pseudo-Vandermonde matrix. Columns correspond to Chebyshev
    # polynomials of increasing degree; the first column is constant.
    vander = chebyshev.chebvander(x_scaled, deg=deg)

    if not intercept:
        vander = np.ascontiguousarray(vander[:, 1:])

    if return_type == 'dataframe':
        columns = [f'p{i + 1 - int(intercept)}' for i in range(vander.shape[1])]
        df_vander = pd.DataFrame(vander, columns=columns)
        return df_vander

    return vander
