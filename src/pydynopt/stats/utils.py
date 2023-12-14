import numpy as np
from numpy.polynomial import chebyshev


__all__ = ["chebyshev_polynomial"]


def chebyshev_polynomial(
    x, deg: int, intercept: bool = False, return_type: str = "ndarray"
):
    """
    Compute the Vandermonde matrix for the  Chebyshev polynomial of degree `deg`
    from a given vector of data points.

    Parameters
    ----------
    x : array_like
        Variable from which to compute the Chebyshev polynomial
    deg : int
        Polynomial degree
    intercept : bool
        If true, return constant as first column.
    return_type : str
        Return type, either "ndarray" or "dataframe"

    Returns
    -------
    np.ndarray or pd.DataFrame
    """

    return_type = return_type.lower()
    if return_type not in ("ndarray", "dataframe"):
        raise ValueError("Invalid return_type argument")

    x = np.atleast_1d(x).flatten()

    xmin, xmax = np.nanmin(x), np.nanmax(x)

    x = 2 * (x - xmin) / (xmax - xmin) - 1
    # Create Pseudo-Vandermonde matrix. Each column corresponds to a Chebyshev
    # polynomial with increasing degree. First column is constant, ignore.
    vander = chebyshev.chebvander(x, deg=deg)

    if not intercept:
        vander = np.ascontiguousarray(vander[:, 1:])

    if return_type.lower() == "dataframe":
        import pandas as pd

        columns = [f"p{i + 1 - int(intercept)}" for i in range(vander.shape[1])]
        df = pd.DataFrame(vander, columns=columns)
        return df
    else:
        return vander
