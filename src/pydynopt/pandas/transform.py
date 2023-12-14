from typing import Optional

import numpy as np
import pandas as pd

from pydynopt.stats import quantile

__all__ = ['winsorize']


def winsorize(
    data: pd.DataFrame | pd.Series | np.ndarray,
    qlb: Optional[float] = None,
    qub: Optional[float] = None,
    *,
    varname: Optional[str] = None,
    weights: Optional[str | pd.Series | np.ndarray] = None,
    inplace: bool = False,
    interpolation: str = "linear",
    **kwargs,
) -> pd.DataFrame | pd.Series | np.ndarray:
    """
    Winsorize data at given lower and upper quantiles.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series or np.ndarray
        Data to winsorize
    qlb : float, optional
        Quantile rank of lower bound
    qub : float, optional
        Quantile rank of upper bound
    varname : str, optional
        Column name to winsorize (only used if `data` is a DataFrame)
    weights : str or pd.Series or np.ndarray, optional
        Sample weights, specified either as column name in `data` or as an array.
    inplace : bool
        Winsorize outliers in place
    interpolation : str
        Interpolation method used to compute quantiles.
    kwargs
        Keyword arguments passed to pydynopt.stats.quantile()

    Returns
    -------

    """
    if any(q is not None and not 0 < q < 1 for q in (qlb, qub)):
        raise ValueError(f"Invalid quantile argument value")

    has_weights = weights is not None

    if isinstance(data, pd.DataFrame):
        d = data[varname]
    else:
        d = data

    if isinstance(weights, str) and isinstance(data, pd.DataFrame):
        weights = data[weights]

    qlb = qlb if qlb is not None else 0.0
    qub = qub if qub is not None else 1.0

    if has_weights:
        d = pd.DataFrame({"value": d, "weight": weights})
        d = d.dropna()
        if (d["weight"] == 0.0).any():
            d = d.loc[d["weight"] > 0.0].copy()

        qntl = quantile(
            d["value"].to_numpy(),
            d["weight"].to_numpy(),
            (qlb, qub),
            interpolation=interpolation,
            **kwargs,
        )
    else:
        d = pd.Series(d)
        qntl = (
            d.quantile((qlb, qub), interpolation=interpolation).to_numpy()
        )

    if not inplace:
        if isinstance(data, np.ndarray):
            data = np.copy(data)
        elif isinstance(data, pd.DataFrame):
            data = data[[varname]].copy(deep=True)
        else:
            data = data.copy()

    if qlb > 0.0:
        lb = qntl[0]
        mask = data[varname] < lb
        if isinstance(data, np.ndarray):
            data[np.asarray(mask)] = lb
        elif isinstance(data, pd.DataFrame):
            data.loc[mask, varname] = lb
        else:
            data.loc[mask] = lb

    if qub < 1:
        ub = qntl[1]
        mask = data[varname] > ub
        if isinstance(data, np.ndarray):
            data[np.asarray(mask)] = ub
        elif isinstance(data, pd.DataFrame):
            data.loc[mask, varname] = ub
        else:
            data.loc[mask] = ub

    return data
