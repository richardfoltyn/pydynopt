import numpy as np
import pandas as pd

from pydynopt.stats import quantile

__all__ = ['winsorize']


def winsorize(
    data: pd.DataFrame | pd.Series | np.ndarray,
    qlb: float | None = None,
    qub: float | None = None,
    *,
    varname: str | None = None,
    weights: str | pd.Series | np.ndarray | None = None,
    inplace: bool = False,
    interpolation: str = 'linear',
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
        raise ValueError('Invalid quantile argument value')

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
        d = pd.DataFrame({'value': d, 'weight': weights})
        d = d.dropna()
        if (d['weight'] == 0.0).any():
            d = d.loc[d['weight'] > 0.0].copy()

        qntl = quantile(
            d['value'].to_numpy(),
            d['weight'].to_numpy(),
            (qlb, qub),
            interpolation=interpolation,
            **kwargs,
        )
    else:
        d = pd.Series(d)
        qntl = d.quantile([qlb, qub], interpolation=interpolation).to_numpy()  # type: ignore

    if not inplace:
        if isinstance(data, np.ndarray):
            data = np.copy(data)
        elif isinstance(data, pd.DataFrame):
            data = data[[varname]].copy(deep=True)
        else:
            data = data.copy()

    if qlb > 0.0:
        lb = qntl[0]
        if isinstance(data, pd.DataFrame):
            mask = data[varname] < lb
            data.loc[mask, varname] = lb
        elif isinstance(data, np.ndarray):
            mask = data < lb
            data[mask] = lb
        else:
            mask = data < lb
            data.loc[mask] = lb

    if qub < 1:
        ub = qntl[1]
        if isinstance(data, pd.DataFrame):
            mask = data[varname] > ub
            data.loc[mask, varname] = ub
        elif isinstance(data, np.ndarray):
            mask = data > ub
            data[mask] = ub
        else:
            mask = data > ub
            data.loc[mask] = ub

    return data
