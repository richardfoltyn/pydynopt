

import numpy as np

from pydynopt.numba import overload


class OptimResult:
    def __init__(self):
        self.x = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        # Some status message or flag of type string
        self.flag = ""

    def __repr__(self):
        """
        Return string representation of result object.

        Returns
        -------
        s : str
        """

        fmt_float = '{:>20s}: {:g}'
        fmt_int = '{:>20s}: {:d}'
        fmt_default = '{:>20s}: {}'

        attrs = ['converged', 'flag', 'function_calls', 'iterations', 'x']

        tokens = []

        for attr in attrs:
            if not hasattr(self, attr):
                continue

            value = getattr(self, attr)
            if isinstance(value, float):
                s = fmt_float.format(attr, value)
            elif isinstance(value, int):
                s = fmt_int.format(attr, value)
            elif isinstance(value, np.ndarray):
                if np.isscalar(value):
                    s = fmt_float.format(attr, value)
                else:
                    x = np.atleast_1d(value)
                    s = ', '.join('{:g}'.format(xi) for xi in x)
                    s = '[' + s + ']'
            else:
                s = fmt_default.format(attr, value)

            tokens.append(s)

        s = '\n'.join(tokens)
        return s


class FunctionWrapper:
    def __init__(self, func, eps=1.0e-8):
        """

        Parameters
        ----------
        func
        jac
        eps : float
            Step sized used for forward-differentiation when Jacobian
            needs to be computed numerically
        """
        self.ncalls = 0
        self.func = func
        self.eps = eps

    def eval_func(self, x, *args):
        """
        Return function evaluated at given point

        Parameters
        ----------
        x
        args
        kwargs

        Returns
        -------

        """
        fx = self.func(x, *args)
        self.ncalls += 1

        return fx

    def eval_jac(self, x, *args):
        """
        Evaluate the Jacobian at a given point.

        Parameters
        ----------
        x : float or array_like
        args
        kwargs

        Returns
        -------

        """

        # No Jacobian provided:
        # need to forward-differentiate
        fx = self.func(x, *args)
        jac = nderiv(self.func, x, fx, eps=self.eps, *args)
        self.ncalls += (1 + np.atleast_1d(x).shape[0])

        return jac

    def eval_func_jac(self, x, *args):
        """self
        Evaluate the function and its Jacobian

        Parameters
        ----------
        x : float or array_like
        args
        kwargs

        Returns
        -------
        fx : float or np.ndarray
        jac : float or np.ndarray
        """

        fx = self.func(x, *args)
        jac = nderiv(self.func, x, fx, self.eps, *args)
        self.ncalls += (1 + np.atleast_1d(x).shape[0])

        return fx, jac


class FunctionJacWrapper:
    def __init__(self, func):
        """

        Parameters
        ----------
        func
        jac
        eps : float
            Step sized used for forward-differentiation when Jacobian
            needs to be computed numerically
        """
        self.ncalls = 0
        self.func = func

    def eval_func(self, x, *args):
        """
        Return function evaluated at given point

        Parameters
        ----------
        x
        args
        kwargs

        Returns
        -------

        """
        fx, fpx = self.func(x, *args)
        return fx

    def eval_jac(self, x, *args):
        """
        Evaluate the Jacobian at a given point.

        Parameters
        ----------
        x : float or array_like
        args
        kwargs

        Returns
        -------

        """

        fx, fpx = self.func(x, *args)
        return fpx

    def eval_func_jac(self, x, *args):
        """
        Evaluate the function and its Jacobian

        Parameters
        ----------
        x : float or array_like
        args
        kwargs

        Returns
        -------
        fx : float or np.ndarray
        jac : float or np.ndarray
        """

        fx, fpx = self.func(x, *args)

        return fx, fpx


def _nderiv_array(func, x, fx=np.nan, eps=1.0e-8, *args):
    """
    Numerically forward-differentiate function and given point.

    Parameters
    ----------
    func : callable
    x : np.ndarray
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : np.ndarray
    """

    if np.isnan(fx):
        fx = func(x, *args)

    fpx = np.zeros_like(x)
    for i, xi in enumerate(x):
        xxi = np.copy(x)
        xxi[i] += eps
        fxi = func(xxi, *args)
        fpx[i] = (fxi - fx) / eps

    return fpx


def _nderiv_scalar(func, x, fx=np.nan, eps=1.0e-8, *args):
    """

    Parameters
    ----------
    func : callable
    x : float
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : float
    """

    x1d = np.array([x])
    fpx1d = nderiv(func, x1d, fx, eps, *args)

    fpx = fpx1d[0]

    return fpx


def nderiv(func, x, fx=np.nan, eps=1.0e-8, *args):
    """
    Numerically forward-differentiate function and given point.

    Parameters
    ----------
    func : callable
    x : float or array_like
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : float or np.ndaray
    """

    eps = float(eps)
    x1d = np.atleast_1d(x)

    fpx = _nderiv_array(func, x1d, fx, eps, *args)

    if np.isscalar(x):
        fpx = fpx.item()

    return fpx


@overload(nderiv, jit_options={'parallel': False, 'nogil': True})
def nderiv_generic(func, x, fx=np.nan, eps=1.0e-8, *args):

    from numba.types import Number
    from numba.types.npytypes import Array

    f = None
    if isinstance(x, Number):
        f = _nderiv_scalar
    elif isinstance(x, Array):
        f = _nderiv_array

    return f

