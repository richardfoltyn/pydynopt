

import numpy as np


class OptimResult:
    def __init__(self):
        self.x = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        self.flag = ""


class FunctionWrapper:
    def __init__(self, func=None, jac=None, eps=1.0e-8):
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
        self.jac = jac
        self.eps = float(eps)

    @property
    def returns_jac(self):
        x = isinstance(self.jac, bool)
        if x:
            x = self.jac
        return x

    def __call__(self, x, *args, **kwargs):
        self.ncalls += 1
        return self.func(x, *args, **kwargs)

    def eval_func(self, x, *args, **kwargs):
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
        if self.returns_jac:
            fx, fpx = self(x, *args, **kwargs)
        else:
            fx = self(x, *args, **kwargs)
        return fx

    def eval_jac(self, x, *args, **kwargs):
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

        if self.returns_jac:
            # Callable computes Jacobian along with function value; discard
            # function value.
            fx, fpx = self(x, *args, **kwargs)
        elif callable(self.jac):
            # Separate callable provided to compute Jacobian without computing
            # function value.
            fpx = self.jac(x, *args, **kwargs)
            self.ncalls += 1
            return fpx
        else:
            # No Jacobian provided:
            # need to forward-differentiate
            fx = self(x, *args, **kwargs)
            fpx = nderiv(self, x, fx, eps=self.eps, *args, **kwargs)

        return fpx

    def eval_func_jac(self, x, *args, **kwargs):
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

        if self.returns_jac:
            fx, jac = self(x, *args, **kwargs)
        elif callable(self.jac):
            fx = self(x, *args, **kwargs)
            jac = self.jac(x, *args, **kwargs)
            self.ncalls += 1
        else:
            fx = self(x, *args, **kwargs)
            jac = nderiv(self, x, fx, self.eps, *args, **kwargs)

        return fx, jac


def nderiv(func, x, fx=None, eps=1.0e-8, *args, **kwargs):
    """
    Numerically forward-differentiate function and given point.

    Parameters
    ----------
    func : callable
    x : float or array_like
    fx :
    eps : float
    args
    kwargs

    Returns
    -------

    """

    eps = float(eps)

    if fx is None:
        fx = func(x, *args, **kwargs)

    if np.isscalar(x):
        xh = x + eps
        fxh = func(xh, *args, **kwargs)
        fpx = (fxh - fx) / eps
    else:
        xx = np.atleast_1d(x)
        fpx = np.zeros_like(xx)
        for i, xi in enumerate(xx):
            xxi = np.copy(xx)
            xxi[i] += eps
            fxi = func(xxi, *args, **kwargs)
            fpx[i] = (fxi - fx) / eps

    return fpx
