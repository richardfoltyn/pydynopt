"""
Exception classes for numerical and optimization routines.

- Convergence failure errors

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

__all__ = [
    'ConvergenceError',
]


class ConvergenceError(Exception):
    """
    Exception raised when an iterative algorithm fails to converge.

    Parameters
    ----------
    iterations
        Number of iterations performed before convergence failure.
    tol
        Achieved tolerance or error metric at termination.
    """

    def __init__(self, iterations: int, tol: float) -> None:
        super().__init__(iterations, tol)
        self._iters = iterations
        self._tol = tol

    @property
    def iterations(self) -> int:
        """
        Number of iterations performed.

        Returns
        -------
        Number of iterations performed before convergence failure.
        """
        return self._iters

    @property
    def tol(self) -> float:
        """
        Achieved tolerance or error metric.

        Returns
        -------
        Achieved tolerance or error metric at termination.
        """
        return self._tol
