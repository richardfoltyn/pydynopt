"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""


class SerialPool:
    """
    Implements dummy Pool that can be used to run code serially without
    changing the API.

    """
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        """
        No-op method to support context managers.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        No-op method to support context managers.
        """

    def starmap(self, func, args):
        """
        Dummy serial implementation for Python's Pool.starmap() method.

        Parameters
        ----------
        func : callable
        args : iterable

        Returns
        -------
        results : list
        """
        results = []
        for arg in args:
            res = func(*arg)
            results.append(res)

        return results
