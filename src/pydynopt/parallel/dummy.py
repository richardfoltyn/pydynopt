"""
Author: Richard Foltyn
"""


class SerialPool:
    """
    Implements dummy Pool that can be used to run code serially without
    changing the API.

    """
    def __init__(self, *args, **kwargs):
        pass

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