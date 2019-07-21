"""
Implement search routines used by numba-fied interpolation routines.

Author: Richard Foltyn
"""

from pydynopt.common.numba import jit, jitclass, int64


@jit
def bsearch(needle, haystack):
    """
    Return the index of the lower bound of a bracketing interval which contains
    needle, ie
        haystack[ilb] <= needle < haystack[ilb+1]

    If `needle` < `haystack`[0] then ilb = 0 is returned.
    If `needle` >= `haystack[-1]` then ilb = len(haystack) - 2 is returned.

    The functions returns negative values to indicate an error, in particular
    if the size of haystack is smaller than 2. These SHOULD NOT be used
    as array indices, even though they might be valid given numpy's indexing.

    Parameters
    ----------
    needle : float or int
    haystack : np.ndarray

    Returns
    -------
    ilb : int
        Index of lower bound of bracketing interval.
    """

    n = haystack.shape[0]

    if n <= 1:
        ilb = -1
        return ilb

    ilb = bsearch_impl(needle, haystack)

    return ilb


@jit
def bsearch_impl(needle, haystack):
    """
    Implementation for bsearch without error checking.

    Parameters
    ----------
    needle : float or int
    haystack : np.ndarray

    Returns
    -------
    ilb : int
        Index of lower bound of bracketing interval.
    """

    n = haystack.shape[0]

    ilb = 0
    iub = n - 1

    while iub > (ilb + 1):
        imid = (iub + ilb) // 2
        if haystack[imid] > needle:
            iub = imid
        else:
            ilb = imid

    return ilb
