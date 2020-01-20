"""
Implement search routines used by numba-fied interpolation routines.

Author: Richard Foltyn
"""

from pydynopt.numba import jit, jitclass, int64

__all__ = ['bsearch', 'bsearch_impl']


JIT_OPTIONS = {'nopython': True, 'nogil': True, 'parallel': False}


@jit(**JIT_OPTIONS)
def bsearch(needle, haystack, ilb=0):
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
    ilb : int
        Optional (cached) index of lower bound of bracketing interval to be
        used as initial value.

    Returns
    -------
    ilb : int
        Index of lower bound of bracketing interval.
    """

    n = haystack.shape[0]

    if n <= 1:
        ilb = -1
        return ilb

    ilb = max(0, min(ilb, n-2))
    ilb = bsearch_impl(needle, haystack, ilb)

    return ilb


@jit(inline='always', **JIT_OPTIONS)
def bsearch_impl(needle, haystack, ilb=0):
    """

    Parameters
    ----------
    needle :
    haystack : np.ndarray
    ilb : int
        Cached value of index of lower bound of bracketing interval.

    Returns
    -------
    ilb : int
        Index of lower bound of bracketing interval.
    """

    n = haystack.shape[0]
    iub = n - 1

    if haystack[ilb] <= needle:
        if haystack[ilb+1] > needle:
            return ilb
        elif ilb == (n - 2):
            return ilb
    else:
        ilb, iub = 0, ilb

    while iub > (ilb + 1):
        imid = (iub + ilb)//2
        if haystack[imid] > needle:
            iub = imid
        else:
            ilb = imid

    return ilb
