"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from pydynopt.numba import register_jitable, JIT_OPTIONS
import numpy as np


@register_jitable(**JIT_OPTIONS)
def chunk_sizes(nthreads, ntasks):
    """
    Split a given number of tasks (almost) uniformly across the given number
    of threads. This function takes care of the case when it is not possible
    to assign the same number of tasks to each thread.

    Parameters
    ----------
    nthreads : int
        Number of threads
    ntasks : int
        Number of task to be divided across threads

    Returns
    -------
    chunks : np.ndarray
        Array containing the number of task for each thread
    istart : np.ndarray
        Start index of the chunk assigned to each thread
    iend : np.ndarray
        (non-inclusive) end index of the chunk assigned to each thread
    """

    nthreads = int(nthreads)
    ntasks = int(ntasks)

    if nthreads < 1:
        raise ValueError('Invalid number of threads')

    base = ntasks // nthreads
    rest = ntasks % nthreads

    chunks = np.ones((nthreads, ), dtype=np.int64) * base
    chunks[:rest] += 1

    assert np.sum(chunks) == ntasks

    iend = np.cumsum(chunks)
    istart = np.empty_like(iend)
    istart[0] = 0
    istart[1:] = iend[:-1]

    return chunks, istart, iend
