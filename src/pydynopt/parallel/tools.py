from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

import numpy as np


def chunk_sizes(nthreads, ntasks, bounds=False):

    nthreads = int(nthreads)
    ntasks = int(ntasks)

    if nthreads < 1:
        raise ValueError('Invalid number of threads: {:d}'.format(nthreads))

    base = ntasks // nthreads
    rest = ntasks % nthreads

    chunks = np.ones((nthreads, ), dtype=np.uint) * base
    chunks[:rest] += 1

    assert np.sum(chunks) == ntasks

    if not bounds:
        return chunks

    iend = np.cumsum(chunks)
    istart = np.hstack((np.array(0, dtype=np.uint), iend[:-1]))

    return chunks, istart, iend