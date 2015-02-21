from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

import numpy as np


def chunk_sizes(nthreads, ntasks):

    nthreads = int(nthreads)
    ntasks = int(ntasks)

    if nthreads < 1:
        raise ValueError('Invalid number of threads: {:d}'.format(nthreads))

    base = ntasks // nthreads
    rest = ntasks % nthreads

    chunks = np.ones((nthreads, ), dtype=np.uint32) * base
    chunks[:rest] += 1

    assert np.sum(chunks) == ntasks

    return list(chunks)