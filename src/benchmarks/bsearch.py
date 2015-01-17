__author__ = 'Richard Foltyn'

from pydynopt.benchmark import repeat

lb, ub, count = 0, 10, 101

setup = r'''
import numpy as np
from cython import double
from pydynopt.utils import _bsearch
stack = np.linspace({lb:.1f}, {ub:.1f}, {count:d})
needle = {{nd:.2f}}
'''.format(lb=lb, ub=ub, count=count)

stmt = r'_bsearch(stack, needle)'

needles = (0.1, 25.01, 50.1, 99.9)
dx = (ub-lb)/count

for needle in needles:
    print('\nTesting with needle={nd:.2f}, haystack {lb:.1f}:{ub:.1f}:{'
          'dx:.2f}'.format(nd=needle, lb=lb, ub=ub, dx=dx))

    setup_n = setup.format(nd=needle)

    r = repeat(stmt, setup_n)
