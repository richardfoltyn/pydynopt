__author__ = 'Richard Foltyn'

from pydynopt.interpolate.fpbspl import fpbspl, my_fpbspl, my_spevl
from scipy.interpolate import splev, splrep

import numpy as np

x = np.arange(1, 5)
y = np.log(x)
(t, c, k) = splrep(x, y)

# h = fpbspl(np.array([1.5]), t, 3)

x_at = np.array([1.5, 2.5, 3.4])

xhat_fp = splev(x_at, (t, c, k))
xhat = my_spevl(x_at, t, c, k)

print(xhat_fp)
print(xhat)
