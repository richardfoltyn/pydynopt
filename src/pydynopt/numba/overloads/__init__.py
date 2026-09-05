"""Expose Numba-compatible NumPy overloads.

- Re-export indexing overloads.
- Re-export additional NumPy function overloads.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from .indexing import *
from .numpy import *
