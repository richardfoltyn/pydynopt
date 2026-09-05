"""Expose optimization result containers and scalar root finders.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from ._zeros_scipy import RootResult
from .common import OptimResult
from .zeros import brentq, newton_bisect

__all__ = ['OptimResult', 'RootResult', 'brentq', 'newton_bisect']
