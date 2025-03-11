"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""
from typing import Optional

from matplotlib.ticker import FuncFormatter


class SuffixFormatter(FuncFormatter):

    def __init__(self, default: Optional[str] = None):
        """

        Parameters
        ----------
        default : str, optional
            Default format string if value does not match any suffix.
        """

        def _suffix_formatter(value: float, pos: int) -> str:
            """
            Format numbers with suffixes k, m, or bn depending
            on their magnitue.

            Parameters
            ----------
            value : float
                Tick value to format
            pos : int
                Tick position within list of tick labels

            Returns
            -------
            str
                Formatted tick label
            """
            suffix = ''
            if abs(value) >= 1.0e12:
                value /= int(1.0e12)
                suffix = 'tr'
            elif abs(value) >= 1.0e9:
                value /= int(1.0e9)
                suffix = 'bn'
            elif abs(value) >= 1.0e6:
                value /= int(1.0e6)
                suffix = 'm'
            elif abs(value) >= 1000:
                value /= 1000
                suffix = 'k'

            if int(value) == value:
                fmt = '.0f'
            elif not suffix and default is not None:
                fmt = default
            else:
                fmt = ''

            fmt = f'{{sgn}}{{v:{fmt}}}{suffix}'

            return fmt.format(v=abs(value), sgn='$-$' if value < 0 else '')

        super().__init__(_suffix_formatter)

