"""
Helpers for compact axis tick labels with magnitude suffixes.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from matplotlib.ticker import FuncFormatter


class SuffixFormatter(FuncFormatter):
    """Format tick values using compact suffixes (`k`, `m`, `bn`, `tr`)."""

    def __init__(self, default: str | None = None) -> None:
        """
        Create a formatter that shortens large magnitudes.

        Parameters
        ----------
        default : str, optional
            Format specifier used for values without suffix (for example,
            `.2f`). If omitted, matplotlib's default formatting is used.
        """
        self.default = default

        def _suffix_formatter(value: float, pos: int) -> str:
            """
            Format numbers with suffixes based on their magnitude.

            Parameters
            ----------
            value : float
                Tick value to format.
            pos : int
                Tick position in matplotlib's sequence.

            Returns
            -------
            str
                Formatted tick label.
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

    def __repr__(self) -> str:
        """
        Return a string representation of the formatter.

        Returns
        -------
        str
            Developer-friendly representation of the SuffixFormatter.
        """
        return f'{type(self).__name__}(default={self.default!r})'
