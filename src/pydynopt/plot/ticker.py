"""
Author: Richard Foltyn
"""


from matplotlib.ticker import FuncFormatter


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
    if value >= 1.0e9:
        value /= int(1.0e9)
        suffix = 'bn'
    elif value >= 1.0e6:
        value /= int(1.0e6)
        suffix = 'm'
    elif value >= 1000:
        value /= 1000
        suffix = 'k'

    if int(value) == value:
        fmt = '.0f'
    else:
        fmt = ''

    fmt = f'{{v:{fmt}}}{suffix}'

    return fmt.format(v=value)


SuffixFormater = FuncFormatter(_suffix_formatter)

