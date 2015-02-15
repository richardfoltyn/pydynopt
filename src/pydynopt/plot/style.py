from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from matplotlib.font_manager import FontProperties
from brewer2mpl import qualitative
import itertools as it


COLORS = ['#d7191c', '#2b83ba', '#1a9641', '#404040', '#ff7f00']
LSTYLES = ['-', '--', '-', '--', '-.']
ALPHAS = [.7, 0.9, 0.7, 1.0]

LEG_KWARGS = {'prop': FontProperties(family='serif'), 'framealpha': .7}
LBL_KWARGS = {'fontproperties': FontProperties(family='serif', size=12)}
TITLE_KWARGS = {'fontproperties':
                    FontProperties(family='serif', size=14, style='italic')}
SUPTITLE_KWARGS = {'fontproperties':
                    FontProperties(family='serif', size=14, style='italic',
                                   weight='semibold')}
SUBPLOT_KWARGS = {'axisbg': 'white', 'axisbelow': True}
GRID_KWARGS = {'color': 'black', 'alpha': 0.6, 'zorder': -1000}

TEXT_KWARGS = {'fontsize': 14, 'alpha': 1.0, 'backgroundcolor': 'white',
               'zorder': -5,
               'fontproperties': FontProperties(family='serif')}

DEFAULT_LINEWIDTH = 2
CELL_SIZE = 6


def default_colors(num):
    nn = max(3, num)
    return qualitative.Set1[nn].hex_colors[:num]


def default_lstyles(num):
    ls = it.cycle(LSTYLES)
    return list(next(ls) for x in range(num))


def default_alphas(num):
    alph = it.cycle(ALPHAS)
    return list(next(alph) for x in range(num))