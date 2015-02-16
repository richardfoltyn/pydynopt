from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from matplotlib.font_manager import FontProperties
from brewer2mpl import qualitative
import itertools as it


COLORS = ['#d7191c', '#2b83ba', '#1a9641', '#404040', '#ff7f00']

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


class AbstractStyle(object):

    def __init__(self):
        self.linewidth = 1

        self.cell_size = None
        self.dpi = 96

        self.grid = False
        self.text = {}
        self.title = {}
        self.legend = {}
        self.suptitle = {}
        self.xlabel = {}
        self.ylabel = {}
        self.subplot = {}
        self.figure = {}


class DefaultStyle(AbstractStyle):

    LINESTYLES = ['-', '--', '-', '--', '-.']
    ALPHAS = [.7, 0.9, 0.7, 1.0]

    def __init__(self):

        super(DefaultStyle, self).__init__()

        self.linewidth = 2
        self.cell_size = 6

        self.grid = GRID_KWARGS
        self.text = TEXT_KWARGS
        self.title = TITLE_KWARGS
        self.legend = LEG_KWARGS
        self.suptitle = SUPTITLE_KWARGS
        self.xlabel = LBL_KWARGS
        self.ylabel = LBL_KWARGS
        self.figure = {'tight_layout': True}

    def color_seq(self, num):
        nn = max(3, num)
        return tuple(qualitative.Set1[nn].hex_colors[:num])

    def lstyle_seq(self, num):
        ls = it.cycle(DefaultStyle.LINESTYLES)
        return tuple(next(ls) for x in range(num))

    def lwidth_seq(self, num):
        lw = it.cycle([self.linewidth])
        return tuple(next(lw) for x in range(num))

    def alpha_seq(self, num):
        alph = it.cycle(DefaultStyle.ALPHAS)
        return tuple(next(alph) for x in range(num))
