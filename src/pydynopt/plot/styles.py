from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from matplotlib.font_manager import FontProperties
from brewer2mpl import qualitative
import itertools as it
import numpy as np


# COLORS = ['#d7191c', '#2b83ba', '#1a9641', '#404040', '#ff7f00']

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


class Colors(object):
    def __init__(self, colors=None):
        self.colors = colors
        self.cache = colors

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            if not self.colors:
                nn = max(4, item + 1)
                self.cache = tuple(qualitative.Set1[nn].hex_colors)
            else:
                col = it.cycle(self.colors)
                self.cache = tuple(next(col) for x in range(item + 1))
        return self.cache[item]


class LineStyle(object):
    def __init__(self, linestyles):
        self.linestyles = linestyles
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            ls = it.cycle(self.linestyles)
            self.cache = tuple(next(ls) for x in range(item + 1))
        return self.cache[item]


class LineWidth(object):
    def __init__(self, linewidths):
        self.linewidths = linewidths
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            lw = it.cycle(self.linewidths)
            self.cache = tuple(next(lw) for x in range(item + 1))
        return self.cache[item]


class Transparency(object):
    def __init__(self, alphas):
        self.alphas = alphas
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            alph = it.cycle(self.alphas)
            self.cache = tuple(next(alph) for x in range(item + 1))
        return self.cache[item]


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

    LINESTYLES = ['-', '--', '-', '--']
    ALPHAS = [.7, 0.9, 0.7, 1.0]
    LINEWIDTH = [2]

    def __init__(self):

        super(DefaultStyle, self).__init__()

        self.cell_size = 6

        self.grid = GRID_KWARGS
        self.text = TEXT_KWARGS
        self.title = TITLE_KWARGS
        self.legend = LEG_KWARGS
        self.suptitle = SUPTITLE_KWARGS
        self.xlabel = LBL_KWARGS
        self.ylabel = LBL_KWARGS
        self.figure = {'tight_layout': True}

        self._color = None
        self._linewidth = None
        self._linestyle = None
        self._alpha = None

    @property
    def color(self):
        if self._color is None:
            self._color = Colors()
        return self._color

    @color.setter
    def color(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._color = Colors(colors=value)

    @property
    def linewidth(self):
        if self._linewidth is None:
            self._linewidth = LineWidth(DefaultStyle.LINEWIDTH)
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._linewidth = LineWidth(value)

    @property
    def linestyle(self):
        if self._linestyle is None:
            self._linestyle = LineStyle(DefaultStyle.LINESTYLES)
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._linestyle = LineStyle(value)

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = Transparency(DefaultStyle.ALPHAS)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._alpha = Transparency(value)


class Presentation(DefaultStyle):

    def __init__(self):

        super().__init__()

        self.cell_size = 5.0
        # Red, Black, Green, Gray

        colors = ('#4daf4a', '#111111', '#e31a1c', '#4d4d4d')
        self.color = colors
        self.linestyle = ('-', '--', '-.', ':')
        self.linewidth = (2.0, 2.0, 2.0, 2.0)
        self.alpha = (.8, .8, .8, 1.0)


