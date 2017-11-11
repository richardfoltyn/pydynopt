from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from matplotlib.font_manager import FontProperties
import itertools as it
import numpy as np


class Colors(object):
    def __init__(self, colors=None):
        self.colors = colors
        self.cache = colors

    def __getitem__(self, item):
        if item >= len(self.cache):
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


class Marker(object):
    def __init__(self, markers):
        self.markers = markers
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            ms = it.cycle(self.markers)
            self.cache = tuple(next(ms) for x in range(item + 1))
        return self.cache[item]


class ConstFillProperty:
    def __init__(self, const, values=None):
        self.values = values
        self.const = const

    def __getitem__(self, item):
        if self.values is None or item >= len(self.values):
            return self.const
        else:
            return self.values[item]


class Transparency(object):
    def __init__(self, alphas):
        self.alphas = alphas
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            alph = it.cycle(self.alphas)
            self.cache = tuple(next(alph) for x in range(item + 1))
        return self.cache[item]


class PlotStyleDict(object):
    def __init__(self, style):
        self.style = style

    def __getitem__(self, item):

        keys = {'color', 'lw', 'ls', 'alpha', 'marker', 'mec', 'markersize',
                'zorder'}
        res = dict()

        for k in keys:
            res[k] = getattr(self.style, k)[item]

        return res


class AbstractStyle(object):

    DEFAULT_KWARGS = {}

    def __init__(self):
        self.linewidth = 1

        self.cell_size = None
        self.dpi = 96

        self.figure = AbstractStyle.DEFAULT_KWARGS

    @property
    def legend(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def grid(self):
        return False

    @property
    def title(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def suptitle(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def xlabel(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def ylabel(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def subplot(self):
        return AbstractStyle.DEFAULT_KWARGS

    @property
    def text(self):
        return AbstractStyle.DEFAULT_KWARGS


class DefaultStyle(AbstractStyle):

    LEG_FONTPROP_KWARGS = {
        'family': 'serif'
    }

    LBL_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 12
    }

    TITLE_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 14,
        'style': 'italic'
    }

    SUBTITLE_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 14,
        'style': 'italic',
        'weight': 'semibold'
    }

    TEXT_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 14
    }

    # Keyword arguments (other than font properties) for various objects
    LEG_KWARGS = {'framealpha': .7}

    LBL_KWARGS = {}
    TITLE_KWARGS = {}
    SUPTITLE_KWARGS = {}

    SUBPLOT_KWARGS = {
        'axisbg': 'white',
        'axisbelow': True
    }

    GRID_KWARGS = {
        'color': 'black',
        'alpha': 0.7,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.5
    }

    TEXT_KWARGS = {
        'alpha': 1.0,
        'zorder': 500
    }

    LINESTYLES = ['-', '--', '-', '--']
    ALPHAS = [.7, 0.9, 0.7, 1.0]
    MARKERS = [None]
    LINEWIDTH = [2]
    MARKERSIZE = 5

    COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#f781bf']

    def __init__(self):

        super(DefaultStyle, self).__init__()

        self.cell_size = 6

        self.figure = {'tight_layout': True}

        self._color = None
        self._linewidth = None
        self._linestyle = None
        self._alpha = None
        self._marker = None
        self._markersize = None
        self._mec = Colors(('white', ))
        self._zorder = None

        self._plot_all = PlotStyleDict(self)

    @property
    def legend(self):
        cls = self.__class__
        fp = FontProperties(**cls.LBL_FONTPROP_KWARGS)
        kwargs = cls.LEG_KWARGS.copy()
        # Add font properties
        kwargs.update({'prop': fp})
        return kwargs

    @property
    def grid(self):
        cls = self.__class__
        kwargs = cls.GRID_KWARGS.copy()
        return kwargs

    @property
    def text(self):
        cls = self.__class__
        fp = FontProperties(**cls.TEXT_FONTPROP_KWARGS)
        kwargs = cls.TEXT_KWARGS.copy()
        kwargs.update({'fontproperties': fp})
        return kwargs

    @property
    def title(self):
        cls = self.__class__
        fp = FontProperties(**cls.TITLE_FONTPROP_KWARGS)
        kwargs = cls.TITLE_KWARGS.copy()
        kwargs.update({'fontproperties': fp})
        return kwargs

    @property
    def suptitle(self):
        cls = self.__class__
        fp = FontProperties(**cls.SUBTITLE_FONTPROP_KWARGS)
        kwargs = cls.SUPTITLE_KWARGS.copy()
        kwargs.update({'fontproperties': fp})
        return kwargs

    @property
    def xlabel(self):
        cls = self.__class__
        fp = FontProperties(**cls.LBL_FONTPROP_KWARGS)
        kwargs = cls.LBL_KWARGS.copy()
        kwargs.update({'fontproperties': fp})
        return kwargs

    @property
    def ylabel(self):
        cls = self.__class__
        fp = FontProperties(**cls.LBL_FONTPROP_KWARGS)
        kwargs = cls.LBL_KWARGS.copy()
        kwargs.update({'fontproperties': fp})
        return kwargs

    @property
    def color(self):
        if self._color is None:
            self._color = Colors(self.__class__.COLORS)
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
    def lw(self):
        return self.linewidth

    @property
    def linestyle(self):
        if self._linestyle is None:
            self._linestyle = LineStyle(DefaultStyle.LINESTYLES)
        return self._linestyle

    @property
    def ls(self):
        return self.linestyle

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

    @property
    def marker(self):
        if self._marker is None:
            self._marker = Marker(DefaultStyle.MARKERS)
        return self._marker

    @marker.setter
    def marker(self, value):
        if np.isscalar(value):
            value = (value,)
        else:
            value = tuple(value)
        self._marker = Marker(value)

    @property
    def markersize(self):
        if self._markersize is None:
            self._markersize = ConstFillProperty(const=DefaultStyle.MARKERSIZE)
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._markersize = ConstFillProperty(DefaultStyle.MARKERSIZE, value)

    @property
    def mec(self):
        if self._mec is None:
            self._mec = Colors(('white', ))
        return self._mec

    @mec.setter
    def mec(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._mec = ConstFillProperty('none', value)

    @property
    def zorder(self):
        if self._zorder is None:
            self._zorder = ConstFillProperty(const=100)
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        if np.isscalar(value):
            value = (value, )
        else:
            value = tuple(value)
        self._zorder = ConstFillProperty(100, value)

    @property
    def plot_kwargs(self):
        return self._plot_all


class Presentation(DefaultStyle):

    def __init__(self):

        super().__init__()

        self.cell_size = 5.0
        # Green, Black/violet, Red, Gray

        colors = ('#4daf4a', '#56124E', '#e31a1c', '#000000', '#D76B00')
        self.color = colors
        self.linestyle = ('-', '-', '-', '--', '-')
        self.linewidth = (2.0, 2.0, 2.0, 2.0, 2.0)
        self.alpha = (.8, .8, .8, .7, .8)
        self.marker = (None, 'p', 'o', None, 'd')
        self._mec = Colors((None, 'White', 'White', None, 'White'))


