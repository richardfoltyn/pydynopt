"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import collections.abc
from collections.abc import Mapping
from copy import deepcopy
from typing import Optional, Union

import matplotlib
from matplotlib.font_manager import FontProperties
import itertools as it
import numpy as np

import copy

from pydynopt.utils import anything_to_tuple

import enum


class FigureLayout(enum.Enum):
    DEFAULT = 1
    TIGHT_LAYOUT = 2
    CONSTRAINED_LAYOUT = 3


_FIGURE_LAYOUT_MAP = {
    'constrained_layout': FigureLayout.CONSTRAINED_LAYOUT,
    'tight_layout': FigureLayout.TIGHT_LAYOUT
}


def _to_tuple(value):
    """
    Return a tuple created from the given value. If `value` is None,
    return a tuple containing only None.

    Parameters
    ----------
    value : object

    Returns
    -------
    builtins.tuple
    """
    value = anything_to_tuple(value)
    if value is None:
        value = (None, )

    return value


class Colors(object):
    def __init__(self, colors=None):
        self.colors = copy.copy(colors)
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            col = it.cycle(self.colors)
            self.cache = tuple(next(col) for x in range(item + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict={}):
        obj = Colors(self.colors)
        return obj


class LineStyle(object):
    def __init__(self, linestyles=None):
        self.linestyles = copy.copy(linestyles)
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            ls = it.cycle(self.linestyles)
            self.cache = tuple(next(ls) for x in range(item + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict={}):
        obj = LineStyle(self.linestyles)
        return obj


class LineWidth(object):
    def __init__(self, linewidths):
        self.linewidths = copy.copy(linewidths)
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            lw = it.cycle(self.linewidths)
            self.cache = tuple(next(lw) for x in range(item + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict={}):
        obj = LineWidth(self.linewidths)
        return obj


class Marker(object):
    def __init__(self, markers):
        self.markers = copy.copy(markers)
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            ms = it.cycle(self.markers)
            self.cache = tuple(next(ms) for x in range(item + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict={}):
        obj = Marker(self.markers)
        return obj


class ConstFillProperty:
    def __init__(self, const, values=None):
        self.values = copy.copy(values)
        self.const = const

    def __getitem__(self, item):
        if self.values is None or item >= len(self.values):
            return self.const
        else:
            return self.values[item]

    def __deepcopy__(self, memodict={}):
        obj = ConstFillProperty(self.const, self.values)
        return obj


class Transparency(object):
    def __init__(self, alphas):
        self.alphas = copy.copy(alphas)
        self.cache = None

    def __getitem__(self, item):
        if not self.cache or item >= len(self.cache):
            alph = it.cycle(self.alphas)
            self.cache = tuple(next(alph) for x in range(item + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict={}):
        obj = Transparency(self.alphas)
        return obj


class PlotStyleDict(object):
    def __init__(self, style):
        self.style = style

    def __getitem__(self, item):

        keys = {'color', 'lw', 'ls', 'alpha', 'marker', 'mec', 'mfc', 'mew',
                'markersize', 'markevery',  'zorder'}
        res = dict()

        for k in keys:
            res[k] = getattr(self.style, k)[item]

        return res


class StyleAttrMapping:
    """
    Wrapper type which should be returned by style properties
    that return several key/value pairs of arguments to matplotlib
    functions.

    The class implements [] such that these key/value pairs can be
    retrieved for a sequence of objects to be plotted.
    """

    def __init__(self, style, mapping):
        self._style = style
        self._mapping = mapping

    def __getitem__(self, item):
        """
        Return the style defined by key/value pairs at a given index.

        Parameters
        ----------
        item : int

        Returns
        -------
        dict
        """

        result = dict()
        for key, attr in self._mapping.items():
            attr = attr if attr is not None else key
            # Handle lists of attributes
            if isinstance(attr, (list, tuple)):
                dct = dict()
                for subattr in attr:
                    value = getattr(self._style, subattr)
                    try:
                        dct[subattr] = value[item]
                    except TypeError:
                        # Constant properties that should not vary across
                        # plotted items.
                        dct[subattr] = value
                result[key] = dct
            elif isinstance(attr, str) and hasattr(self._style, attr):
                # attr is attribute of style object, extract value
                value = getattr(self._style, attr)
                try:
                    result[key] = value[item]
                except TypeError:
                    # Constant properties that should not vary across
                    # plotted items.
                    result[key] = value
            else:
                # Use literal value, does not map into existing attribute
                result[key] = attr

        return result


class AbstractStyle:

    LEG_FONTPROP_KWARGS = {}
    LBL_FONTPROP_KWARGS = {}
    TICKLABEL_FONTPROP_KWARGS = {}
    TITLE_FONTPROP_KWARGS = {}
    SUBTITLE_FONTPROP_KWARGS = {}
    TEXT_FONTPROP_KWARGS = {}

    LEG_KWARGS = {}
    LBL_KWARGS = {}
    TICKLABEL_KWARGS = {}
    TITLE_KWARGS = {}
    SUPTITLE_KWARGS = {}
    FIGURE_KWARGS = {}
    SUBPLOT_KWARGS = {}
    GRID_KWARGS = {}
    TEXT_KWARGS = {}
    GUIDELINE_KWARGS = {}

    COLORS = ['black']
    FACECOLORS = ['white']
    LINESTYLES = ['-']
    EDGELINESTYLE = ['-']
    ALPHAS = [1.0]
    EDGEALPHA = [0.3]
    MARKERS = [None]
    LINEWIDTH = [1.0]
    EDGELINEWIDTH = [0.25]
    MARKERSIZE = [1.0]
    MEC = ['none']
    MFC = [None]
    HATCH = [None]

    def __init__(self):
        cls = self.__class__

        self.cell_size = 6
        self.dpi = 96
        self.aspect = 1.0
        self.ax_aspect = None   # Aspect ratio to set via ax.set_aspect()
        self._margins = 0.02
        self._grid = cls.GRID_KWARGS.copy()
        self._color = None
        self._facecolor = None
        self._facealpha = None
        self._linewidth = None
        self._linestyle = None
        self._edgecolor = None
        self._edgelinestyle = None
        self._edgelinewidth = None
        self._edgealpha = None
        self._ecolor = None
        self._elinewidth = None
        self.capsize = 0.0
        self.capthick = None
        self._alpha = None
        self._marker = None
        self._markersize = None
        self._markevery = None
        self._mec = None
        self._mfc = None
        self._mew = None
        self._hatch = None
        self._barmargin = 0.0
        self._zorder = None
        self._figure = cls.FIGURE_KWARGS.copy()
        self._subplot = cls.SUBPLOT_KWARGS.copy()
        self._ylabel = None
        self._xlabel = None
        self._xticklabels = None
        self._yticklabels = None
        self._title = None
        self._suptitle = None
        self._legend = None
        self._text = None
        self.split_scatter = False
        self.rotate_yticklabels = False
        self._guideline = dict(cls.GUIDELINE_KWARGS)

        self._plot_all = PlotStyleDict(self)

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        obj = cls()

        blacklist = ()

        for attr in dir(self):
            if attr in blacklist:
                continue
            elif attr.startswith('__'):
                # Skip all private stuff used by Python
                continue
            elif isinstance(getattr(cls, attr, None), property):
                # Skip all properties
                continue
            elif hasattr(cls, attr):
                # Skip class attributes
                continue

            value = getattr(self, attr)

            if callable(value):
                # Skip all methods
                continue
            elif value is None:
                setattr(self, attr, None)
            elif isinstance(value, (int, float)):
                setattr(obj, attr, value)
            else:
                setattr(obj, attr, deepcopy(value, memodict))

        # Manually fix this legacy thing, it will otherwise point to wrong
        # object
        obj._plot_all = PlotStyleDict(obj)

        return obj

    @property
    def figure(self):
        return self._figure

    @property
    def legend(self):
        if self._legend is None:
            cls = self.__class__
            fp = FontProperties(**cls.LEG_FONTPROP_KWARGS)
            self._legend = cls.LEG_KWARGS.copy()
            # Add font properties
            self._legend.update({'prop': fp})
        return self._legend

    @legend.setter
    def legend(self, value):
        self._legend = dict(value)

    @property
    def text(self):
        if self._text is None:
            cls = self.__class__
            fp = FontProperties(**cls.TEXT_FONTPROP_KWARGS)
            self._text = cls.TEXT_KWARGS.copy()
            self._text.update({'fontproperties': fp})
        return self._text

    @text.setter
    def text(self, value):
        self._text = dict(value)

    @property
    def title(self):
        if self._title is None:
            cls = self.__class__
            fp = FontProperties(**cls.TITLE_FONTPROP_KWARGS)
            self._title = cls.TITLE_KWARGS.copy()
            self._title.update({'fontproperties': fp})
        return self._title

    @title.setter
    def title(self, value):
        self._title = dict(value)

    @property
    def suptitle(self):
        if self._suptitle is None:
            cls = self.__class__
            fp = FontProperties(**cls.SUBTITLE_FONTPROP_KWARGS)
            self._suptitle = cls.SUPTITLE_KWARGS.copy()
            self._suptitle.update({'fontproperties': fp})
        return self._suptitle

    @suptitle.setter
    def suptitle(self, value):
        self._suptitle = dict(value)

    @property
    def xlabel(self):
        if self._xlabel is None:
            cls = self.__class__
            fp = FontProperties(**cls.LBL_FONTPROP_KWARGS)
            self._xlabel = cls.LBL_KWARGS.copy()
            self._xlabel.update({'fontproperties': fp})
        return self._xlabel

    @xlabel.setter
    def xlabel(self, value):
        self._xlabel = dict(value)

    @property
    def ylabel(self):
        if self._ylabel is None:
            cls = self.__class__
            fp = FontProperties(**cls.LBL_FONTPROP_KWARGS)
            self._ylabel = cls.LBL_KWARGS.copy()
            self._ylabel.update({'fontproperties': fp})
        return self._ylabel

    @ylabel.setter
    def ylabel(self, value):
        self._ylabel = dict(value)

    @property
    def xticklabels(self):
        if self._xticklabels is None:
            cls = self.__class__
            fp = FontProperties(**cls.TICKLABEL_FONTPROP_KWARGS)
            self._xticklabels = cls.TICKLABEL_KWARGS.copy()
            self._xticklabels.update({'fontproperties': fp})
        return self._xticklabels

    @xticklabels.setter
    def xticklabels(self, value):
        """
        Set the keyword arguments passed to set_xticklabels()

        Parameters
        ----------
        value : collections.abc.Mapping
        """
        self._xticklabels = dict(value)

    @property
    def yticklabels(self):
        if self._yticklabels is None:
            cls = self.__class__
            fp = FontProperties(**cls.TICKLABEL_FONTPROP_KWARGS)
            self._yticklabels = cls.TICKLABEL_KWARGS.copy()
            self._yticklabels.update({'fontproperties': fp})
        return self._yticklabels

    @yticklabels.setter
    def yticklabels(self, value):
        """
        Set the keyword arguments passed to set_yticklabels()

        Parameters
        ----------
        value : collections.abc.Mapping
        """
        self._yticklabels = dict(value)

    @property
    def grid(self) -> dict:
        return self._grid

    @grid.setter
    def grid(self, value: Union[Mapping, bool]):
        if isinstance(value, bool):
            # b has been removed from recent versions of MPL
            b = self._grid.get('b', True)
            visible = self._grid.get('visible', True)
            if value:
                if not (b or visible):
                    # re-apply default grid params, as just setting b=True will
                    # not produce any grid once it's been turned off.
                    # Do this only if b=False, otherwise ignore grid=True
                    # as it's enabled in some form anyways.
                    self._grid = self.__class__.GRID_KWARGS.copy()
                self._grid['visible'] = True
            else:
                self._grid = {'visible': False}
        else:
            # Create dictionary from given value
            self._grid = dict(value)

    @property
    def subplot(self):
        if self._subplot is None:
            cls = self.__class__
            self._subplot = cls.SUBPLOT_KWARGS.copy()
        return self._subplot

    @property
    def color(self):
        if self._color is None:
            self._color = Colors(type(self).COLORS)
        return self._color

    @color.setter
    def color(self, value):
        if isinstance(value, Colors):
            self._color = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._color = Colors(colors=value)

    @property
    def edgecolor(self):
        if self._edgecolor is None:
            # If nothing is set use the default colors
            self._edgecolor = self.color
        return self._edgecolor

    @edgecolor.setter
    def edgecolor(self, value):
        if isinstance(value, Colors):
            self._edgecolor = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgecolor = Colors(colors=value)

    @property
    def facecolor(self):
        cls = self.__class__
        if self._facecolor is None:
            if cls.FACECOLORS:
                self._facecolor = Colors(cls.FACECOLORS)
            else:
                self._facecolor = deepcopy(self.color)
        return self._facecolor

    @facecolor.setter
    def facecolor(self, value):
        if isinstance(value, Colors):
            self._facecolor = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._facecolor = Colors(colors=value)

    @property
    def facealpha(self):
        if self._facealpha is None:
            self._facealpha = Transparency(type(self).ALPHAS)
        return self._facealpha

    @facealpha.setter
    def facealpha(self, value):
        if isinstance(value, Transparency):
            self._facealpha = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._facealpha = Colors(colors=value)

    @property
    def linewidth(self):
        cls = self.__class__
        if self._linewidth is None:
            self._linewidth = LineWidth(cls.LINEWIDTH)
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if isinstance(value, LineWidth):
            self._linewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._linewidth = LineWidth(value)

    @property
    def lw(self):
        return self.linewidth

    @property
    def edgelinewidth(self):
        if self._edgelinewidth is None:
            self._edgelinewidth = LineWidth(type(self).EDGELINEWIDTH)
        return self._edgelinewidth

    @edgelinewidth.setter
    def edgelinewidth(self, value):
        if isinstance(self, LineWidth):
            self._edgelinewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgelinewidth = LineWidth(value)

    @property
    def linestyle(self):
        if self._linestyle is None:
            self._linestyle = LineStyle(type(self).LINESTYLES)
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        if isinstance(value, LineStyle):
            self._linestyle = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._linestyle = LineStyle(value)

    @property
    def ls(self):
        return self.linestyle

    @property
    def edgelinestyle(self):
        if self._edgelinestyle is None:
            self._edgelinestyle = LineStyle(type(self).EDGELINESTYLE)
        return self._edgelinestyle

    @edgelinestyle.setter
    def edgelinestyle(self, value):
        if isinstance(value, LineStyle):
            self._edgelinestyle = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgelinestyle = LineStyle(value)

    @property
    def edgealpha(self):
        if self._edgealpha is None:
            self._edgealpha = Transparency(type(self).EDGEALPHA)
        return self._edgealpha

    @edgealpha.setter
    def edgealpha(self, value):
        if isinstance(value, Transparency):
            self._edgealpha = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgealpha = Transparency(value)

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = Transparency(type(self).ALPHAS)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if isinstance(value, Transparency):
            self._alpha = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._alpha = Transparency(value)

    @property
    def ecolor(self):
        if self._ecolor is None:
            # If nothing is set use the default colors
            self._ecolor = Colors(['black'])
        return self._ecolor

    @ecolor.setter
    def ecolor(self, value):
        if isinstance(value, Colors):
            self._ecolor = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._ecolor = Colors(colors=value)

    @property
    def elinewidth(self):
        if self._elinewidth is None:
            self._elinewidth = LineWidth((1.0, ))
        return self._elinewidth

    @elinewidth.setter
    def elinewidth(self, value):
        if isinstance(value, LineWidth):
            self._elinewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._elinewidth = LineWidth(value)

    @property
    def marker(self):
        if self._marker is None:
            self._marker = Marker(type(self).MARKERS)
        return self._marker

    @marker.setter
    def marker(self, value):
        if isinstance(value, Marker):
            self._marker = deepcopy(value)
        else:
            value = _to_tuple(value)
            if value is None:
                value = [None]
            self._marker = Marker(value)

    @property
    def markersize(self):
        if self._markersize is None:
            self._markersize = ConstFillProperty(const=type(self).MARKERSIZE)
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        if isinstance(value, ConstFillProperty):
            self._markersize = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = value[-1] if value else 0.0
            self._markersize = ConstFillProperty(default, value)

    @property
    def markevery(self):
        if self._markevery is None:
            self._markevery = ConstFillProperty(const=1)
        return self._markevery

    @markevery.setter
    def markevery(self, value):
        if isinstance(value, ConstFillProperty):
            self._markevery = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = value[-1]
            self._markevery = ConstFillProperty(default, value)

    @property
    def mec(self):
        if self._mec is None:
            self._mec = Colors(type(self).MEC)
        return self._mec

    @mec.setter
    def mec(self, value):
        if isinstance(value, Colors):
            self._mec = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._mec = Colors(colors=value)

    @property
    def mfc(self):
        if self._mfc is None:
            self._mfc = Colors(type(self).MFC)
        return self._mfc

    @mfc.setter
    def mfc(self, value):
        if isinstance(value, Colors):
            self._mfc = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._mfc = Colors(colors=value)

    @property
    def mew(self):
        if self._mew is None:
            self._mew = LineWidth((0.5, ))
        return self._mew

    @mew.setter
    def mew(self, value):
        if isinstance(value, LineWidth):
            self._mew = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._mew = LineWidth(value)

    @property
    def hatch(self):
        if self._hatch is None:
            self._hatch = ConstFillProperty(None)
        return self._hatch

    @hatch.setter
    def hatch(self, value):
        if isinstance(value, ConstFillProperty):
            self._hatch = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = None
            self._hatch = ConstFillProperty(default, value)

    @property
    def barmargin(self):
        return self._barmargin

    @barmargin.setter
    def barmargin(self, value):
        try:
            value = float(value)
        except TypeError:
            raise ValueError('Margin must be float!')

        if value < 0.0 or value >= 0.5:
            raise ValueError('Margin value must be in [0, 0.5)')

        self._barmargin = value

    @property
    def zorder(self):
        if self._zorder is None:
            self._zorder = ConstFillProperty(const=10)
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        if isinstance(value, ConstFillProperty):
            self._zorder = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._zorder = ConstFillProperty(10, value)

    @property
    def guideline(self) -> dict:
        return self._guideline

    @guideline.setter
    def guideline(self, value: Optional[Mapping] = None):
        if value is not None:
            self._guideline = dict(value)
        else:
            self._guideline = dict()

    @property
    def margins(self):
        return self._margins

    @margins.setter
    def margins(self, value):
        """
        Set subplot margins. Values are relative to the data margins and
        need to be in the interval [0, 1]. If multiple values are given,
        these are interpreted to be in the order (left, bottom, right, top).

        If set to None, scaling or margins is disabled.

        Parameters
        ----------
        value : int or array_like, optional
        """
        if value is not None:
            try:
                value = float(value)
            except TypeError:
                try:
                    value = np.atleast_1d(value)
                    if value.size != 1 and value.size != 4:
                        raise ValueError('margins value not understood')
                    if np.all(value[0] == value[1:]):
                        # Store as float since it's the same value for all sides
                        value = float(value[0])
                except TypeError:
                    raise ValueError('margins value not understood')

        self._margins = value

    @property
    def figure_layout(self):
        """
        Return the figure layout setting

        Returns
        -------
        FigureLayout
        """
        if not self._figure:
            return FigureLayout.DEFAULT

        for k, v in _FIGURE_LAYOUT_MAP.items():
            if self._figure.get(k, False):
                return v
        else:
            return FigureLayout.DEFAULT

    @figure_layout.setter
    def figure_layout(self, value):
        """
        Set the figure layout

        Parameters
        ----------
        value : FigureLayout
        """
        if not isinstance(value, FigureLayout):
            raise ValueError('Argument must be of FigureLayout type')

        # Delete all layout entries, add (back) only the one requested by caller

        for k in _FIGURE_LAYOUT_MAP.keys():
            if k in self._figure:
                del self._figure[k]

        for k, v in _FIGURE_LAYOUT_MAP.items():
            if value == v:
                self._figure[k] = True

    @property
    def plot_kwargs(self):
        return self._plot_all

    @property
    def fill_between_kwargs(self):
        """
        Return a sequence of collections of key/value pairs that can be passed
        to matplotlib's fill_between()

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'edgecolor': None,
            'facecolor': None,
            'lw': 'edgelinewidth',
            'ls': 'edgelinestyle',
            'alpha': 'facealpha',
            'zorder': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def fill_between_edge_kwargs(self):
        """
        Return a sequence of collections of key/value pairs that can be passed to
        matplotlib's plot() function when separately plotting the lower
        and upper edge lines of the area shaded by fill_between()

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'color': 'edgecolor',
            'ls': 'edgelinestyle',
            'lw': 'edgelinewidth',
            'alpha': 'edgealpha',
            'zorder': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def errorbar_kwargs(self):
        """
        Return a sequence of collections of key/value pairs that can be passed to
        matplotlib's errorbar().

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'ecolor': None,
            'elinewidth': None,
            'capsize': None,
            'capthick': None,
            'color': None,
            'ls': None,
            'lw': None,
            'alpha': None,
            'marker': None,
            'mec': None,
            'mfc': None,
            'mew': None,
            'markersize': None,
            'markevery': None,
            'zorder': None,
            'errorevery': 'markevery'
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def errorbar_no_marker_kwargs(self):
        """
        Return a sequence of collections of key/value pairs that can be passed to
        matplotlib's errorbar(). All marker-related attributes are
        stripped and marker attribute is set to 'None' so that no markers
        are plotted.

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'ecolor': None,
            'elinewidth': None,
            'capsize': None,
            'capthick': None,
            'color': None,
            'ls': None,
            'lw': None,
            'alpha': None,
            'zorder': None,
            'marker': 'None',
            'errorevery': 'markevery',
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def marker_no_line_kwargs(self):
        """
        Return a sequence of collections of key/value pairs that can be passed to
        matplotlib's plot() or errorbar(). Only includes marker-related
        attributes and disables connecting lines.

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'color': 'facecolor',
            'marker': None,
            'markersize': None,
            'mec': None,
            'mfc': None,
            'mew': None,
            'ls': 'none',
            'lw': 0,
            'alpha': 'facealpha',
            'zorder': None,
            'markevery': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def bar_kwargs(self):
        """
        Returns a sequence of collections of key/value pairs that can be passed
        to matplotlib's bar().

        Returns
        -------
        StyleAttrMapping
        """

        # NOTE: Do not use facealpha for bar charts. facealpha is meant for
        # shaded areas such as CIs, etc. which should be in the background,
        # which is not what we'd want for bars.

        mapping = {
            'color': 'facecolor',
            'edgecolor': None,
            'lw': 'edgelinewidth',
            'ls': None,
            'alpha': None,
            'zorder': None,
            'hatch': None,
            'capsize': None,
            'ecolor': None,
            'error_kw': ['elinewidth', 'capthick']
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_kwargs(self):
        """
        Returns a sequence of collections of key/value pairs that can be passed
        to matplotlib's scatter().

        Returns
        -------
        StyleAttrMapping
        """

        mapping = {
            'facecolors': 'facecolor',
            'edgecolors': 'edgecolor',
            'linewidths': 'edgelinewidth',
            'linestyles': 'edgelinestyle',
            'alpha': 'alpha',
            'marker': None,
            'zorder': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_face_kwargs(self):
        """
        Returns a sequence of collections of key/value parts that can be used
        to plot the "face" component of split scatter plots and should
        be passed to matplotlib's scatter().

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'facecolors': 'facecolor',
            'alpha': 'facealpha',
            'linewidths': 0,
            'zorder': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_edge_kwargs(self):
        """
        Returns a sequence of collections of key/value parts that can be used
        to plot the "edge" component of split scatter plots and should
        be passed to matplotlib's scatter().

        Returns
        -------
        StyleAttrMapping
        """

        mapping = {
            'facecolor': 'none',
            'edgecolors': 'edgecolor',
            'linewidths': 'edgelinewidth',
            'linestyles': 'edgelinestyle',
            'alpha': 'edgealpha',
            'marker': None,
            'zorder': None
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs


class DefaultStyle(AbstractStyle):

    LEG_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'small'
    }

    LBL_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'medium'
    }

    TICKLABEL_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'small'
    }

    TITLE_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'medium',
        'style': 'italic'
    }

    SUBTITLE_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'x-large',
        'style': 'italic',
        'weight': 'semibold'
    }

    TEXT_FONTPROP_KWARGS = {
        'family': 'serif',
        'style': 'italic',
        'size': 'small'
    }

    GUIDELINE_KWARGS = {
        'lw': 0.75,
        'ls': (0, (1, 1)),
        'alpha': 0.7,
        'color': 'black',
        'zorder': -10
    }

    # Keyword arguments (other than font properties) for various objects
    LEG_KWARGS = {'framealpha': .7, 'frameon': True, 'fancybox': False}

    LBL_KWARGS = {}
    TITLE_KWARGS = {}
    SUPTITLE_KWARGS = {}

    SUBPLOT_KWARGS = {
        'facecolor': 'white',
        'axisbelow': True
    }

    FIGURE_KWARGS = {
        'constrained_layout': True
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
    EDGELINESTYLE = ['-']
    ALPHAS = [.9, 0.7, 0.7, 1.0]
    MARKERS = [None]
    LINEWIDTH = [2]
    EDGELINEWIDTH = [0.5]
    MARKERSIZE = 5
    MEC = ['white']
    COLORS = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#f781bf']
    # Default values for facecolor: force same as color
    FACECOLORS = None

    def __init__(self):

        super(DefaultStyle, self).__init__()


class PurpleBlue(DefaultStyle):

    COLORS = ['#810f7c', '#737373', '#045a8d', '#807dba', '#f768a1', '#3690c0']
    FACECOLORS = ['#8c6bb1', '#dadaeb', '#0570b0', '#8f8cd0', '#fcc5c0', '#a6bddb']


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


class AlternatingStyle(DefaultStyle):
    """
    Style definition that alternates solid colored lines with black lines
    with dashed/dotted/etc. line styles.
    """

    GRID_KWARGS = {
        'color': 'black',
        'alpha': 0.5,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.5
    }

    def __init__(self):

        super().__init__()

        colors = ['#0570b0', '#e31a1c', '#88419d', '#fc8d59', '#252525']
        # colors = ['#0570b0', '#d94801', '#41ae76', '#6a51a3', '#d7301f']
        ls_colors = ['-'] * len(colors)
        lw_colors = [1.5] * len(colors)
        alpha_color = [0.8] * len(colors)
        markers_color = [None, 'o', 'X', 'D', None]

        ls_black = ['-', '--', '-.', '-', (0, (2, 1))]
        lw_black = [1.05] * len(ls_black)
        black = ['black'] * len(ls_black)
        alpha_black = [0.75] * len(ls_black)
        markers_black = [None, None, None, 'o', None]

        colors = it.chain(*zip(colors, black))
        ls = it.chain(*zip(ls_colors, ls_black))
        lw = it.chain(*zip(lw_colors, lw_black))
        alpha = it.chain(*zip(alpha_color, alpha_black))
        markers = it.chain(*zip(markers_color, markers_black))

        self.color = colors
        self.linestyle = ls
        self.linewidth = lw
        self.alpha = alpha
        self.marker = markers


class QualitativeStyle(DefaultStyle):
    """
    Style definition that with identical line styles but alternating
    colors, similar to the qualitative color schemes on colorbrewer2.org
    """

    GRID_KWARGS = {
        'color': 'black',
        'alpha': 0.5,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.35
    }

    def __init__(self):

        super().__init__()

        colors = ['#0570b0', 'black', '#e31a1c', '#88419d', '#fc8d59', '#aa5500']

        self.color = colors
        self.linestyle = ['-']
        self.linewidth = 1.0
        self.alpha = [0.8, 0.7, 0.8, 0.8, 0.9, 0.8]


MPL_VERSION = matplotlib.__version__.split('.')
try:
    major = int(MPL_VERSION[0])
    if major < 2:
        val = DefaultStyle.SUBPLOT_KWARGS['facecolor']
        del DefaultStyle.SUBPLOT_KWARGS['facecolor']
        DefaultStyle.SUBPLOT_KWARGS['axisbg'] = val
except:
    pass
