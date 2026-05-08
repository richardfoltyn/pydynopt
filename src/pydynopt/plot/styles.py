"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Iterator, Mapping, Sequence
import numpy as np
from copy import deepcopy

from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import itertools as it

import copy

from pydynopt.utils import anything_to_tuple

import enum


class FigureLayout(enum.Enum):
    DEFAULT = 1
    TIGHT_LAYOUT = 2
    CONSTRAINED_LAYOUT = 3


_FIGURE_LAYOUT_MAP = {
    'constrained_layout': FigureLayout.CONSTRAINED_LAYOUT,
    'tight_layout': FigureLayout.TIGHT_LAYOUT,
}

# Mapping from keyword arguments for ticklabels() to tick_params()
_TICKLABEL_PARAMS_MAP = dict(
    fontfamily='labelfontfamily',
    fontsize='labelsize',
    color='labelcolor',
    rotation='labelrotation',
)


class PaletteType(enum.Enum):
    DIVERGING = 0
    QUALITATIVE = 1
    SEQUENTIAL = 2

    def __str__(self) -> str:
        return self.name.lower()


_DEFAULT_MPL_MAP = dict(
    linestyle='ls',
    linewidth='lw',
    markeredgecolor='mec',
    markeredgewidth='mew',
    markerfacecolor='mfc',
    markersize='ms',
    color='c',
)


def select_font(font_family: str | Sequence[str], default: str = 'serif') -> str:
    """
    Select a font family from the system's available fonts. If the desired font
    is not available, return a default font family.

    Parameters
    ----------
    font_family : str
        The desired font family.
    default : str, optional
        The default font family to use if the desired font is not available.

    Returns
    -------
    str
        The selected font family.
    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if isinstance(font_family, str):
        return font_family if font_family in available_fonts else default
    else:
        for ff in font_family:
            if ff in available_fonts:
                return ff
        return default


class UniqueDict(dict):
    """
    Sub-class of dict which maps long forms of MPL arguments to plot() and other
    functions into the abbreviated forms, so that ever argument has one unique value
    ("lw" instead of possibly "lw" and "linewidth", etc.)
    """

    def __init__(
        self, mapping: Mapping[str, object] | None = None, **kwargs: object
    ) -> None:
        self.mapping = mapping
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: object) -> None:
        key = self.mapping.get(key, key)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> object:
        key = self.mapping.get(key, key)
        return super().__getitem__(key)


def _to_tuple(value: object) -> tuple[object, ...]:
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

    if isinstance(value, tuple):
        value = copy.copy(value)
    elif value is None:
        value = (None,)
    else:
        value = anything_to_tuple(value)

    return value


class Cycler:
    def __init__(self, items: object = None) -> None:
        self.items: tuple = _to_tuple(items)
        self.cache: tuple | None = None

    def __getitem__(self, item: int | slice) -> object:
        if isinstance(item, slice):
            end = item.stop
            if end is None and self.cache:
                end = len(self.items)
        else:
            end = item
        if not self.cache or end >= len(self.cache):
            col = it.cycle(self.items)
            self.cache = tuple(next(col) for _ in range(end + 1))
        return self.cache[item]

    def __deepcopy__(self, memodict: dict[object, object] = {}) -> 'Cycler':
        obj = type(self)(self.items)
        return obj

    def __repr__(self) -> str:
        s = f'{self.items}'.strip('()[]')
        s = f'{type(self).__name__}({s})'
        return s


class Colors(Cycler):
    pass


class LineStyle(Cycler):
    pass


class LineWidth(Cycler):
    pass


class Marker(Cycler):
    pass


class Transparency(Cycler):
    pass


class ConstFillProperty:
    def __init__(self, const: object, values: object = None) -> None:
        self.values = copy.copy(values)
        self.const = const

    def __getitem__(self, item: int) -> object:
        if self.values is None or item >= len(self.values):
            return self.const
        else:
            return self.values[item]

    def __deepcopy__(self, memodict: dict[object, object] = {}) -> 'ConstFillProperty':
        obj = ConstFillProperty(self.const, self.values)
        return obj


class PlotStyleDict(object):
    def __init__(self, style: 'AbstractStyle') -> None:
        self.style = style

    def __getitem__(self, item: int) -> dict[str, object]:
        keys = {
            'color',
            'lw',
            'ls',
            'alpha',
            'marker',
            'mec',
            'mfc',
            'mew',
            'markersize',
            'markevery',
            'zorder',
        }
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

    def __init__(self, style: 'AbstractStyle', mapping: Mapping[str, object]) -> None:
        self._style = style
        self._mapping = mapping

    def __getitem__(self, item: int) -> dict[str, object]:
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
    LEG_TITLE_FONTPROP_KWARGS = {}
    LBL_FONTPROP_KWARGS = {}
    TITLE_FONTPROP_KWARGS = {}
    SUPTITLE_FONTPROP_KWARGS = {}
    TEXT_FONTPROP_KWARGS = {}

    LEG_KWARGS = {}
    LBL_KWARGS = {}
    TICKLABEL_KWARGS = {}
    TICK_PARAMS_KWARGS = {}
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

    @classmethod
    def _iter_class_defaults(
        cls: type['AbstractStyle'],
    ) -> Iterator[tuple[str, object]]:
        """
        Yield class-level default attributes used to initialize style instances.

        The method walks the method resolution order (MRO) from base classes to
        subclasses so that subclass attributes override base-class attributes with
        the same name. Only public, ALL_CAPS attributes are included; methods,
        properties, descriptors, and private names are excluded.

        Parameters
        ----------
        cls : type[AbstractStyle]
            Style class whose defaults should be collected.

        Yields
        ------
        tuple[str, object]
            Pairs of attribute name and attribute value for each default.
        """
        for base in reversed(cls.mro()):
            if not issubclass(base, AbstractStyle):
                continue
            for name, value in base.__dict__.items():
                if not name.isupper() or name.startswith('_'):
                    continue
                if isinstance(value, (staticmethod, classmethod, property)):
                    continue
                if callable(value):
                    continue
                yield name, value

    def _freeze_class_defaults(self) -> None:
        """
        Copy class-level defaults onto the instance.

        This creates instance-owned snapshots of all default style attributes so
        that subsequent mutations to class attributes do not affect existing
        objects.

        Parameters
        ----------
        self : AbstractStyle

        Returns
        -------
        None
        """
        for name, value in type(self)._iter_class_defaults():
            setattr(self, name, deepcopy(value))

    def __init__(self, fontfamily: str | None = None) -> None:
        self._freeze_class_defaults()

        self.cell_size = 6
        self.dpi = 96
        self.aspect = 1.0
        # Note: MPL defines aspect as height / width
        self.ax_aspect = None  # Aspect ratio to set via ax.set_aspect()
        # Note: MPL defines aspect as height / width
        self.ax_box_aspect = None  # Aspect ratio to set via ax.set_box_aspect()
        self._margins = 0.02
        self._grid = self.GRID_KWARGS.copy()
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
        self._figure = self.FIGURE_KWARGS.copy()
        self._subplot = self.SUBPLOT_KWARGS.copy()
        self._ylabel = None
        self._xlabel = None
        self._xticklabels = None
        self._xtick_params = None
        self._yticklabels = None
        self._ytick_params = None
        self._title = None
        self._suptitle = None
        self._legend = None
        self._text = None
        self.split_scatter = False
        self._guideline = UniqueDict(mapping=_DEFAULT_MPL_MAP, **self.GUIDELINE_KWARGS)

        self._plot_all = PlotStyleDict(self)

        # Override font family instead of having to redefine all font properties
        # in subclasses. This can be useful if we want different fonts for
        # slides and tables.
        if fontfamily:
            for attr, value in self.__dict__.items():
                # Update font family in dicts used to construct font properties
                if attr.find('FONTPROP') != -1 and isinstance(value, dict):
                    value['family'] = fontfamily
                # Update font family in dicts used as keyword arguments
                elif attr.endswith('KWARGS') and isinstance(value, dict):
                    if 'fontfamily' in value:
                        value['fontfamily'] = fontfamily
                    elif 'family' in value:
                        value['family'] = fontfamily

    def __deepcopy__(self, memodict: dict[object, object] = {}) -> 'AbstractStyle':
        cls = self.__class__
        obj = cls()

        for attr, value in self.__dict__.items():
            if attr == '_plot_all':
                continue
            if callable(value):
                continue
            if value is None:
                setattr(obj, attr, None)
            elif isinstance(value, (int, float)):
                setattr(obj, attr, value)
            else:
                setattr(obj, attr, deepcopy(value, memodict))

        # Manually fix this legacy thing, it will otherwise point to wrong
        # object
        obj._plot_all = PlotStyleDict(obj)

        return obj

    @property
    def figure(self) -> dict[str, object]:
        return self._figure

    @property
    def legend(self) -> dict[str, object]:
        if self._legend is None:
            self._legend = self.LEG_KWARGS.copy()
            # Add font properties
            fp = FontProperties(**self.LEG_FONTPROP_KWARGS)
            self._legend.update({'prop': fp})
            if self.LEG_TITLE_FONTPROP_KWARGS:
                fp = FontProperties(**self.LEG_TITLE_FONTPROP_KWARGS)
                self._legend.update(title_fontproperties=fp)
        return self._legend

    @legend.setter
    def legend(self, value: Mapping[str, object]) -> None:
        self._legend = dict(value)

    @property
    def text(self) -> dict[str, object]:
        if self._text is None:
            fp = FontProperties(**self.TEXT_FONTPROP_KWARGS)
            self._text = self.TEXT_KWARGS.copy()
            self._text.update({'fontproperties': fp})
        return self._text

    @text.setter
    def text(self, value: Mapping[str, object]) -> None:
        self._text = dict(value)

    @property
    def title(self) -> dict[str, object]:
        if self._title is None:
            fp = FontProperties(**self.TITLE_FONTPROP_KWARGS)
            self._title = self.TITLE_KWARGS.copy()
            self._title.update({'fontproperties': fp})
        return self._title

    @title.setter
    def title(self, value: Mapping[str, object]) -> None:
        self._title = dict(value)

    @property
    def suptitle(self) -> dict[str, object]:
        if self._suptitle is None:
            fp = FontProperties(**self.SUPTITLE_FONTPROP_KWARGS)
            self._suptitle = self.SUPTITLE_KWARGS.copy()
            self._suptitle.update({'fontproperties': fp})
        return self._suptitle

    @suptitle.setter
    def suptitle(self, value: Mapping[str, object]) -> None:
        self._suptitle = dict(value)

    @property
    def xlabel(self) -> dict[str, object]:
        if self._xlabel is None:
            fp = FontProperties(**self.LBL_FONTPROP_KWARGS)
            self._xlabel = self.LBL_KWARGS.copy()
            self._xlabel.update({'fontproperties': fp})
        return self._xlabel

    @xlabel.setter
    def xlabel(self, value: Mapping[str, object]) -> None:
        self._xlabel = dict(value)

    @property
    def ylabel(self) -> dict[str, object]:
        if self._ylabel is None:
            fp = FontProperties(**self.LBL_FONTPROP_KWARGS)
            self._ylabel = self.LBL_KWARGS.copy()
            self._ylabel.update({'fontproperties': fp})
        return self._ylabel

    @ylabel.setter
    def ylabel(self, value: Mapping[str, object]) -> None:
        self._ylabel = dict(value)

    @property
    def xticklabels(self) -> dict[str, object]:
        if self._xticklabels is None:
            self._xticklabels = self.TICKLABEL_KWARGS.copy()
        return self._xticklabels

    @xticklabels.setter
    def xticklabels(self, value: Mapping[str, object]) -> None:
        """
        Set the keyword arguments passed to set_xticklabels()

        Parameters
        ----------
        value : collections.abc.Mapping
        """
        self._xticklabels = dict(value)

    @property
    def yticklabels(self) -> dict[str, object]:
        if self._yticklabels is None:
            self._yticklabels = self.TICKLABEL_KWARGS.copy()
        return self._yticklabels

    @yticklabels.setter
    def yticklabels(self, value: Mapping[str, object]) -> None:
        """
        Set the keyword arguments passed to set_yticklabels()

        Parameters
        ----------
        value : collections.abc.Mapping
        """
        self._yticklabels = dict(value)

    @property
    def xtick_params(self) -> dict[str, object]:
        """
        Returns collection of keyword arguments that can be passed to
        set_tick_params().

        Returns
        -------
        dict
        """
        if self._xtick_params is None:
            self._xtick_params = self.TICK_PARAMS_KWARGS.copy()
            # Update with label properties. Keyword arguments have different names
            # when passed to tick_params() and only a subset is supported.
            self._xtick_params.update(
                {
                    _TICKLABEL_PARAMS_MAP[k]: v
                    for k, v in self.xticklabels.items()
                    if k in _TICKLABEL_PARAMS_MAP
                }
            )
        return self._xtick_params

    @xtick_params.setter
    def xtick_params(self, value: Mapping[str, object]) -> None:
        """
        Sets the collection of keyword arguments passed to
        set_tick_params().

        Parameters
        ----------
        value : Mapping
        """
        self._xtick_params = dict(value)

    @property
    def ytick_params(self) -> dict[str, object]:
        """
        Returns collection of keyword arguments that can be passed to
        set_tick_params().

        Returns
        -------
        dict
        """
        if self._ytick_params is None:
            self._ytick_params = self.TICK_PARAMS_KWARGS.copy()
            # Update with label properties. Keyword arguments have different names
            # when passed to tick_params() and only a subset is supported.
            self._ytick_params.update(
                {
                    _TICKLABEL_PARAMS_MAP[k]: v
                    for k, v in self.yticklabels.items()
                    if k in _TICKLABEL_PARAMS_MAP
                }
            )
        return self._ytick_params

    @ytick_params.setter
    def ytick_params(self, value: Mapping[str, object]) -> None:
        """
        Sets the collection of keyword arguments passed to
        set_tick_params().

        Parameters
        ----------
        value : Mapping
        """
        self._ytick_params = dict(value)

    @property
    def grid(self) -> dict[str, object]:
        return self._grid

    @grid.setter
    def grid(self, value: Mapping[str, object] | bool) -> None:
        if isinstance(value, bool):
            visible = self._grid.get('visible', True)
            if value:
                if not visible:
                    # re-apply default grid params, as just setting b=True will
                    # not produce any grid once it's been turned off.
                    # Do this only if b=False, otherwise ignore grid=True
                    # as it's enabled in some form anyway.
                    self._grid = self.GRID_KWARGS.copy()
                self._grid['visible'] = True
            else:
                self._grid = {'visible': False}
        else:
            # Filter legacy 'b' if present
            value = dict(value)
            if 'b' in value:
                if 'visible' not in value:
                    value['visible'] = value['b']
                del value['b']
            # Create dictionary from given value
            self._grid = dict(value)

    @property
    def subplot(self) -> dict[str, object]:
        if self._subplot is None:
            self._subplot = self.SUBPLOT_KWARGS.copy()
        return self._subplot

    @property
    def color(self) -> Colors:
        if self._color is None:
            self._color = Colors(self.COLORS)
        return self._color

    @color.setter
    def color(self, value: object) -> None:
        if isinstance(value, Colors):
            self._color = deepcopy(value)
        else:
            self._color = Colors(value)

    @property
    def edgecolor(self) -> Colors:
        if self._edgecolor is None:
            # If nothing is set use the default colors
            self._edgecolor = self.color
        return self._edgecolor

    @edgecolor.setter
    def edgecolor(self, value: object) -> None:
        if isinstance(value, Colors):
            self._edgecolor = deepcopy(value)
        else:
            self._edgecolor = Colors(value)

    @property
    def facecolor(self) -> Colors:
        if self._facecolor is None:
            if self.FACECOLORS:
                self._facecolor = Colors(self.FACECOLORS)
            else:
                self._facecolor = deepcopy(self.color)
        return self._facecolor

    @facecolor.setter
    def facecolor(self, value: object) -> None:
        if isinstance(value, Colors):
            self._facecolor = deepcopy(value)
        else:
            self._facecolor = Colors(value)

    @property
    def facealpha(self) -> Transparency:
        if self._facealpha is None:
            self._facealpha = Transparency(self.ALPHAS)
        return self._facealpha

    @facealpha.setter
    def facealpha(self, value: object) -> None:
        if isinstance(value, Transparency):
            self._facealpha = deepcopy(value)
        else:
            self._facealpha = Colors(value)

    @property
    def linewidth(self) -> LineWidth:
        if self._linewidth is None:
            self._linewidth = LineWidth(self.LINEWIDTH)
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value: object) -> None:
        if isinstance(value, LineWidth):
            self._linewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._linewidth = LineWidth(value)

    @property
    def lw(self) -> LineWidth:
        return self.linewidth

    @property
    def edgelinewidth(self) -> LineWidth:
        if self._edgelinewidth is None:
            self._edgelinewidth = LineWidth(self.EDGELINEWIDTH)
        return self._edgelinewidth

    @edgelinewidth.setter
    def edgelinewidth(self, value: object) -> None:
        if isinstance(self, LineWidth):
            self._edgelinewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgelinewidth = LineWidth(value)

    @property
    def linestyle(self) -> LineStyle:
        if self._linestyle is None:
            self._linestyle = LineStyle(self.LINESTYLES)
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value: object) -> None:
        if isinstance(value, LineStyle):
            self._linestyle = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._linestyle = LineStyle(value)

    @property
    def ls(self) -> LineStyle:
        return self.linestyle

    @property
    def edgelinestyle(self) -> LineStyle:
        if self._edgelinestyle is None:
            self._edgelinestyle = LineStyle(self.EDGELINESTYLE)
        return self._edgelinestyle

    @edgelinestyle.setter
    def edgelinestyle(self, value: object) -> None:
        if isinstance(value, LineStyle):
            self._edgelinestyle = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgelinestyle = LineStyle(value)

    @property
    def edgealpha(self) -> Transparency:
        if self._edgealpha is None:
            self._edgealpha = Transparency(self.EDGEALPHA)
        return self._edgealpha

    @edgealpha.setter
    def edgealpha(self, value: object) -> None:
        if isinstance(value, Transparency):
            self._edgealpha = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgealpha = Transparency(value)

    @property
    def alpha(self) -> Transparency:
        if self._alpha is None:
            self._alpha = Transparency(self.ALPHAS)
        return self._alpha

    @alpha.setter
    def alpha(self, value: object) -> None:
        if isinstance(value, Transparency):
            self._alpha = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._alpha = Transparency(value)

    @property
    def ecolor(self) -> Colors:
        if self._ecolor is None:
            # If nothing is set use the default colors
            self._ecolor = Colors(['black'])
        return self._ecolor

    @ecolor.setter
    def ecolor(self, value: object) -> None:
        if isinstance(value, Colors):
            self._ecolor = deepcopy(value)
        else:
            self._ecolor = Colors(value)

    @property
    def elinewidth(self) -> LineWidth:
        if self._elinewidth is None:
            self._elinewidth = LineWidth((1.0,))
        return self._elinewidth

    @elinewidth.setter
    def elinewidth(self, value: object) -> None:
        if isinstance(value, LineWidth):
            self._elinewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._elinewidth = LineWidth(value)

    @property
    def marker(self) -> Marker:
        if self._marker is None:
            self._marker = Marker(self.MARKERS)
        return self._marker

    @marker.setter
    def marker(self, value: object) -> None:
        if isinstance(value, Marker):
            self._marker = deepcopy(value)
        else:
            value = _to_tuple(value)
            if value is None:
                value = [None]
            self._marker = Marker(value)

    @property
    def markersize(self) -> ConstFillProperty:
        if self._markersize is None:
            self._markersize = ConstFillProperty(const=self.MARKERSIZE)
        return self._markersize

    @markersize.setter
    def markersize(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            self._markersize = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = value[-1] if value else 0.0
            self._markersize = ConstFillProperty(default, value)

    @property
    def markevery(self) -> ConstFillProperty:
        if self._markevery is None:
            self._markevery = ConstFillProperty(const=1)
        return self._markevery

    @markevery.setter
    def markevery(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            self._markevery = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = value[-1]
            self._markevery = ConstFillProperty(default, value)

    @property
    def mec(self) -> Colors:
        if self._mec is None:
            self._mec = Colors(self.MEC)
        return self._mec

    @mec.setter
    def mec(self, value: object) -> None:
        if isinstance(value, Colors):
            self._mec = deepcopy(value)
        else:
            self._mec = Colors(value)

    @property
    def mfc(self) -> Colors:
        if self._mfc is None:
            self._mfc = Colors(self.MFC)
        return self._mfc

    @mfc.setter
    def mfc(self, value: object) -> None:
        if isinstance(value, Colors):
            self._mfc = deepcopy(value)
        else:
            self._mfc = Colors(value)

    @property
    def mew(self) -> LineWidth:
        if self._mew is None:
            self._mew = LineWidth((0.5,))
        return self._mew

    @mew.setter
    def mew(self, value: object) -> None:
        if isinstance(value, LineWidth):
            self._mew = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._mew = LineWidth(value)

    @property
    def hatch(self) -> ConstFillProperty:
        if self._hatch is None:
            self._hatch = ConstFillProperty(None)
        return self._hatch

    @hatch.setter
    def hatch(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            self._hatch = deepcopy(value)
        else:
            value = _to_tuple(value)
            default = None
            self._hatch = ConstFillProperty(default, value)

    @property
    def barmargin(self) -> float:
        return self._barmargin

    @barmargin.setter
    def barmargin(self, value: object) -> None:
        try:
            value = float(value)
        except TypeError:
            raise ValueError('Margin must be float!')

        if value < 0.0 or value >= 0.5:
            raise ValueError('Margin value must be in [0, 0.5)')

        self._barmargin = value

    @property
    def zorder(self) -> ConstFillProperty:
        if self._zorder is None:
            self._zorder = ConstFillProperty(const=10)
        return self._zorder

    @zorder.setter
    def zorder(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            self._zorder = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._zorder = ConstFillProperty(10, value)

    @property
    def guideline(self) -> dict[str, object]:
        return self._guideline

    @guideline.setter
    def guideline(self, value: Mapping[str, object] | None = None) -> None:
        if value is not None:
            self._guideline = UniqueDict(mapping=_DEFAULT_MPL_MAP, **value)
        else:
            self._guideline = UniqueDict(mapping=_DEFAULT_MPL_MAP)

    @property
    def margins(self) -> float | np.ndarray | None:
        return self._margins

    @margins.setter
    def margins(self, value: object) -> None:
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
    def figure_layout(self) -> FigureLayout:
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
    def figure_layout(self, value: FigureLayout) -> None:
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
    def plot_kwargs(self) -> PlotStyleDict:
        return self._plot_all

    @property
    def fill_between_kwargs(self) -> StyleAttrMapping:
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
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def fill_between_face_kwargs(self) -> StyleAttrMapping:
        """
        Return a sequence of collections of key/value pairs that can be passed
        to matplotlib's fill_between() when plotting the "face" component.

        Returns
        -------
        StyleAttrMapping
        """
        mapping = {
            'facecolor': None,
            'alpha': 'facealpha',
            'lw': 0,
            'ls': '',
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def fill_between_edge_kwargs(self) -> StyleAttrMapping:
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
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def errorbar_kwargs(self) -> StyleAttrMapping:
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
            'errorevery': 'markevery',
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def errorbar_no_marker_kwargs(self) -> StyleAttrMapping:
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
    def marker_no_line_kwargs(self) -> StyleAttrMapping:
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
            'markevery': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def bar_kwargs(self) -> StyleAttrMapping:
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
            'error_kw': ['elinewidth', 'capthick'],
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_kwargs(self) -> StyleAttrMapping:
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
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_face_kwargs(self) -> StyleAttrMapping:
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
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs

    @property
    def scatter_edge_kwargs(self) -> StyleAttrMapping:
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
            'zorder': None,
        }

        kwargs = StyleAttrMapping(self, mapping)

        return kwargs


class DefaultStyle(AbstractStyle):
    LEG_FONTPROP_KWARGS = {'family': 'serif', 'size': 'small'}
    LEG_TITLE_FONTPROP_KWARGS = {'family': 'serif', 'size': 'small', 'weight': 'bold'}

    LBL_FONTPROP_KWARGS = {'family': 'serif', 'size': 'medium'}

    TICKLABEL_KWARGS = {'fontfamily': 'serif', 'fontsize': 'small'}

    TITLE_FONTPROP_KWARGS = {'family': 'serif', 'size': 'medium', 'style': 'italic'}

    SUPTITLE_FONTPROP_KWARGS = {
        'family': 'serif',
        'size': 'x-large',
        'style': 'italic',
        'weight': 'semibold',
    }

    TEXT_FONTPROP_KWARGS = {'family': 'serif', 'style': 'italic', 'size': 'small'}

    GUIDELINE_KWARGS = {
        'lw': 0.75,
        'ls': (0, (1, 1)),
        'alpha': 0.7,
        'color': 'black',
        'zorder': -10,
    }

    # Keyword arguments (other than font properties) for various objects
    LEG_KWARGS = {'framealpha': 0.7, 'frameon': True, 'fancybox': False}

    LBL_KWARGS = {}
    TITLE_KWARGS = {}
    SUPTITLE_KWARGS = {}

    SUBPLOT_KWARGS = {'facecolor': 'white', 'axisbelow': True}

    FIGURE_KWARGS = {'constrained_layout': True}

    GRID_KWARGS = {
        'color': 'black',
        'alpha': 0.7,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.5,
    }

    TEXT_KWARGS = {'alpha': 1.0, 'zorder': 500}

    LINESTYLES = ['-', '--', '-', '--']
    EDGELINESTYLE = ['-']
    ALPHAS = [0.9, 0.7, 0.7, 1.0]
    MARKERS = [None]
    LINEWIDTH = [2]
    EDGELINEWIDTH = [0.5]
    MARKERSIZE = 5
    MEC = ['white']
    COLORS = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#f781bf']
    # Default values for facecolor: force same as color
    FACECOLORS = None


class PurpleBlue(DefaultStyle):
    COLORS = ['#810f7c', '#737373', '#045a8d', '#807dba', '#f768a1', '#3690c0']
    FACECOLORS = ['#8c6bb1', '#dadaeb', '#0570b0', '#8f8cd0', '#fcc5c0', '#a6bddb']


class Presentation(DefaultStyle):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.cell_size = 5.0
        # Green, Black/violet, Red, Gray

        colors = ('#4daf4a', '#56124E', '#e31a1c', '#000000', '#D76B00')
        self.color = colors
        self.linestyle = ('-', '-', '-', '--', '-')
        self.linewidth = (2.0, 2.0, 2.0, 2.0, 2.0)
        self.alpha = (0.8, 0.8, 0.8, 0.7, 0.8)
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
        'linewidth': 0.5,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
        'linewidth': 0.35,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        colors = ['#0570b0', 'black', '#e31a1c', '#88419d', '#fc8d59', '#aa5500']

        self.color = colors
        self.linestyle = ['-']
        self.linewidth = 1.0
        self.alpha = [0.8, 0.7, 0.8, 0.8, 0.9, 0.8]


class ColorBrewerStyle(DefaultStyle):
    """
    Style using a selected color palette from colorbrewer2.org.
    Requires the palettable package to be installed from conda or PyPI.
    """

    def __init__(
        self,
        name: str = 'Set1',
        ptype: PaletteType = PaletteType.QUALITATIVE,
        ncolors: int = 5,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        name : str
            Palette name
        ptype : PaletteType
            Palette type (diverging, etc.)
        ncolors: int
            Number of colors. This cannot be arbitrary but has to correspong to the
            available number of colors for a given palette.

        """
        super().__init__(**kwargs)

        import pkgutil

        try:
            import palettable
        except ImportError:
            raise ImportError("Required package 'palettable' is missing")

        obj = pkgutil.resolve_name(f'palettable.colorbrewer.{ptype}.{name}_{ncolors}')

        self.color = list(obj.hex_colors)
        self.facecolor = self.color
        self.linestyle = '-'
        self.linewidth = 1.0
