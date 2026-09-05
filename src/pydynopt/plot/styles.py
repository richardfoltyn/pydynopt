"""
Define classes for plot styles.

Author: Richard Foltyn

This work is licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
"""

from collections.abc import Iterator, Mapping, Sequence
import copy
from copy import deepcopy
import enum
import itertools as it
from typing import Any, ClassVar, overload

from matplotlib.font_manager import FontProperties
import numpy as np

from pydynopt.utils import anything_to_tuple


class FigureLayout(enum.Enum):
    """Figure layout modes."""

    DEFAULT = 1
    TIGHT_LAYOUT = 2
    CONSTRAINED_LAYOUT = 3


_FIGURE_LAYOUT_MAP = {
    'constrained_layout': FigureLayout.CONSTRAINED_LAYOUT,
    'tight_layout': FigureLayout.TIGHT_LAYOUT,
}

# Mapping from keyword arguments for ticklabels() to tick_params()
_TICKLABEL_PARAMS_MAP = {
    'fontfamily': 'labelfontfamily',
    'fontsize': 'labelsize',
    'color': 'labelcolor',
    'rotation': 'labelrotation',
}


class PaletteType(enum.Enum):
    """Palette type classifications."""

    DIVERGING = 0
    QUALITATIVE = 1
    SEQUENTIAL = 2

    def __str__(self) -> str:
        """Return the string representation of the palette type."""
        return self.name.lower()


_DEFAULT_MPL_MAP = {
    'linestyle': 'ls',
    'linewidth': 'lw',
    'markeredgecolor': 'mec',
    'markeredgewidth': 'mew',
    'markerfacecolor': 'mfc',
    'markersize': 'ms',
    'color': 'c',
    'font_properties': 'fontproperties',
    'font': 'fontproperties',
}


class UniqueDict(dict):
    """
    Sub-class of dict which maps long forms of MPL arguments abbreviated forms.

    This ensures that every argument has one unique value
    ("lw" instead of possibly "lw" and "linewidth", etc.).
    """

    def __init__(self, mapping: Mapping[str, str] | None = None, **kwargs: Any) -> None:
        self.mapping: dict[str, str] = dict(mapping) if mapping else {}
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: object) -> None:
        """Set key/value pair mapping keys via internal aliases."""
        key = self.mapping.get(key, key)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> object:
        """Get item mapping key via internal aliases."""
        key = self.mapping.get(key, key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        """Check if key is in the dictionary mapping keys via internal aliases."""
        if isinstance(key, str):
            key = self.mapping.get(key, key)
        return super().__contains__(key)

    def get(self, key: str, default: object = None) -> object:  # type: ignore
        """Get item mapping key via internal aliases, with fallback."""
        key = self.mapping.get(key, key)
        return super().get(key, default)

    def setdefault(self, key: str, default: object = None) -> object:
        """Set default value mapping key via internal aliases."""
        key = self.mapping.get(key, key)
        return super().setdefault(key, default)

    def pop(self, key: str, *args: Any) -> object:  # type: ignore
        """Pop item mapping key via internal aliases."""
        key = self.mapping.get(key, key)
        return super().pop(key, *args)

    def update(self, other: Any = None, **kwargs: Any) -> None:
        """Update dictionary mapping keys via internal aliases."""
        if other is not None:
            if hasattr(other, 'keys'):
                for k in other:
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v


def _to_tuple(value: object) -> tuple[Any] | tuple[Any, ...]:
    """
    Return a tuple created from the given value.

    If `value` is None, return a tuple containing only None.

    Parameters
    ----------
    value
        The value to convert.

    Returns
    -------
    Object(s) wrapped in a tuple.
    """
    if isinstance(value, tuple):
        value = copy.copy(value)
    elif value is None:
        value = (None,)
    else:
        value = anything_to_tuple(value, force=True)

    return value


class Cycler[T]:
    """Wrapper type for style properties that should cycle through a given sequence."""

    def __init__(self, items: Any = None) -> None:
        self.items: tuple[T, ...] = _to_tuple(items)
        self.cache: tuple[T, ...] | None = None

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[T, ...]: ...

    def __getitem__(self, item: int | slice) -> T | tuple[T, ...]:
        """Get item(s) from the cached cycler sequence."""
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

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> 'Cycler[T]':
        """Return a deep copy of the Cycler."""
        if memo is None:
            memo = {}
        obj = type(self)(self.items)
        return obj

    def __repr__(self) -> str:
        """Return string representation."""
        s = f'{self.items}'.strip('()[]')
        s = f'{type(self).__name__}({s})'
        return s


class Colors(Cycler[Any]):
    """Cycler for color sequences."""


class LineStyle(Cycler[str]):
    """Cycler for line style sequences."""


class LineWidth(Cycler[float]):
    """Cycler for line width sequences."""


class Marker(Cycler[Any]):
    """Cycler for marker sequences."""


class Transparency(Cycler[float]):
    """Cycler for transparency/alpha sequences."""


class ConstFillProperty[T]:
    """Property that uses a constant fallback or a sequence of values."""

    def __init__(self, const: T, values: Sequence[T] | None = None) -> None:
        self.values: Sequence[T] | None = copy.copy(values)
        self.const = const

    def __getitem__(self, item: int) -> T:
        """Get value at index or return fallback constant."""
        if self.values is None or item >= len(self.values):
            return self.const
        else:
            return self.values[item]

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> 'ConstFillProperty[T]':
        """Return a deep copy of the ConstFillProperty."""
        if memo is None:
            memo = {}
        obj = ConstFillProperty(self.const, self.values)
        return obj


class PlotStyleDict:
    """Dict-like container for index-based style properties."""

    def __init__(self, style: 'AbstractStyle') -> None:
        self.style = style

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Get a dictionary of plotted style attributes at given index."""
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
        res = {}

        for k in keys:
            res[k] = getattr(self.style, k)[item]

        return res


class StyleAttrMapping:
    """
    Type for style properties that pass several key/value pairs to matplotlib functions.

    The class implements [] such that these key/value pairs can be
    retrieved for a sequence of objects to be plotted.
    """

    def __init__(self, style: 'AbstractStyle', mapping: Mapping[str, object]) -> None:
        self._style = style
        self._mapping = mapping

    def __getitem__(self, item: int) -> dict[str, Any]:
        """
        Return the style defined by key/value pairs at a given index.

        Parameters
        ----------
        item
            The index of the plotted item.

        Returns
        -------
        The style configuration dict.
        """
        result = {}
        for key, attr in self._mapping.items():
            attr = attr if attr is not None else key
            # Handle lists of attributes
            if isinstance(attr, (list, tuple)):
                dct = {}
                for subattr in attr:
                    assert isinstance(subattr, str)
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
    """Abstract base class for all plot styles."""

    LEG_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    LEG_TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    LBL_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    SUPTITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    TEXT_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}
    TEXT_TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {}

    LEG_KWARGS: ClassVar[dict[str, Any]] = {}
    LBL_KWARGS: ClassVar[dict[str, Any]] = {}
    TICKLABEL_KWARGS: ClassVar[dict[str, Any]] = {}
    TICK_PARAMS_KWARGS: ClassVar[dict[str, Any]] = {}
    TITLE_KWARGS: ClassVar[dict[str, Any]] = {}
    SUPTITLE_KWARGS: ClassVar[dict[str, Any]] = {}
    FIGURE_KWARGS: ClassVar[dict[str, Any]] = {}
    SUBPLOT_KWARGS: ClassVar[dict[str, Any]] = {}
    GRID_KWARGS: ClassVar[dict[str, Any]] = {}
    TEXT_KWARGS: ClassVar[dict[str, Any]] = {}
    TEXT_TITLE_KWARGS: ClassVar[dict[str, Any]] = {}
    GUIDELINE_KWARGS: ClassVar[dict[str, Any]] = {}
    CBAR_KWARGS: ClassVar[dict[str, Any]] = {}
    CBAR_TICKLABEL_KWARGS: ClassVar[dict[str, Any]] = {}

    COLORS: ClassVar[list[Any]] = ['black']
    FACECOLORS: ClassVar[list[Any] | None] = ['white']
    LINESTYLES: ClassVar[list[str]] = ['-']
    EDGELINESTYLE: ClassVar[list[str]] = ['-']
    ALPHAS: ClassVar[list[float]] = [1.0]
    EDGEALPHA: ClassVar[list[float]] = [0.3]
    MARKERS: ClassVar[list[Any]] = [None]
    LINEWIDTH: ClassVar[list[float]] = [1.0]
    EDGELINEWIDTH: ClassVar[list[float]] = [0.25]
    MARKERSIZE: ClassVar[Any] = [1.0]
    MEC: ClassVar[list[str]] = ['none']
    MFC: ClassVar[list[Any]] = [None]
    HATCH: ClassVar[list[Any]] = [None]

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
        cls
            Style class whose defaults should be collected.

        Yields
        ------
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
        """
        for name, value in type(self)._iter_class_defaults():
            setattr(self, name, deepcopy(value))

    def __init__(self, fontfamily: str | None = None) -> None:
        self._freeze_class_defaults()

        self.cell_size: float = 6
        self.dpi: float = 96
        self.aspect: float = 1.0
        # Aspect ratio to set via ax.set_aspect()
        # Note: MPL defines aspect as height / width
        self.ax_aspect: float | None = None
        # Aspect ratio to set via ax.set_box_aspect()
        # Note: MPL defines aspect as height / width
        self.ax_box_aspect: float | None = None
        self._margins: float | np.ndarray | None = 0.02
        self._grid: dict[str, object] = self.GRID_KWARGS.copy()
        self._color: Colors | None = None
        self._facecolor: Colors | None = None
        self._facealpha: Transparency | None = None
        self._linewidth: LineWidth | None = None
        self._linestyle: LineStyle | None = None
        self._edgecolor: Colors | None = None
        self._edgelinestyle: LineStyle | None = None
        self._edgelinewidth: LineWidth | None = None
        self._edgealpha: Transparency | None = None
        self._ecolor: Colors | None = None
        self._elinewidth: LineWidth | None = None
        self.capsize: float = 0.0
        self.capthick: float | None = None
        self._alpha: Transparency | None = None
        self._marker: Marker | None = None
        self._markersize: ConstFillProperty[float] | None = None
        self._markevery: ConstFillProperty[Any] | None = None
        self._mec: Colors | None = None
        self._mfc: Colors | None = None
        self._mew: LineWidth | None = None
        self._hatch: ConstFillProperty[Any] | None = None
        self._barmargin: float = 0.0
        self._barwidth: float = 0.8
        self._zorder: ConstFillProperty[float] | None = None
        self._figure: dict[str, object] = self.FIGURE_KWARGS.copy()
        self._subplot: dict[str, object] = self.SUBPLOT_KWARGS.copy()
        self._ylabel: dict[str, object] | None = None
        self._xlabel: dict[str, object] | None = None
        self._xticklabels: dict[str, object] | None = None
        self._xtick_params: dict[str, object] | None = None
        self._yticklabels: dict[str, object] | None = None
        self._ytick_params: dict[str, object] | None = None
        self._title: dict[str, object] | None = None
        self._suptitle: dict[str, object] | None = None
        self._legend: dict[str, object] | None = None
        self._text: dict[str, object] | None = None
        self._text_title: dict[str, object] | None = None
        self._colorbar: dict[str, object] | None = None
        self._cbar_ticklabels: dict[str, object] | None = None
        self.split_scatter: bool = False
        self._guideline: UniqueDict = UniqueDict(
            mapping=_DEFAULT_MPL_MAP, **self.GUIDELINE_KWARGS
        )

        self._plot_all: PlotStyleDict = PlotStyleDict(self)

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

    def __deepcopy__(self, memo: dict[Any, Any] | None = None) -> 'AbstractStyle':
        """Return a deep copy of the style."""
        if memo is None:
            memo = {}
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
                setattr(obj, attr, deepcopy(value, memo))

        # Manually fix this legacy thing, it will otherwise point to wrong object
        obj._plot_all = PlotStyleDict(obj)

        return obj

    @property
    def figure(self) -> dict[str, Any]:
        """Return the figure properties."""
        return self._figure

    @property
    def legend(self) -> dict[str, Any]:
        """Return the legend properties."""
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
    def text(self) -> dict[str, Any]:
        """Return the text properties."""
        if self._text is None:
            fp = FontProperties(**self.TEXT_FONTPROP_KWARGS)
            self._text = self.TEXT_KWARGS.copy()
            self._text.update({'fontproperties': fp})
        return self._text

    @text.setter
    def text(self, value: Mapping[str, object]) -> None:
        self._text = dict(value)

    @property
    def text_title(self) -> dict[str, Any]:
        """Return the text title properties."""
        if self._text_title is None:
            fp = FontProperties(**self.TEXT_TITLE_FONTPROP_KWARGS)
            self._text_title = self.TEXT_TITLE_KWARGS.copy()
            self._text_title.update({'fontproperties': fp})
        return self._text_title

    @text_title.setter
    def text_title(self, value: Mapping[str, object]) -> None:
        self._text_title = dict(value)

    @property
    def title(self) -> dict[str, Any]:
        """Return the title properties."""
        if self._title is None:
            fp = FontProperties(**self.TITLE_FONTPROP_KWARGS)
            self._title = self.TITLE_KWARGS.copy()
            self._title.update({'fontproperties': fp})
        return self._title

    @title.setter
    def title(self, value: Mapping[str, object]) -> None:
        self._title = dict(value)

    @property
    def suptitle(self) -> dict[str, Any]:
        """Return the suptitle properties."""
        if self._suptitle is None:
            fp = FontProperties(**self.SUPTITLE_FONTPROP_KWARGS)
            self._suptitle = self.SUPTITLE_KWARGS.copy()
            self._suptitle.update({'fontproperties': fp})
        return self._suptitle

    @suptitle.setter
    def suptitle(self, value: Mapping[str, object]) -> None:
        self._suptitle = dict(value)

    @property
    def xlabel(self) -> dict[str, Any]:
        """Return the xlabel properties."""
        if self._xlabel is None:
            fp = FontProperties(**self.LBL_FONTPROP_KWARGS)
            self._xlabel = self.LBL_KWARGS.copy()
            self._xlabel.update({'fontproperties': fp})
        return self._xlabel

    @xlabel.setter
    def xlabel(self, value: Mapping[str, object]) -> None:
        self._xlabel = dict(value)

    @property
    def ylabel(self) -> dict[str, Any]:
        """Return the ylabel properties."""
        if self._ylabel is None:
            fp = FontProperties(**self.LBL_FONTPROP_KWARGS)
            self._ylabel = self.LBL_KWARGS.copy()
            self._ylabel.update({'fontproperties': fp})
        return self._ylabel

    @ylabel.setter
    def ylabel(self, value: Mapping[str, object]) -> None:
        self._ylabel = dict(value)

    @property
    def xticklabels(self) -> dict[str, Any]:
        """Return the xticklabels properties."""
        if self._xticklabels is None:
            self._xticklabels = self.TICKLABEL_KWARGS.copy()
        return self._xticklabels

    @xticklabels.setter
    def xticklabels(self, value: Mapping[str, object]) -> None:
        self._xticklabels = dict(value)

    @property
    def yticklabels(self) -> dict[str, Any]:
        """Return the yticklabels properties."""
        if self._yticklabels is None:
            self._yticklabels = self.TICKLABEL_KWARGS.copy()
        return self._yticklabels

    @yticklabels.setter
    def yticklabels(self, value: Mapping[str, object]) -> None:
        self._yticklabels = dict(value)

    @property
    def xtick_params(self) -> dict[str, Any]:
        """Return collection of keyword arguments that can be passed to set_tick_params()."""
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
        self._xtick_params = dict(value)

    @property
    def ytick_params(self) -> dict[str, Any]:
        """Return collection of keyword arguments that can be passed to set_tick_params()."""
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
        self._ytick_params = dict(value)

    @property
    def grid(self) -> dict[str, Any]:
        """Return the grid properties."""
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
    def subplot(self) -> dict[str, Any]:
        """Return the subplot properties."""
        if self._subplot is None:
            self._subplot = self.SUBPLOT_KWARGS.copy()
        return self._subplot

    @property
    def colorbar(self) -> dict[str, Any]:
        """Return the colorbar properties."""
        if self._colorbar is None:
            self._colorbar = self.CBAR_KWARGS.copy()
        return self._colorbar

    @colorbar.setter
    def colorbar(self, value: Mapping[str, object]) -> None:
        self._colorbar = dict(value)

    @property
    def cbar_ticklabels(self) -> dict[str, Any]:
        """Return the colorbar ticklabel properties."""
        if self._cbar_ticklabels is None:
            self._cbar_ticklabels = self.CBAR_TICKLABEL_KWARGS.copy()
        return self._cbar_ticklabels

    @cbar_ticklabels.setter
    def cbar_ticklabels(self, value: Mapping[str, object]) -> None:
        self._cbar_ticklabels = dict(value)

    @property
    def color(self) -> Colors:
        """Return the line/fill color cycler."""
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
        """Return the edge color cycler."""
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
        """Return the face color cycler."""
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
        """Return the face alpha/transparency cycler."""
        if self._facealpha is None:
            self._facealpha = Transparency(self.ALPHAS)
        return self._facealpha

    @facealpha.setter
    def facealpha(self, value: object) -> None:
        if isinstance(value, Transparency):
            self._facealpha = deepcopy(value)
        else:
            self._facealpha = Transparency(value)

    @property
    def linewidth(self) -> LineWidth:
        """Return the line width cycler."""
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
        """Return the line width cycler."""
        return self.linewidth

    @property
    def edgelinewidth(self) -> LineWidth:
        """Return the edge line width cycler."""
        if self._edgelinewidth is None:
            self._edgelinewidth = LineWidth(self.EDGELINEWIDTH)
        return self._edgelinewidth

    @edgelinewidth.setter
    def edgelinewidth(self, value: object) -> None:
        if isinstance(value, LineWidth):
            self._edgelinewidth = deepcopy(value)
        else:
            value = _to_tuple(value)
            self._edgelinewidth = LineWidth(value)

    @property
    def linestyle(self) -> LineStyle:
        """Return the line style cycler."""
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
        """Return the line style cycler."""
        return self.linestyle

    @property
    def edgelinestyle(self) -> LineStyle:
        """Return the edge line style cycler."""
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
        """Return the edge alpha/transparency cycler."""
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
        """Return the alpha/transparency cycler."""
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
        """Return the errorbar color cycler."""
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
        """Return the errorbar line width cycler."""
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
        """Return the marker cycler."""
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
    def markersize(self) -> ConstFillProperty[float]:
        """Return the markersize cycler."""
        if self._markersize is None:
            self._markersize = ConstFillProperty(const=self.MARKERSIZE)
        return self._markersize

    @markersize.setter
    def markersize(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            val: Any = deepcopy(value)
            self._markersize = val
        else:
            value = _to_tuple(value)
            default = value[-1] if value else 0.0
            self._markersize = ConstFillProperty(default, value)

    @property
    def markevery(self) -> ConstFillProperty[Any]:
        """Return the markevery cycler."""
        if self._markevery is None:
            self._markevery = ConstFillProperty(const=1)
        return self._markevery

    @markevery.setter
    def markevery(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            val: Any = deepcopy(value)
            self._markevery = val
        else:
            value = _to_tuple(value)
            default = value[-1]
            self._markevery = ConstFillProperty(default, value)

    @property
    def mec(self) -> Colors:
        """Return the marker edge color cycler."""
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
        """Return the marker face color cycler."""
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
        """Return the marker edge width cycler."""
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
    def hatch(self) -> ConstFillProperty[Any]:
        """Return the hatch cycler."""
        if self._hatch is None:
            self._hatch = ConstFillProperty(None)
        return self._hatch

    @hatch.setter
    def hatch(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            val: Any = deepcopy(value)
            self._hatch = val
        else:
            value = _to_tuple(value)
            default = None
            self._hatch = ConstFillProperty(default, value)

    @property
    def barmargin(self) -> float:
        """Return the bar margin value."""
        return self._barmargin

    @barmargin.setter
    def barmargin(self, value: float | np.floating[Any]) -> None:
        try:
            value = float(value)
        except TypeError as err:
            raise ValueError('Margin must be float!') from err

        if value < 0.0 or value >= 0.5:
            raise ValueError('Margin value must be in [0, 0.5)')

        self._barmargin = value

    @property
    def barwidth(self) -> float:
        """Return the width of the all bars within a `by` group."""
        return self._barwidth

    @barwidth.setter
    def barwidth(self, value: float | np.floating[Any]) -> None:
        try:
            value = float(value)
        except TypeError as err:
            raise ValueError('Width must be float!') from err

        if value <= 0.0 or value > 1.0:
            raise ValueError('Width value must be in (0, 1]')

        self._barwidth = value

    @property
    def zorder(self) -> ConstFillProperty[float]:
        """Return the zorder cycler."""
        if self._zorder is None:
            self._zorder = ConstFillProperty(const=10)
        return self._zorder

    @zorder.setter
    def zorder(self, value: object) -> None:
        if isinstance(value, ConstFillProperty):
            val: Any = deepcopy(value)
            self._zorder = val
        else:
            value = _to_tuple(value)
            self._zorder = ConstFillProperty(10, value)

    @property
    def guideline(self) -> dict[str, Any]:
        """Return the guideline dictionary."""
        return self._guideline

    @guideline.setter
    def guideline(self, value: Mapping[str, object] | None = None) -> None:
        if value is not None:
            self._guideline = UniqueDict(mapping=_DEFAULT_MPL_MAP, **value)
        else:
            self._guideline = UniqueDict(mapping=_DEFAULT_MPL_MAP)

    @property
    def margins(self) -> float | np.ndarray | None:
        """
        Return the subplot margins.

        Margins are relative to the data margins and need to be in the interval [0, 1].
        If multiple values are given, they are in the order (left, bottom, right, top).
        If None, scaling or margins is disabled.
        """
        return self._margins

    @margins.setter
    def margins(self, value: float | Sequence[float] | np.ndarray | None) -> None:
        if value is not None:
            if isinstance(value, (int, float, np.floating)):
                value = float(value)
            else:
                try:
                    arr = np.atleast_1d(value)
                    if arr.size != 1 and arr.size != 4:
                        raise ValueError('margins value not understood') from None
                    # Store as float since it's the same value for all sides
                    value = float(arr[0]) if np.all(arr[0] == arr[1:]) else arr
                except (TypeError, ValueError) as inner_err:
                    raise ValueError('margins value not understood') from inner_err

        self._margins = value

    @property
    def figure_layout(self) -> FigureLayout:
        """Return the figure layout setting."""
        if not self._figure:
            return FigureLayout.DEFAULT

        for k, v in _FIGURE_LAYOUT_MAP.items():
            if self._figure.get(k, False):
                return v
        return FigureLayout.DEFAULT

    @figure_layout.setter
    def figure_layout(self, value: FigureLayout) -> None:
        if not isinstance(value, FigureLayout):
            raise TypeError('Argument must be of FigureLayout type')

        # Delete all layout entries, add (back) only the one requested by caller

        for k in _FIGURE_LAYOUT_MAP:
            if k in self._figure:
                del self._figure[k]

        for k, v in _FIGURE_LAYOUT_MAP.items():
            if value == v:
                self._figure[k] = True

    @property
    def plot_kwargs(self) -> PlotStyleDict:
        """Return the plot style dictionary."""
        return self._plot_all

    @property
    def fill_between_kwargs(self) -> StyleAttrMapping:
        """Return key/value pairs for matplotlib's fill_between()."""
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
        """Return key/value pairs for fill_between() face component."""
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
        """Return key/value pairs for fill_between() edge component."""
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
        """Return key/value pairs for matplotlib's errorbar()."""
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
        """Return key/value pairs for errorbar() without markers."""
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
        """Return key/value pairs for marker without connecting lines."""
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
        """Return key/value pairs for matplotlib's bar()."""
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
        """Return key/value pairs for matplotlib's scatter()."""
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
        """Return key/value pairs for scatter() face component."""
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
        """Return key/value pairs for scatter() edge component."""
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
    """Default plot style setting default values."""

    LEG_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {'family': 'serif', 'size': 'small'}
    LEG_TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'size': 'small',
        'weight': 'bold',
    }

    LBL_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'size': 'medium',
    }

    TICKLABEL_KWARGS: ClassVar[dict[str, Any]] = {
        'fontfamily': 'serif',
        'fontsize': 'small',
    }

    TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'size': 'medium',
        'style': 'italic',
    }

    SUPTITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'size': 'x-large',
        'style': 'italic',
        'weight': 'semibold',
    }

    TEXT_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'style': 'italic',
        'size': 'small',
    }

    TEXT_TITLE_FONTPROP_KWARGS: ClassVar[dict[str, Any]] = {
        'family': 'serif',
        'style': 'italic',
        'size': 'medium',
    }

    GUIDELINE_KWARGS: ClassVar[dict[str, Any]] = {
        'lw': 0.75,
        'ls': (0, (1, 1)),
        'alpha': 0.7,
        'color': 'black',
        'zorder': -10,
    }

    # Keyword arguments (other than font properties) for various objects
    LEG_KWARGS: ClassVar[dict[str, Any]] = {
        'framealpha': 0.7,
        'frameon': True,
        'fancybox': False,
    }

    LBL_KWARGS: ClassVar[dict[str, Any]] = {}
    TITLE_KWARGS: ClassVar[dict[str, Any]] = {}
    SUPTITLE_KWARGS: ClassVar[dict[str, Any]] = {}

    SUBPLOT_KWARGS: ClassVar[dict[str, Any]] = {'facecolor': 'white', 'axisbelow': True}

    FIGURE_KWARGS: ClassVar[dict[str, Any]] = {'constrained_layout': True}

    GRID_KWARGS: ClassVar[dict[str, Any]] = {
        'color': 'black',
        'alpha': 0.7,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.5,
    }

    TEXT_KWARGS: ClassVar[dict[str, Any]] = {'alpha': 1.0, 'zorder': 500}

    TEXT_TITLE_KWARGS: ClassVar[dict[str, Any]] = {'alpha': 1.0, 'zorder': 1000}

    CBAR_KWARGS: ClassVar[dict[str, Any]] = {}
    CBAR_TICKLABEL_KWARGS: ClassVar[dict[str, Any]] = {
        'fontfamily': 'serif',
        'fontsize': 'small',
    }

    LINESTYLES: ClassVar[list[str]] = ['-', '--', '-', '--']
    EDGELINESTYLE: ClassVar[list[str]] = ['-']
    ALPHAS: ClassVar[list[float]] = [0.9, 0.7, 0.7, 1.0]
    MARKERS: ClassVar[list[Any]] = [None]
    LINEWIDTH: ClassVar[list[float]] = [2]
    EDGELINEWIDTH: ClassVar[list[float]] = [0.5]
    MARKERSIZE: ClassVar[Any] = 5
    MEC: ClassVar[list[str]] = ['white']
    COLORS: ClassVar[list[str]] = [
        '#377eb8',
        '#e41a1c',
        '#4daf4a',
        '#ff7f00',
        '#f781bf',
    ]
    # Default values for facecolor: force same as color
    FACECOLORS: ClassVar[list[str] | None] = None


class PurpleBlue(DefaultStyle):
    """Purple-blue theme style."""

    COLORS: ClassVar[list[str]] = [
        '#810f7c',
        '#737373',
        '#045a8d',
        '#807dba',
        '#f768a1',
        '#3690c0',
    ]
    FACECOLORS: ClassVar[list[str] | None] = [
        '#8c6bb1',
        '#dadaeb',
        '#0570b0',
        '#8f8cd0',
        '#fcc5c0',
        '#a6bddb',
    ]


class Presentation(DefaultStyle):
    """Presentation theme style."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.cell_size: float = 5.0
        # Green, Black/violet, Red, Gray

        colors = ('#4daf4a', '#56124E', '#e31a1c', '#000000', '#D76B00')
        self.color = colors
        self.linestyle = ('-', '-', '-', '--', '-')
        self.linewidth = (2.0, 2.0, 2.0, 2.0, 2.0)
        self.alpha = (0.8, 0.8, 0.8, 0.7, 0.8)
        self.marker = (None, 'p', 'o', None, 'd')
        self._mec: Colors | None = Colors((None, 'White', 'White', None, 'White'))


class AlternatingStyle(DefaultStyle):
    """
    Alternating solid and black lines.

    This style alternates solid colored lines with black lines
    with dashed/dotted/etc. line styles.
    """

    GRID_KWARGS: ClassVar[dict[str, Any]] = {
        'color': 'black',
        'alpha': 0.5,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.5,
    }

    def __init__(self, **kwargs: Any) -> None:
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

        colors = it.chain(*zip(colors, black, strict=False))
        ls = it.chain(*zip(ls_colors, ls_black, strict=False))
        lw = it.chain(*zip(lw_colors, lw_black, strict=False))
        alpha = it.chain(*zip(alpha_color, alpha_black, strict=False))
        markers = it.chain(*zip(markers_color, markers_black, strict=False))

        self.color = colors
        self.linestyle = ls
        self.linewidth = lw
        self.alpha = alpha
        self.marker = markers


class QualitativeStyle(DefaultStyle):
    """
    Qualitative style theme.

    This style uses identical line styles but alternating colors,
    similar to the qualitative color schemes on colorbrewer2.org.
    """

    GRID_KWARGS: ClassVar[dict[str, Any]] = {
        'color': 'black',
        'alpha': 0.5,
        'zorder': -1000,
        'linestyle': ':',
        'linewidth': 0.35,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        colors = ['#0570b0', 'black', '#e31a1c', '#88419d', '#fc8d59', '#aa5500']

        self.color = colors
        self.linestyle = ['-']
        self.linewidth = 1.0
        self.alpha = [0.8, 0.7, 0.8, 0.8, 0.9, 0.8]


class ColorBrewerStyle(DefaultStyle):
    """
    ColorBrewer style theme.

    This style uses a selected color palette from colorbrewer2.org.
    It requires the palettable package to be installed from conda or PyPI.
    """

    def __init__(
        self,
        name: str = 'Set1',
        ptype: PaletteType = PaletteType.QUALITATIVE,
        ncolors: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize a colorbrewer2.org style palette."""
        super().__init__(**kwargs)

        import importlib.util
        import pkgutil

        if importlib.util.find_spec('palettable') is None:
            raise ImportError("Required package 'palettable' is missing")

        obj = pkgutil.resolve_name(f'palettable.colorbrewer.{ptype}.{name}_{ncolors}')

        self.color = list(obj.hex_colors)
        self.facecolor = self.color
        self.linestyle = '-'
        self.linewidth = 1.0
