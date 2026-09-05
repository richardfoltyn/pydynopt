"""
Unit tests for plot styles and UniqueDict.

Author: Richard Foltyn
"""

from pydynopt.plot.styles import _DEFAULT_MPL_MAP, UniqueDict


def test_unique_dict_mpl_map() -> None:
    """Verify that UniqueDict maps attributes correctly in all major dictionary methods."""
    d = UniqueDict(_DEFAULT_MPL_MAP)

    # 1. __setitem__ & __getitem__
    d['linewidth'] = 2.5
    assert d['lw'] == 2.5
    assert d['linewidth'] == 2.5
    assert 'linewidth' in d
    assert 'lw' in d

    # 2. __init__ with kwargs
    d2 = UniqueDict(_DEFAULT_MPL_MAP, linestyle='--', color='blue')
    assert d2['ls'] == '--'
    assert d2['linestyle'] == '--'
    assert d2['c'] == 'blue'
    assert d2['color'] == 'blue'

    # 3. get()
    assert d2.get('linestyle') == '--'
    assert d2.get('ls') == '--'
    assert d2.get('nonexistent', 'default') == 'default'

    # 4. update()
    d2.update(linewidth=3.0)
    assert d2['lw'] == 3.0
    d2.update({'markeredgecolor': 'red'})
    assert d2['mec'] == 'red'

    # 5. pop()
    val = d2.pop('color')
    assert val == 'blue'
    assert 'c' not in d2
    assert 'color' not in d2

    # 6. setdefault()
    d2.setdefault('markerfacecolor', 'green')
    assert d2['mfc'] == 'green'


def test_font_properties_mapping() -> None:
    """Verify font_properties and font map to fontproperties."""
    d = UniqueDict(_DEFAULT_MPL_MAP)

    d['font_properties'] = 'serif'
    assert d['fontproperties'] == 'serif'
    assert d['font'] == 'serif'
    assert d['font_properties'] == 'serif'

    d.update(font='sans-serif')
    assert d['fontproperties'] == 'sans-serif'
    assert d['font'] == 'sans-serif'
    assert d['font_properties'] == 'sans-serif'


def test_barwidth() -> None:
    """Verify that barwidth can be get and set correctly, with proper validation."""
    import pytest

    from pydynopt.plot.styles import DefaultStyle

    style = DefaultStyle()
    assert style.barwidth == 0.8

    # Test valid values
    style.barwidth = 0.5
    assert style.barwidth == 0.5
    style.barwidth = 1.0
    assert style.barwidth == 1.0
    style.barwidth = 0.0001
    assert style.barwidth == 0.0001

    # Test invalid values (must satisfy 0 < value <= 1)
    with pytest.raises(ValueError):
        style.barwidth = 0.0

    with pytest.raises(ValueError):
        style.barwidth = -0.5

    with pytest.raises(ValueError):
        style.barwidth = 1.001

    # Test invalid type
    with pytest.raises(ValueError):
        style.barwidth = None  # type: ignore
