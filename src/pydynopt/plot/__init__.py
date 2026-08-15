"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from .baseplots import plot_grid
from .fonts import (
    MplFontConfig,
    MplFontRenderConfig,
    configure_mpl_fonts,
    is_usetex_available,
    select_font,
)
from .ndarraylattice import NDArrayLattice
from .plotmap import PlotMap, plot_pm
from .styles import (
    AbstractStyle,
    AlternatingStyle,
    ColorBrewerStyle,
    DefaultStyle,
    Presentation,
    PurpleBlue,
    QualitativeStyle,
)
