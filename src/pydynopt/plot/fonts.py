"""
Matplotlib font and TeX rendering configuration utilities.

This module centralizes all logic that determines how text and math are rendered
in plots across projects that use ``pydynopt.plot``.

How the system is organized
---------------------------
The design separates rendering settings from selection logic:

1. ``MplFontRenderConfig`` stores one concrete rendering profile.
    A profile defines:
    - the Matplotlib family category (for example, ``serif`` or
      ``sans-serif``),
    - an ordered list of concrete font family candidates,
    - an optional LaTeX preamble for ``text.usetex=True``, and
    - optional mathtext rcParams for ``text.usetex=False``.

2. ``MplFontConfig`` groups two profiles into one full policy:
    - preferred render profile (for ``text.usetex=True``), and
    - fallback profile (for ``text.usetex=False`` or unavailable TeX).
    It also stores TeX dependency requirements (required ``.sty`` files and
    executable names for ``latex`` and ``kpsewhich``).

How profile selection works at runtime
--------------------------------------
Callers provide ``usetex`` to ``configure_mpl_fonts`` together with an
``MplFontConfig`` instance.

The selection flow is:

1. If ``usetex`` is ``False``, the function always selects a fallback profile
    for non-TeX rendering.

2. If ``usetex`` is ``True``, the function checks TeX availability through
    ``is_usetex_available``.

3. TeX availability is determined by:
    - verifying that both executables are on ``PATH`` via ``shutil.which``, and
    - probing every required style file through ``kpsewhich``.
    Results are cached by ``_usetex_is_available_cached`` to avoid repeated
    subprocess calls for the same dependency tuple.

4. If dependencies are present, the preferred render profile is used.
    If dependencies are missing, a ``RuntimeWarning`` is emitted and the
    fallback profile is used.

How rcParams are applied
------------------------
The private helper ``_apply_profile`` writes the selected profile into
``matplotlib.rcParams``:

- always sets ``text.usetex``, ``font.family``, and ``font.<family>``,
- adds ``text.latex.preamble`` only in TeX mode when provided, and
- merges ``mathtext`` rcParams only in fallback mode when provided.

The module intentionally does not reset rcParams; it only applies the selected
policy. Callers are expected to invoke configuration before figure creation.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import shutil
import subprocess
from typing import Any
import warnings

import matplotlib as mpl
import matplotlib.font_manager as fm

_DEFAULT_USETEX_REQUIRED_STY = ('type1cm.sty', 'type1ec.sty')


@dataclass(frozen=True)
class MplFontRenderConfig:
    """
    Rendering settings for a single Matplotlib text mode/profile.

    Parameters
    ----------
    family
        Matplotlib font family category (typically ``'serif'`` or
        ``'sans-serif'``).
    families
        Ordered list of family candidates for ``font.<family>``.
    latex_preamble
        LaTeX preamble used when ``text.usetex=True``.
    mathtext
        Additional rcParams used when ``text.usetex=False``.
    """

    family: str
    families: tuple[str, ...]
    latex_preamble: str | None = None
    mathtext: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MplFontConfig:
    """
    Full font configuration for one plotting target.

    Parameters
    ----------
    render
        Preferred profile for ``text.usetex=True``.
    fallback
        Fallback profile for ``text.usetex=False`` or unavailable TeX.
    required_sty
        TeX style files required by Matplotlib's usetex backend.
    latex_cmd
        Name/path of the LaTeX executable.
    kpsewhich_cmd
        Name/path of ``kpsewhich`` executable.
    """

    render: MplFontRenderConfig
    fallback: MplFontRenderConfig
    required_sty: tuple[str, ...] = _DEFAULT_USETEX_REQUIRED_STY
    latex_cmd: str = 'latex'
    kpsewhich_cmd: str = 'kpsewhich'


def select_font(font_family: str | Sequence[str], default: str = 'serif') -> str:
    """
    Select a font family from installed system fonts.

    Parameters
    ----------
    font_family
        Preferred font family or ordered list of candidates.
    default
        Fallback family when none of the candidates are available.

    Returns
    -------
    Selected font family.
    """
    available_fonts = [font.name for font in fm.fontManager.ttflist]
    if isinstance(font_family, str):
        return font_family if font_family in available_fonts else default

    for family in font_family:
        if family in available_fonts:
            return family

    return default


@lru_cache(maxsize=32)
def _usetex_is_available_cached(
    required_sty: tuple[str, ...],
    latex_cmd: str,
    kpsewhich_cmd: str,
) -> bool:
    if shutil.which(latex_cmd) is None or shutil.which(kpsewhich_cmd) is None:
        return False

    for sty in required_sty:
        result = subprocess.run(
            [kpsewhich_cmd, sty],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False

    return True


def is_usetex_available(config: MplFontConfig) -> bool:
    """
    Check whether Matplotlib's usetex pipeline is available.

    Parameters
    ----------
    config
        Font configuration with TeX dependency settings.
    """
    return _usetex_is_available_cached(
        tuple(config.required_sty),
        config.latex_cmd,
        config.kpsewhich_cmd,
    )


def _apply_profile(profile: MplFontRenderConfig, usetex: bool) -> None:
    rc: dict[str, Any] = {
        'text.usetex': usetex,
        'font.family': profile.family,
        f'font.{profile.family}': list(profile.families),
    }

    if usetex:
        if profile.latex_preamble:
            rc['text.latex.preamble'] = profile.latex_preamble
    elif profile.mathtext:
        rc.update(dict(profile.mathtext))

    mpl.rcParams.update(rc)


def configure_mpl_fonts(
    *,
    usetex: bool,
    config: MplFontConfig,
) -> None:
    """
    Configure Matplotlib rcParams for text/math rendering.

    Parameters
    ----------
    usetex
        Request LaTeX-based text rendering.
    config
        Complete rendering configuration.
    """
    # Use Type 42 (TrueType) fonts instead of Type 3 to reduce file size
    # and improve searchability in PDFs.
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    if not usetex:
        _apply_profile(config.fallback, usetex=False)
        return

    if is_usetex_available(config):
        _apply_profile(config.render, usetex=True)
        return

    warnings.warn(
        (
            'LaTeX dependencies for Matplotlib usetex are unavailable '
            '(missing required .sty files). Falling back '
            'to mathtext.'
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    _apply_profile(config.fallback, usetex=False)
