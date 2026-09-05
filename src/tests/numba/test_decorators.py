"""Test Numba decorator wrappers and pure-Python substitutes."""

import subprocess
import sys
from typing import Any

from numba import njit
from numba.core.errors import TypingError
import pytest

from pydynopt.numba import overload


def _relaxed_target(x: float) -> float:
    return x


@overload(_relaxed_target)
def _overload_relaxed_target(x: Any) -> Any:
    def impl(x: int) -> int:
        return x + 1

    return impl


@njit
def _call_relaxed_target(x: int) -> float:
    return _relaxed_target(x)


def _strict_target(x: float) -> float:
    return x


@overload(_strict_target, strict=True)
def _overload_strict_target(x: Any) -> Any:
    def impl(x: int) -> int:
        return x + 1

    return impl


@njit
def _call_strict_target(x: int) -> float:
    return _strict_target(x)


def test_overload_defaults_to_relaxed_signature_matching() -> None:
    assert _call_relaxed_target(1) == 2
    assert _call_relaxed_target.nopython_signatures


def test_overload_forwards_explicit_strict_mode() -> None:
    with pytest.raises(TypingError):
        _call_strict_target(1)


def test_no_numba_decorators_and_interpolation_in_clean_process() -> None:
    code = """
import numpy as np
import pydynopt
pydynopt.use_numba = False
from pydynopt.interpolate import (
    interp1d, interp1d_eval, interp1d_locate,
    interp2d, interp2d_eval, interp2d_locate,
)
from pydynopt.numba.dummy import (
    jit_dummy, jitclass_dummy, overload_dummy, register_jitable_dummy,
)

def func(x):
    return x + 1

class Example:
    pass

assert jit_dummy(func) is func
assert jit_dummy()(func) is func
assert jit_dummy(nopython=True)(func) is func
assert jitclass_dummy(Example) is Example
assert jitclass_dummy()(Example) is Example
assert jitclass_dummy({})(Example) is Example
assert overload_dummy(func)(func) is func
assert overload_dummy(func, strict=False)(func) is func
assert register_jitable_dummy(func) is func
assert register_jitable_dummy()(func) is func
assert register_jitable_dummy(inline='always')(func) is func

xp = np.array([0.0, 1.0])
fp = np.array([0.0, 2.0])
index, weight = interp1d_locate(np.array([0.5]), xp)
assert np.allclose(interp1d_eval(index, weight, fp), [1.0])
assert interp1d(0.5, xp, fp) == 1.0
fp2 = np.array([[0.0, 2.0], [1.0, 3.0]])
index2, weight2 = interp2d_locate(0.5, 0.5, xp, xp)
assert interp2d_eval(index2, weight2, fp2) == 1.5
assert interp2d(0.5, 0.5, xp, xp, fp2) == 1.5
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
