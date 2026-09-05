"""Provide optional Numba integration with pure-Python fallbacks.

- Export Numba decorators and scalar types when Numba is enabled.
- Export no-op substitutes otherwise.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Mapping
from typing import Any, overload as typing_overload

from .dummy import (
    boolean,
    float32,
    float64,
    from_dtype,
    int8,
    int16,
    int32,
    int64,
    jit as _jit,
    jitclass as _jitclass,
    overload as _overload,
    prange,
    register_jitable as _register_jitable,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)

# Numba and fallback implementations have unrelated runtime types.
_jit: Any
_jitclass: Any
_overload: Any
_register_jitable: Any
boolean: Any
float32: Any
float64: Any
int8: Any
int16: Any
int32: Any
int64: Any
string: Any
uint8: Any
uint16: Any
uint32: Any
uint64: Any

has_numba: bool = False
JIT_OPTIONS: dict[str, Any] = {}
JIT_OPTIONS_INLINE: dict[str, Any] = {}

from pydynopt import use_numba

if use_numba:
    try:
        from numba import from_dtype, jit as _jit, prange, types as _types
        from numba.experimental import jitclass as _jitclass
        from numba.extending import (
            overload as _overload,
            register_jitable as _register_jitable,
        )

        boolean = _types.boolean
        float32 = _types.float32
        float64 = _types.float64
        int8 = _types.int8
        int16 = _types.int16
        int32 = _types.int32
        int64 = _types.int64
        string = _types.string
        uint8 = _types.uint8
        uint16 = _types.uint16
        uint32 = _types.uint32
        uint64 = _types.uint64

        has_numba = True

        JIT_OPTIONS = {'nopython': True, 'nogil': True, 'parallel': False}
        JIT_OPTIONS_INLINE = {
            'nopython': True,
            'nogil': True,
            'parallel': False,
            'inline': 'always',
        }

    except ImportError:
        # Use the default decorators and types imported above.
        pass


@typing_overload
def jit[F: Callable[..., Any]](
    signature_or_function: F,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> F: ...


@typing_overload
def jit[F: Callable[..., Any]](
    signature_or_function: Any = None,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> Callable[[F], F]: ...


def jit(
    signature_or_function: Any = None,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> Any:
    """Apply the active JIT decorator without changing static callable types.

    Parameters
    ----------
    signature_or_function
        Function or Numba signature passed to the active implementation.
    *jit_args
        Positional arguments passed to the active implementation.
    **jit_kwargs
        Keyword arguments passed to the active implementation.

    Returns
    -------
    Decorated callable or decorator returned by the active implementation.
    """
    return _jit(signature_or_function, *jit_args, **jit_kwargs)


@typing_overload
def jitclass[T](cls_or_spec: type[T], spec: Any = None) -> type[T]: ...


@typing_overload
def jitclass[T](
    cls_or_spec: Any = None, spec: Any = None
) -> Callable[[type[T]], type[T]]: ...


def jitclass(cls_or_spec: Any = None, spec: Any = None) -> Any:
    """Apply the active JIT-class decorator without changing static class types.

    Parameters
    ----------
    cls_or_spec
        Class to decorate or Numba class specification.
    spec
        Numba class specification when a class is supplied directly.

    Returns
    -------
    Decorated class or decorator supplied by the active implementation.
    """
    return _jitclass(cls_or_spec, spec)


def overload[F: Callable[..., Any]](
    func: Callable[..., Any],
    jit_options: Mapping[str, Any] | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Create an overload decorator that preserves the implementation type.

    Parameters
    ----------
    func
        Function for which an overload is registered.
    jit_options
        JIT options passed to the active implementation.
    strict
        Whether argument and implementation signatures must match strictly.
        Defaults to ``False`` because Python annotations are not Numba dispatch
        signatures.
    **kwargs
        Additional options passed to the active implementation.

    Returns
    -------
    Decorator supplied by the active implementation.
    """
    active_overload: Any = _overload
    options = {} if jit_options is None else jit_options
    return active_overload(
        func,
        jit_options=options,
        strict=strict,
        **kwargs,
    )


@typing_overload
def register_jitable[F: Callable[..., Any]](func: F, /) -> F: ...


@typing_overload
def register_jitable[F: Callable[..., Any]](
    *args: Any, **kwargs: Any
) -> Callable[[F], F]: ...


def register_jitable(*args: Any, **kwargs: Any) -> Any:
    """Register a callable while preserving its static type.

    Parameters
    ----------
    *args
        Positional arguments passed to the active implementation.
    **kwargs
        Keyword arguments passed to the active implementation.

    Returns
    -------
    Registered callable or decorator returned by the active implementation.
    """
    return _register_jitable(*args, **kwargs)
