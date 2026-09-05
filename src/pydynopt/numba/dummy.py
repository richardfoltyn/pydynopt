"""Define pure-Python substitutes for the optional Numba API.

- Provide no-op decorator replacements.
- Provide subscriptable stand-ins for Numba scalar types.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Mapping
from typing import Any, Self, overload as typing_overload

__all__ = [
    'boolean',
    'float32',
    'float64',
    'from_dtype',
    'int8',
    'int16',
    'int32',
    'int64',
    'jit',
    'jitclass',
    'overload',
    'prange',
    'register_jitable',
    'string',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
]


@typing_overload
def jit_dummy[F: Callable[..., Any]](
    signature_or_function: F,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> F: ...


@typing_overload
def jit_dummy[F: Callable[..., Any]](
    signature_or_function: Any = None,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> Callable[[F], F]: ...


def jit_dummy(
    signature_or_function: Any = None,
    *jit_args: Any,
    **jit_kwargs: Any,
) -> Any:
    """Return a function unchanged or create a no-op JIT decorator.

    Parameters
    ----------
    signature_or_function
        Function to return or a signature accepted for API compatibility.
    *jit_args
        Positional JIT options, which are ignored.
    **jit_kwargs
        Keyword JIT options, which are ignored.

    Returns
    -------
    The original function or a decorator that returns its function unchanged.
    """
    if signature_or_function is None or isinstance(signature_or_function, list):
        pyfunc = None
    else:
        pyfunc = signature_or_function

    if pyfunc is not None:
        return pyfunc

    def decorate[F: Callable[..., Any]](func: F) -> F:
        """Return the decorated function unchanged."""
        return func

    return decorate


def jitclass_dummy[T](spec: Any) -> Callable[[type[T]], type[T]]:
    """Create a no-op replacement for Numba's ``jitclass`` decorator.

    Parameters
    ----------
    spec
        JIT class specification, which is ignored.

    Returns
    -------
    A decorator that returns its class unchanged.
    """

    def decorate(cls: type[T]) -> type[T]:
        """Return the decorated class unchanged."""
        return cls

    return decorate


def overload_dummy[F: Callable[..., Any]](
    func: Callable[..., Any],
    jit_options: Mapping[str, Any] | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Create a no-op replacement for Numba's ``overload`` decorator.

    Parameters
    ----------
    func
        Function for which an overload would be registered.
    jit_options
        JIT options, which are ignored.
    strict
        Strictness flag, which is ignored.
    **kwargs
        Additional overload options, which are ignored.

    Returns
    -------
    A decorator that returns the overload function unchanged.
    """

    def decorate(overload_func: F) -> F:
        """Return the overload function unchanged."""
        return overload_func

    return decorate


@typing_overload
def register_jitable_dummy[F: Callable[..., Any]](func: F, /) -> F: ...


@typing_overload
def register_jitable_dummy[F: Callable[..., Any]](
    *args: Any, **kwargs: Any
) -> Callable[[F], F]: ...


def register_jitable_dummy(*args: Any, **kwargs: Any) -> Any:
    """Create a pure-Python replacement for ``register_jitable``.

    Parameters
    ----------
    *args
        Positional registration options, which are ignored.
    **kwargs
        Keyword registration options forwarded to ``overload``.

    Returns
    -------
    A decorator that returns its function unchanged.
    """

    def wrap[F: Callable[..., Any]](fn: F) -> F:
        """Register a no-op overload and return the function unchanged."""

        @overload(fn, jit_options=kwargs, strict=False)
        def ov_wrap(*args: Any, **kwargs: Any) -> F:
            """Return the original function as the overload implementation."""
            return fn

        return fn

    if kwargs:
        return wrap
    return wrap(*args)


def from_dtype(obj: Any) -> Any:
    """Return an object unchanged in place of Numba's ``from_dtype``.

    Parameters
    ----------
    obj
        Object to return.

    Returns
    -------
    The input object.
    """
    return obj


class SubscriptableType:
    """Provide a subscriptable stand-in for a Numba scalar type."""

    def __getitem__(self, item: Any) -> Self:
        """Ignore an array layout specification and return this instance.

        Parameters
        ----------
        item
            Layout specification, which is ignored.

        Returns
        -------
        This instance.
        """
        return self


int8 = SubscriptableType()
uint8 = SubscriptableType()
int16 = SubscriptableType()
uint16 = SubscriptableType()
int32 = SubscriptableType()
uint32 = SubscriptableType()
int64 = SubscriptableType()
uint64 = SubscriptableType()
float32 = SubscriptableType()
float64 = SubscriptableType()
boolean = SubscriptableType()
string = str

jit = jit_dummy
jitclass = jitclass_dummy
overload = overload_dummy
register_jitable = register_jitable_dummy
prange = range
