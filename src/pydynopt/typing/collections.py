"""
Custom collection and sequence types for typing and static analysis.

This work is licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar, overload

_T_co = TypeVar('_T_co', covariant=True)


class SequenceNotStr(Protocol[_T_co]):
    """
    A sequence protocol that excludes `str`.

    In Python, `str` is technically a subtype of `Sequence[str]`. This causes
    static type checkers (like Pyright or Mypy) to accept a single string when
    a sequence of strings is expected.

    This protocol excludes `str` by defining `__contains__` with a signature
    that accepts `object`. Standard sequence/collection types like `list` and
    `tuple` implement `__contains__` accepting any `object`, whereas `str`
    restricts its `__contains__` argument type strictly to `str`.

    As a result, type checkers will reject `str` as not satisfying this
    protocol, while continuing to accept lists, tuples, and other collections
    of `_T_co`.
    """

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> _T_co: ...

    @overload
    def __getitem__(self, idx: slice, /) -> Sequence[_T_co]: ...

    def __contains__(self, x: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
