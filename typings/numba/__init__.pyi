from collections.abc import Callable, Iterable
from typing import Protocol

class _Decorator(Protocol):
    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...

class Type: ...

class Argument(Type):
    def __call__(self, *_: Argument) -> Type: ...

class Scalar(Argument):
    def __getitem__(self, val: slice | tuple[slice, ...]) -> Argument: ...

float64: Scalar
int64: Scalar
uint64: Scalar
uint8: Scalar
void: Argument

def jit(
    sig: Type,
    /,
    *,
    cache: bool = ...,
    parallel: bool = ...,
    nogil: bool = ...,
    fastmath: bool = ...,
    error_model: str = ...,
) -> _Decorator: ...
def prange(start: int) -> Iterable[int]: ...
