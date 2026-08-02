from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

class BenchmarkFixture:
    def __call__(
        self, func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R: ...
