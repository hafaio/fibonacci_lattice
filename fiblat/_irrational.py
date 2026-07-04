import numba as nb
import numpy as np
from numpy.typing import NDArray

_SMALL_PRIMES = np.array([2, 3, 5, 7, 11], "i8")  # first five primes


@nb.jit(nb.int64[:](nb.int64), cache=True, nogil=True)
def n_primes(n: int) -> NDArray[np.int64]:  # pragma: nocover
    """Create an array of the first n primes."""
    if n <= 0:
        return np.empty(0, "i8")
    elif n <= _SMALL_PRIMES.size:
        return _SMALL_PRIMES[:n].copy()
    else:
        res = np.empty(n, "i8")
        res[0] = 2
        # Rosser's bound: p_n < n (ln n + ln ln n) for n >= 6
        max_num = int(n * (np.log(n) + np.log(np.log(n)))) + 1
        seive = np.ones((max_num - 1) // 2, "?")
        j = 1
        for i in range(3, max_num + 1, 2):
            if seive[(i - 3) // 2]:
                res[j] = i
                j += 1
                if j == n:
                    return res
                else:
                    seive[(i**2 - 3) // 2 :: i] = False
        raise ValueError("inaccurate bound")
