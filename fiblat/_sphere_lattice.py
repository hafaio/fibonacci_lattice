import math
from math import log, pi

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._cube_lattice import cube_lattice


@nb.jit(
    nb.float64(nb.float64, nb.int64),
    fastmath=True,
    error_model="numpy",
    cache=True,
    nogil=True,
)
def int_sin_m(x: float, m: int) -> float:  # pragma: no cover
    """Compute the integral of sin^m(t) dt from 0 to x recursively."""
    cosx = np.cos(x)
    sinx = np.sin(x)
    sinx2 = sinx**2
    start = m % 2
    res = x if start == 0 else 1 - cosx
    sinxp = sinx if start == 0 else sinx2
    for p in range(start + 1, m, 2):
        res = p / (p + 1) * res - cosx * sinxp / (p + 1)
        sinxp *= sinx2
    return res


@nb.jit(
    nb.float64(nb.float64, nb.int64, nb.float64, nb.float64, nb.float64),
    cache=True,
    nogil=True,
)
def inv_int_sin_m(
    target: float,
    m: int,
    lower: float,
    upper: float,
    atol: float,
) -> float:  # pragma: no cover
    """Invert mult * integral of sin(x) ** m.

    The inverse is accurate to an absolute tolerance of atol, and must be
    monotonically increasing over the interval lower to upper.
    """
    mid = (lower + upper) / 2
    approx = int_sin_m(mid, m)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = int_sin_m(mid, m)
    return mid


@nb.jit(
    nb.void(nb.float64[:], nb.float64[:], nb.int64, nb.float64, nb.float64, nb.float64),
    cache=True,
    nogil=True,
)
def inv_int_sin_ms(  # noqa: PLR0913
    results: NDArray[np.float64],
    targets: NDArray[np.float64],
    m: int,
    lower: float,
    upper: float,
    atol: float,
) -> None:  # pragma: no cover
    """Invert mult * integral of sin(x) ** m.

    The inverse is accurate to an absolute tolerance of atol, and must be
    monotonically increasing over the interval lower to upper.
    """
    if targets.size == 0:
        pass
    elif targets.size == 1:
        results[0] = inv_int_sin_m(targets[0], m, lower, upper, atol)
    else:
        mid = (lower + upper) / 2
        if mid <= lower or mid >= upper:
            results[:] = mid
        else:
            approx = int_sin_m(mid, m)
            ind = np.searchsorted(targets, approx)
            inv_int_sin_ms(results[:ind], targets[:ind], m, lower, mid, atol)
            inv_int_sin_ms(results[ind:], targets[ind:], m, mid, upper, atol)


_HLP = log(pi) / 2


@nb.jit(
    nb.float64[:, :](nb.int64, nb.int64),
    cache=True,
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def sphere_lattice(
    dim: int, num_points: int
) -> NDArray[np.float64]:  # pragma: no cover
    """Generate num_points points over the dim - 1 dimensional hypersphere.

    Generate a `num_points` length list of `dim`-dimensional tuples such the
    each element has an l2 norm of 1, and their nearest neighbor is roughly
    identical for each point.

    Parameters
    ----------
    dim : the dimension of points to sample, i.e. the length of tuples in the
        returned list
    num_points : the number of points to generate
    """
    if dim < 2:  # noqa: PLR2004
        raise ValueError(f"dimension must be greater than one: {dim}")
    elif num_points < 1:
        raise ValueError(f"must request at least one point: {num_points}")
    cube = cube_lattice(dim - 1, num_points)

    output = np.ones((num_points, dim), "f8")
    output[:, 0] *= np.sin(2 * np.pi * cube[:, 0])
    output[:, 1] *= np.cos(2 * np.pi * cube[:, 0])
    pdegs = np.empty(num_points)

    for d, targets in enumerate(cube.T[1:], 2):
        mult = np.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2) - _HLP)
        order = targets.argsort()
        inv_int_sin_ms(pdegs, targets[order] / mult, d - 1, 0, np.pi, 1e-10)
        output[order, :d] *= np.sin(pdegs)[:, None]
        output[order, d] *= np.cos(pdegs)
    return output
