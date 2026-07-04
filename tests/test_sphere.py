"""Test sphere lattice."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from fiblat import sphere_lattice
from fiblat._sphere_lattice import inv_int_sin_m, inv_int_sin_ms


def test_on_unit_sphere(benchmark: BenchmarkFixture) -> None:
    """Test points are on the unit sphere."""
    lattice = benchmark(sphere_lattice, 27, 1000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_on_unit_sphere_large(benchmark: BenchmarkFixture) -> None:
    """Test high dimensional points are on the unit sphere."""
    lattice = benchmark(sphere_lattice, 600, 3)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


@pytest.mark.long
def test_on_large_case(benchmark: BenchmarkFixture) -> None:
    """Test many high dimensional points on the unit sphere."""
    lattice = benchmark(sphere_lattice, 300, 30_000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_evenly_distributed() -> None:
    """Test that points are close to evenly distributed."""
    lattice = sphere_lattice(27, 100)
    dists = 1 - (lattice @ lattice.T) + 2 * np.eye(100)
    min_dists = dists.min(-1)
    errors = np.sum(min_dists < 0.3)  # noqa: PLR2004
    assert errors < 9  # noqa: PLR2004


def test_inv_int_sin_ms_duplicate_targets() -> None:
    """Test duplicate targets don't trigger unbounded recursion."""
    results = np.empty(2)
    inv_int_sin_ms(results, np.array([0.5, 0.5]), 2, 0.0, np.pi, 1e-10)
    expected = inv_int_sin_m(0.5, 2, 0.0, np.pi, 1e-10)
    assert np.allclose(results, expected)


def test_invalid_inputs() -> None:
    """Test sphere raises for invalid inputs."""
    with pytest.raises(ValueError):
        sphere_lattice(1, 2)
    with pytest.raises(ValueError):
        sphere_lattice(2, 0)
