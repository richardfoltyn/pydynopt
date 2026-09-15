"""Timed interpolation workload kernels.

Author: Richard Foltyn
"""

from numba import njit
import numpy as np

from pydynopt.interpolate import (
    interp1d,
    interp1d_eval,
    interp1d_locate,
    interp2d,
    interp2d_eval,
    interp2d_locate,
)


@njit(nogil=True)
def bench_1d_locate(x: np.ndarray, xp: np.ndarray, rounds: int) -> float:
    """Benchmark stateful scalar 1D location."""
    acc = 0.0
    mask = x.size - 1
    index = 0
    for r in range(rounds):
        for k in range(x.size):
            i = (k + r) & mask
            index, weight = interp1d_locate(x[i], xp, index)
            acc += weight + index * 1.0e-4

    return acc


@njit(nogil=True)
def bench_1d_pipeline(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark one location reused across several 1D fields."""
    acc = 0.0
    mask = x.size - 1
    index = 0
    for r in range(rounds):
        for k in range(x.size):
            i = (k + r) & mask
            index, weight = interp1d_locate(x[i], xp, index)
            for j in range(fp.shape[0]):
                acc += interp1d_eval(index, weight, fp[j])

    return acc


@njit(nogil=True)
def bench_1d_interp(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    hints: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark fused scalar 1D interpolation with supplied hints."""
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        for k in range(x.size):
            i = (k + r) & mask
            acc += interp1d(x[i], xp, fp, hints[i])

    return acc


@njit(nogil=True)
def bench_1d_array_reuse(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark 1D array interpolation with a reusable output buffer."""
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        result = interp1d(x, xp, fp, 0, True, np.nan, np.nan, out)
        acc += result[r & mask]

    return acc


@njit(nogil=True)
def bench_1d_array_allocate(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark 1D array interpolation with output allocation."""
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        result = interp1d(x, xp, fp)
        acc += result[r & mask]

    return acc


@njit(nogil=True)
def bench_2d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark repeated 2D field evaluation with array coordinates."""
    acc = 0.0
    mask = index.shape[0] - 1
    for r in range(rounds):
        for k in range(index.shape[0]):
            i = (k + r) & mask
            for j in range(fp.shape[0]):
                acc += float(interp2d_eval(index[i], weight[i], fp[j]))

    return acc


@njit(nogil=True)
def bench_2d_eval_tuple(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark repeated 2D field evaluation with scalar tuples."""
    acc = 0.0
    mask = index.shape[0] - 1
    for r in range(rounds):
        for k in range(index.shape[0]):
            i = (k + r) & mask
            idx = (index[i, 0], index[i, 1])
            wgt = (weight[i, 0], weight[i, 1])
            for j in range(fp.shape[0]):
                acc += float(interp2d_eval(idx, wgt, fp[j]))

    return acc


@njit(nogil=True)
def bench_2d_locate(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark buffered stateful scalar 2D location."""
    acc = 0.0
    mask = x0.size - 1
    index = np.zeros(2, dtype=np.int64)
    weight = np.empty(2, dtype=np.float64)
    for r in range(rounds):
        for k in range(x0.size):
            i = (k + r) & mask
            interp2d_locate(
                x0[i],
                x1[i],
                xp0,
                xp1,
                index,
                index,
                weight,
            )
            acc += weight[0] + weight[1] + (index[0] + index[1]) * 1.0e-4

    return acc


@njit(nogil=True)
def bench_2d_interp(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    hints: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark fused scalar 2D interpolation with supplied hints."""
    acc = 0.0
    mask = x0.size - 1
    for r in range(rounds):
        for k in range(x0.size):
            i = (k + r) & mask
            acc += interp2d(x0[i], x1[i], xp0, xp1, fp, hints[i])

    return acc


@njit(nogil=True)
def bench_2d_array_reuse(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    hint: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark 2D array interpolation with a reusable output buffer."""
    acc = 0.0
    mask = x0.size - 1
    for r in range(rounds):
        result = interp2d(x0, x1, xp0, xp1, fp, hint, True, out)
        acc += result[r & mask]

    return acc


@njit(nogil=True)
def bench_2d_array_allocate(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark 2D array interpolation with output allocation."""
    acc = 0.0
    mask = x0.size - 1
    for r in range(rounds):
        result = interp2d(x0, x1, xp0, xp1, fp)
        acc += result[r & mask]

    return acc


def bench_python_1d_array(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark the direct Python/NumPy 1D array API."""
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        result = interp1d(x, xp, fp, out=out)
        acc += float(result.flat[r & mask])

    return acc


def bench_python_2d_broadcast(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    """Benchmark the direct Python/NumPy broadcasting 2D API."""
    acc = 0.0
    mask = out.size - 1
    for r in range(rounds):
        result = interp2d(x0, x1, xp0, xp1, fp, out=out)
        acc += float(result.flat[r & mask])

    return acc
