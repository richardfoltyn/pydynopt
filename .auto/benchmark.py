"""Benchmark Numba-compiled one- and two-dimensional interpolation.

The workloads call pydynopt's public interpolation API from Numba-compiled loops.
Two-dimensional evaluation follows the scalar, repeated-field access pattern used
by economic-model continuation-value kernels.

Author: Richard Foltyn
"""

from collections.abc import Callable
from dataclasses import dataclass
import gc
import math
import os
from statistics import fmean, median, pstdev
from time import perf_counter_ns

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

EXPECTED_AFFINITY = frozenset(range(8))
N_QUERY = 2048
N_FIELDS = 4
SAMPLES = 7
TARGET_SAMPLE_NS = 40_000_000


@dataclass(frozen=True)
class Case:
    """Describe one benchmark case."""

    name: str
    func: Callable[..., float]
    args: tuple[object, ...]
    groups: tuple[str, ...]


@njit(nogil=True)
def _bench_1d_locate(x: np.ndarray, xp: np.ndarray, rounds: int) -> float:
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
def _bench_1d_pipeline(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
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
def _bench_1d_interp(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    hints: np.ndarray,
    rounds: int,
) -> float:
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        for k in range(x.size):
            i = (k + r) & mask
            acc += interp1d(x[i], xp, fp, hints[i])
    return acc


@njit(nogil=True)
def _bench_1d_array(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    acc = 0.0
    mask = x.size - 1
    for r in range(rounds):
        result = interp1d(x, xp, fp, 0, True, np.nan, np.nan, out)
        acc += result[r & mask]
    return acc


@njit(nogil=True)
def _bench_2d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    rounds: int,
) -> float:
    acc = 0.0
    mask = index.shape[0] - 1
    for r in range(rounds):
        for k in range(index.shape[0]):
            i = (k + r) & mask
            for j in range(fp.shape[0]):
                acc += float(interp2d_eval(index[i], weight[i], fp[j]))
    return acc


@njit(nogil=True)
def _bench_2d_locate(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    rounds: int,
) -> float:
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
def _bench_2d_interp(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    hints: np.ndarray,
    rounds: int,
) -> float:
    acc = 0.0
    mask = x0.size - 1
    for r in range(rounds):
        for k in range(x0.size):
            i = (k + r) & mask
            acc += interp2d(x0[i], x1[i], xp0, xp1, fp, hints[i])
    return acc


@njit(nogil=True)
def _bench_2d_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    hint: np.ndarray,
    out: np.ndarray,
    rounds: int,
) -> float:
    acc = 0.0
    mask = x0.size - 1
    for r in range(rounds):
        result = interp2d(x0, x1, xp0, xp1, fp, hint, True, out)
        acc += result[r & mask]
    return acc


def _geomean(values: list[float]) -> float:
    """Return the geometric mean of positive values."""
    return math.exp(fmean(math.log(value) for value in values))


def _triangle_indices(count: int, intervals: int, stride: int = 1) -> np.ndarray:
    """Create a cyclic sequence moving only between neighboring intervals."""
    period = 2 * intervals - 2
    phase = (np.arange(count, dtype=np.int64) // stride) % period
    index = np.where(phase < intervals, phase, period - phase)
    return np.asarray(index, dtype=np.int64)


def _queries(xp: np.ndarray, index: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Create interior query points from interval indices and upper weights."""
    dx = xp[index + 1] - xp[index]
    return np.asarray(xp[index] + fraction * dx, dtype=np.float64)


def _previous_indices(index: np.ndarray) -> np.ndarray:
    """Return the preceding interval as the search hint for each query."""
    hints = np.empty_like(index)
    hints[0] = 0
    hints[1:] = index[:-1]
    return hints


def _make_cases() -> list[Case]:
    """Create deterministic core workloads."""
    rng = np.random.default_rng(917_341)

    u1 = np.linspace(0.0, 1.0, 129)
    xp1 = -2.0 + 12.0 * u1**1.65
    index1_local = _triangle_indices(N_QUERY, xp1.size - 1)
    index1_random = rng.integers(0, xp1.size - 1, size=N_QUERY, dtype=np.int64)
    fraction1 = 0.1 + 0.8 * rng.random(N_QUERY)
    x1_local = _queries(xp1, index1_local, fraction1)
    x1_random = _queries(xp1, index1_random, fraction1)
    hint1_local = _previous_indices(index1_local)
    hint1_random = _previous_indices(index1_random)

    fp1 = np.empty((N_FIELDS, xp1.size), dtype=np.float64)
    for j in range(N_FIELDS):
        fp1[j] = np.sin((j + 1.0) * xp1 * 0.17) + (j + 0.5) * xp1 * 0.03

    u0 = np.linspace(0.0, 1.0, 17)
    u2 = np.linspace(0.0, 1.0, 129)
    xp0 = -0.75 + 2.5 * u0**1.3
    xp2 = -2.0 + 12.0 * u2**1.65
    index0_local = _triangle_indices(N_QUERY, xp0.size - 1, stride=16)
    index2_local = _triangle_indices(N_QUERY, xp2.size - 1)
    index0_random = rng.integers(0, xp0.size - 1, size=N_QUERY, dtype=np.int64)
    index2_random = rng.integers(0, xp2.size - 1, size=N_QUERY, dtype=np.int64)
    fraction0 = 0.1 + 0.8 * rng.random(N_QUERY)
    fraction2 = 0.1 + 0.8 * rng.random(N_QUERY)
    x0_local = _queries(xp0, index0_local, fraction0)
    x2_local = _queries(xp2, index2_local, fraction2)
    x0_random = _queries(xp0, index0_random, fraction0)
    x2_random = _queries(xp2, index2_random, fraction2)

    index2d = np.column_stack((index0_local, index2_local))
    weight2d = np.column_stack((1.0 - fraction0, 1.0 - fraction2))
    hint2d_local = np.column_stack(
        (_previous_indices(index0_local), _previous_indices(index2_local))
    )
    hint2d_random = np.column_stack(
        (_previous_indices(index0_random), _previous_indices(index2_random))
    )

    z0, z1 = np.meshgrid(xp0, xp2, indexing='ij')
    fp2 = np.empty((N_FIELDS, xp0.size, xp2.size), dtype=np.float64)
    for j in range(N_FIELDS):
        fp2[j] = (
            np.sin((j + 1.0) * z0 * 0.31)
            + np.cos((j + 0.5) * z1 * 0.13)
            + 0.02 * z0 * z1
        )

    fp2_base = np.empty(
        (N_FIELDS, xp0.size, 2, xp2.size),
        dtype=np.float64,
    )
    fp2_strided = fp2_base[:, :, 0, :]
    fp2_strided[...] = fp2

    groups_1d_scalar = ('1d', 'scalar')
    groups_2d_scalar = ('2d', 'scalar')
    return [
        Case(
            'locate1d_local',
            _bench_1d_locate,
            (x1_local, xp1),
            groups_1d_scalar,
        ),
        Case(
            'locate1d_random',
            _bench_1d_locate,
            (x1_random, xp1),
            groups_1d_scalar,
        ),
        Case(
            'pipeline1d_4field',
            _bench_1d_pipeline,
            (x1_local, xp1, fp1),
            groups_1d_scalar,
        ),
        Case(
            'interp1d_local',
            _bench_1d_interp,
            (x1_local, xp1, fp1[0], hint1_local),
            groups_1d_scalar,
        ),
        Case(
            'interp1d_random',
            _bench_1d_interp,
            (x1_random, xp1, fp1[0], hint1_random),
            groups_1d_scalar,
        ),
        Case(
            'array1d_local',
            _bench_1d_array,
            (x1_local, xp1, fp1[0], np.empty_like(x1_local)),
            ('1d', 'array'),
        ),
        Case(
            'eval2d_4field_c',
            _bench_2d_eval,
            (index2d, weight2d, fp2),
            groups_2d_scalar,
        ),
        Case(
            'eval2d_4field_strided',
            _bench_2d_eval,
            (index2d, weight2d, fp2_strided),
            groups_2d_scalar,
        ),
        Case(
            'locate2d_local',
            _bench_2d_locate,
            (x0_local, x2_local, xp0, xp2),
            groups_2d_scalar,
        ),
        Case(
            'interp2d_local',
            _bench_2d_interp,
            (x0_local, x2_local, xp0, xp2, fp2[0], hint2d_local),
            groups_2d_scalar,
        ),
        Case(
            'interp2d_random',
            _bench_2d_interp,
            (x0_random, x2_random, xp0, xp2, fp2[0], hint2d_random),
            groups_2d_scalar,
        ),
        Case(
            'array2d_local',
            _bench_2d_array,
            (
                x0_local,
                x2_local,
                xp0,
                xp2,
                fp2[0],
                np.zeros(2, dtype=np.int64),
                np.empty_like(x0_local),
            ),
            ('2d', 'array'),
        ),
    ]


def _time_case(case: Case) -> tuple[float, float]:
    """Return median nanoseconds per query and sample CV percentage."""
    result = case.func(*case.args, 1)
    if not math.isfinite(result):
        msg = f'{case.name} produced a non-finite warm-up checksum'
        raise RuntimeError(msg)

    probe_rounds = 1
    start = perf_counter_ns()
    case.func(*case.args, probe_rounds)
    elapsed = perf_counter_ns() - start
    if elapsed < 2_000_000:
        probe_rounds = min(4096, max(1, math.ceil(2_000_000 / max(elapsed, 1))))
        start = perf_counter_ns()
        case.func(*case.args, probe_rounds)
        elapsed = perf_counter_ns() - start

    rounds = min(
        100_000,
        max(1, math.ceil(TARGET_SAMPLE_NS * probe_rounds / max(elapsed, 1))),
    )
    case.func(*case.args, max(1, rounds // 4))

    samples: list[float] = []
    for _ in range(SAMPLES):
        start = perf_counter_ns()
        result = case.func(*case.args, rounds)
        duration = perf_counter_ns() - start
        if not math.isfinite(result):
            msg = f'{case.name} produced a non-finite timed checksum'
            raise RuntimeError(msg)
        samples.append(duration / (rounds * N_QUERY))

    value = median(samples)
    cv_pct = 100.0 * pstdev(samples) / fmean(samples)
    return value, cv_pct


def main() -> None:
    """Run all benchmark cases and emit structured metrics."""
    affinity = frozenset(os.sched_getaffinity(0))
    if affinity != EXPECTED_AFFINITY:
        msg = f'benchmark affinity must be 0-7, got {sorted(affinity)}'
        raise RuntimeError(msg)

    gc.disable()
    cases = _make_cases()
    timings: dict[str, float] = {}
    cvs: dict[str, float] = {}
    groups: dict[str, list[float]] = {'1d': [], '2d': [], 'scalar': [], 'array': []}

    for case in cases:
        value, cv_pct = _time_case(case)
        timings[case.name] = value
        cvs[case.name] = cv_pct
        for group in case.groups:
            groups[group].append(value)

    case_summary = ' '.join(f'{name}={value:.3f}' for name, value in timings.items())
    print(f'AFFINITY 0-7; CASE_NS {case_summary}')
    print(f'METRIC geomean_ns={_geomean(list(timings.values())):.6f}')
    print(f'METRIC one_d_ns={_geomean(groups["1d"]):.6f}')
    print(f'METRIC two_d_ns={_geomean(groups["2d"]):.6f}')
    print(f'METRIC scalar_ns={_geomean(groups["scalar"]):.6f}')
    print(f'METRIC array_ns={_geomean(groups["array"]):.6f}')
    print(f'METRIC locate1d_local_ns={timings["locate1d_local"]:.6f}')
    eval2d = _geomean([timings['eval2d_4field_c'], timings['eval2d_4field_strided']])
    print(f'METRIC eval2d_ebl_ns={eval2d:.6f}')
    print(f'METRIC max_cv_pct={max(cvs.values()):.6f}')


if __name__ == '__main__':
    main()
