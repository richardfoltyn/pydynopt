"""Construct deterministic interpolation benchmark cases.

Author: Richard Foltyn
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from benchmarks._runner import BenchmarkCase
from benchmarks.interpolate import kernels

N_FIELDS = 4
N_QUERY = 2048
RANDOM_SEED = 917_341


@dataclass(frozen=True)
class OneDimensionalData:
    """Inputs shared by the one-dimensional workloads."""

    xp: np.ndarray
    fp: np.ndarray
    same: np.ndarray
    local: np.ndarray
    random: np.ndarray
    extrapolated: np.ndarray
    local_hints: np.ndarray
    random_hints: np.ndarray


@dataclass(frozen=True)
class TwoDimensionalData:
    """Inputs shared by the two-dimensional workloads."""

    xp0: np.ndarray
    xp1: np.ndarray
    fp: np.ndarray
    fp_strided: np.ndarray
    x0_local: np.ndarray
    x1_local: np.ndarray
    x0_random: np.ndarray
    x1_random: np.ndarray
    x0_extrapolated: np.ndarray
    x1_extrapolated: np.ndarray
    index: np.ndarray
    weight: np.ndarray
    local_hints: np.ndarray
    random_hints: np.ndarray
    x0_broadcast: np.ndarray
    x1_broadcast: np.ndarray


@dataclass(frozen=True)
class ThreeDimensionalData:
    """Store the frozen inputs shared by all 3D workloads.

    The data include local and random point sequences, prelocated coordinates,
    C-contiguous and arbitrary-strided fields, extrapolated samples, and broadcast
    views. Sharing one object keeps numerical inputs identical across paired cases.
    """

    xp0: np.ndarray
    xp1: np.ndarray
    xp2: np.ndarray
    fp: np.ndarray
    fp_strided: np.ndarray
    x0_local: np.ndarray
    x1_local: np.ndarray
    x2_local: np.ndarray
    x0_random: np.ndarray
    x1_random: np.ndarray
    x2_random: np.ndarray
    x0_extrapolated: np.ndarray
    x1_extrapolated: np.ndarray
    x2_extrapolated: np.ndarray
    index: np.ndarray
    weight: np.ndarray
    local_hints: np.ndarray
    random_hints: np.ndarray
    x0_broadcast: np.ndarray
    x1_broadcast: np.ndarray
    x2_broadcast: np.ndarray


def _triangle_indices(count: int, intervals: int, stride: int = 1) -> np.ndarray:
    """Create a cyclic sequence moving only between neighboring intervals."""
    period = 2 * intervals - 2
    phase = (np.arange(count, dtype=np.int64) // stride) % period
    index = np.where(phase < intervals, phase, period - phase)
    result = np.asarray(index, dtype=np.int64)

    return result


def _queries(
    xp: np.ndarray,
    index: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Create interior queries from interval indices and upper weights."""
    dx = xp[index + 1] - xp[index]
    result = xp[index] + fraction * dx

    return result


def _previous_indices(index: np.ndarray) -> np.ndarray:
    """Return the preceding interval as the search hint for each query."""
    hints = np.empty_like(index)
    hints[0] = 0
    hints[1:] = index[:-1]

    return hints


def _make_1d_data(rng: np.random.Generator) -> OneDimensionalData:
    """Create representative one-dimensional interpolation inputs."""
    u = np.linspace(0.0, 1.0, 129)
    xp = -2.0 + 12.0 * u**1.65
    index_local = _triangle_indices(N_QUERY, xp.size - 1)
    index_random = rng.integers(0, xp.size - 1, size=N_QUERY, dtype=np.int64)
    fraction = 0.1 + 0.8 * rng.random(N_QUERY)
    local = _queries(xp, index_local, fraction)
    random = _queries(xp, index_random, fraction)
    same = _queries(
        xp,
        np.full(N_QUERY, xp.size // 3, dtype=np.int64),
        fraction,
    )
    extrapolated = local.copy()
    extrapolated[::16] = xp[0] - 0.25
    extrapolated[8::16] = xp[-1] + 0.25

    fp = np.empty((N_FIELDS, xp.size), dtype=np.float64)
    for j in range(N_FIELDS):
        fp[j] = np.sin((j + 1.0) * xp * 0.17) + (j + 0.5) * xp * 0.03

    data = OneDimensionalData(
        xp=xp,
        fp=fp,
        same=same,
        local=local,
        random=random,
        extrapolated=extrapolated,
        local_hints=_previous_indices(index_local),
        random_hints=_previous_indices(index_random),
    )

    return data


def _make_2d_data(rng: np.random.Generator) -> TwoDimensionalData:
    """Create representative two-dimensional interpolation inputs."""
    u0 = np.linspace(0.0, 1.0, 17)
    u1 = np.linspace(0.0, 1.0, 129)
    xp0 = -0.75 + 2.5 * u0**1.3
    xp1 = -2.0 + 12.0 * u1**1.65
    index0_local = _triangle_indices(N_QUERY, xp0.size - 1, stride=16)
    index1_local = _triangle_indices(N_QUERY, xp1.size - 1)
    index0_random = rng.integers(0, xp0.size - 1, size=N_QUERY, dtype=np.int64)
    index1_random = rng.integers(0, xp1.size - 1, size=N_QUERY, dtype=np.int64)
    fraction0 = 0.1 + 0.8 * rng.random(N_QUERY)
    fraction1 = 0.1 + 0.8 * rng.random(N_QUERY)
    x0_local = _queries(xp0, index0_local, fraction0)
    x1_local = _queries(xp1, index1_local, fraction1)
    x0_random = _queries(xp0, index0_random, fraction0)
    x1_random = _queries(xp1, index1_random, fraction1)

    index = np.column_stack((index0_local, index1_local))
    weight = np.column_stack((1.0 - fraction0, 1.0 - fraction1))
    local_hints = np.column_stack(
        (_previous_indices(index0_local), _previous_indices(index1_local))
    )
    random_hints = np.column_stack(
        (_previous_indices(index0_random), _previous_indices(index1_random))
    )

    fp = np.empty((N_FIELDS, xp0.size, xp1.size), dtype=np.float64)
    for j in range(N_FIELDS):
        fp[j] = (
            np.sin((j + 1.0) * xp0[:, None] * 0.31)
            + np.cos((j + 1.5) * xp1[None, :] * 0.13)
            + (j + 0.25) * xp0[:, None] * xp1[None, :] * 0.01
        )

    backing = np.empty((N_FIELDS, xp0.size, 2 * xp1.size), dtype=np.float64)
    backing[:, :, ::2] = fp
    fp_strided = backing[:, :, ::2]

    x0_extrapolated = x0_local.copy()
    x1_extrapolated = x1_local.copy()
    x0_extrapolated[::16] = xp0[0] - 0.25
    x0_extrapolated[8::16] = xp0[-1] + 0.25
    x1_extrapolated[4::16] = xp1[0] - 0.25
    x1_extrapolated[12::16] = xp1[-1] + 0.25

    data = TwoDimensionalData(
        xp0=xp0,
        xp1=xp1,
        fp=fp,
        fp_strided=fp_strided,
        x0_local=x0_local,
        x1_local=x1_local,
        x0_random=x0_random,
        x1_random=x1_random,
        x0_extrapolated=x0_extrapolated,
        x1_extrapolated=x1_extrapolated,
        index=index,
        weight=weight,
        local_hints=local_hints,
        random_hints=random_hints,
        x0_broadcast=x0_local[:64, None],
        x1_broadcast=x1_local[:32][None, :],
    )

    return data


def _make_3d_data(rng: np.random.Generator) -> ThreeDimensionalData:
    """Create deterministic inputs for the balanced 3D benchmark group.

    Nonuniform grids have different axis lengths so row-major addressing costs are
    realistic. Query sets cover locally moving and random intervals, while field
    storage covers C-contiguous and genuinely arbitrary-strided layouts.
    """
    # Keep axis 2 largest and fastest-moving, matching row-major model arrays and
    # making the local sequence exercise a distinct movement rate on every axis.
    u0 = np.linspace(0.0, 1.0, 9)
    u1 = np.linspace(0.0, 1.0, 17)
    u2 = np.linspace(0.0, 1.0, 65)
    xp0 = -0.5 + 1.5 * u0**1.2
    xp1 = -0.75 + 2.5 * u1**1.3
    xp2 = -2.0 + 12.0 * u2**1.65
    # Slow axes retain an interval across many samples; axis 2 moves every sample.
    # Random sequences provide the counterweight to these stateful local searches.
    index0_local = _triangle_indices(N_QUERY, xp0.size - 1, stride=256)
    index1_local = _triangle_indices(N_QUERY, xp1.size - 1, stride=16)
    index2_local = _triangle_indices(N_QUERY, xp2.size - 1)
    index0_random = rng.integers(0, xp0.size - 1, size=N_QUERY, dtype=np.int64)
    index1_random = rng.integers(0, xp1.size - 1, size=N_QUERY, dtype=np.int64)
    index2_random = rng.integers(0, xp2.size - 1, size=N_QUERY, dtype=np.int64)
    fraction0 = 0.1 + 0.8 * rng.random(N_QUERY)
    fraction1 = 0.1 + 0.8 * rng.random(N_QUERY)
    fraction2 = 0.1 + 0.8 * rng.random(N_QUERY)
    x0_local = _queries(xp0, index0_local, fraction0)
    x1_local = _queries(xp1, index1_local, fraction1)
    x2_local = _queries(xp2, index2_local, fraction2)
    x0_random = _queries(xp0, index0_random, fraction0)
    x1_random = _queries(xp1, index1_random, fraction1)
    x2_random = _queries(xp2, index2_random, fraction2)

    index = np.column_stack((index0_local, index1_local, index2_local))
    weight = np.column_stack((1.0 - fraction0, 1.0 - fraction1, 1.0 - fraction2))
    local_hints = np.column_stack(
        (
            _previous_indices(index0_local),
            _previous_indices(index1_local),
            _previous_indices(index2_local),
        )
    )
    random_hints = np.column_stack(
        (
            _previous_indices(index0_random),
            _previous_indices(index1_random),
            _previous_indices(index2_random),
        )
    )

    # Use smooth, nonseparable fields so all eight corners affect the checksum and
    # repeated-field evaluation cannot collapse to one common affine expression.
    fp = np.empty((N_FIELDS, xp0.size, xp1.size, xp2.size), dtype=np.float64)
    for j in range(N_FIELDS):
        fp[j] = (
            np.sin((j + 1.0) * xp0[:, None, None] * 0.31)
            + np.cos((j + 1.5) * xp1[None, :, None] * 0.21)
            + np.sin((j + 0.5) * xp2[None, None, :] * 0.13)
            + (j + 0.25)
            * xp0[:, None, None]
            * xp1[None, :, None]
            * xp2[None, None, :]
            * 0.01
        )

    # Slice a doubled final axis to retain non-unit strides without changing values.
    # This creates Numba A-layout inputs for the generic point evaluator.
    backing = np.empty((N_FIELDS, xp0.size, xp1.size, 2 * xp2.size), dtype=np.float64)
    backing[:, :, :, ::2] = fp
    fp_strided = backing[:, :, :, ::2]

    # Stagger exterior coordinates across axes and sides so extrapolation cases do
    # not benchmark one repeatedly predictable boundary branch.
    x0_extrapolated = x0_local.copy()
    x1_extrapolated = x1_local.copy()
    x2_extrapolated = x2_local.copy()
    x0_extrapolated[::32] = xp0[0] - 0.25
    x0_extrapolated[16::32] = xp0[-1] + 0.25
    x1_extrapolated[8::32] = xp1[0] - 0.25
    x1_extrapolated[24::32] = xp1[-1] + 0.25
    x2_extrapolated[4::16] = xp2[0] - 0.25
    x2_extrapolated[12::16] = xp2[-1] + 0.25

    data = ThreeDimensionalData(
        xp0=xp0,
        xp1=xp1,
        xp2=xp2,
        fp=fp,
        fp_strided=fp_strided,
        x0_local=x0_local,
        x1_local=x1_local,
        x2_local=x2_local,
        x0_random=x0_random,
        x1_random=x1_random,
        x2_random=x2_random,
        x0_extrapolated=x0_extrapolated,
        x1_extrapolated=x1_extrapolated,
        x2_extrapolated=x2_extrapolated,
        index=index,
        weight=weight,
        local_hints=local_hints,
        random_hints=random_hints,
        # These views broadcast to 8 * 16 * 16 == N_QUERY output elements.
        x0_broadcast=x0_local[:8, None, None],
        x1_broadcast=x1_local[:16][None, :, None],
        x2_broadcast=x2_local[:16][None, None, :],
    )

    return data


def _case(
    name: str,
    func: Callable[..., float],
    args: tuple[object, ...],
    groups: tuple[str, ...],
) -> BenchmarkCase:
    """Create one interpolation benchmark case."""
    case = BenchmarkCase(
        name=name,
        func=func,
        args=args,
        operations_per_round=N_QUERY,
        groups=groups,
    )

    return case


def make_cases() -> list[BenchmarkCase]:
    """Create the frozen interpolation benchmark suite."""
    rng = np.random.default_rng(RANDOM_SEED)
    one = _make_1d_data(rng)
    two = _make_2d_data(rng)
    three = _make_3d_data(rng)

    cases = [
        _case(
            'locate1d_same',
            kernels.bench_1d_locate,
            (one.same, one.xp),
            ('numba', 'numba.search', '1d'),
        ),
        _case(
            'locate1d_local',
            kernels.bench_1d_locate,
            (one.local, one.xp),
            ('core', 'numba', 'numba.search', '1d'),
        ),
        _case(
            'locate1d_random',
            kernels.bench_1d_locate,
            (one.random, one.xp),
            ('core', 'numba', 'numba.search', '1d'),
        ),
        _case(
            'pipeline1d_4field',
            kernels.bench_1d_pipeline,
            (one.local, one.xp, one.fp),
            ('core', 'numba', 'numba.scalar', 'ebl', '1d'),
        ),
        _case(
            'interp1d_local',
            kernels.bench_1d_interp,
            (one.local, one.xp, one.fp[0], one.local_hints),
            ('core', 'numba', 'numba.scalar', '1d'),
        ),
        _case(
            'interp1d_random',
            kernels.bench_1d_interp,
            (one.random, one.xp, one.fp[0], one.random_hints),
            ('core', 'numba', 'numba.scalar', '1d'),
        ),
        _case(
            'array1d_local',
            kernels.bench_1d_array_reuse,
            (one.local, one.xp, one.fp[0], np.empty_like(one.local)),
            ('core', 'numba', 'numba.array', '1d'),
        ),
        _case(
            'array1d_random',
            kernels.bench_1d_array_reuse,
            (one.random, one.xp, one.fp[0], np.empty_like(one.random)),
            ('numba', 'numba.array', '1d'),
        ),
        _case(
            'array1d_extrapolate',
            kernels.bench_1d_array_reuse,
            (
                one.extrapolated,
                one.xp,
                one.fp[0],
                np.empty_like(one.extrapolated),
            ),
            ('numba', 'numba.array', '1d'),
        ),
        _case(
            'array1d_allocate',
            kernels.bench_1d_array_allocate,
            (one.local, one.xp, one.fp[0]),
            ('numba', 'numba.array', '1d'),
        ),
        _case(
            'eval2d_4field_c',
            kernels.bench_2d_eval,
            (two.index, two.weight, two.fp),
            ('core', 'numba', 'numba.scalar', 'ebl', '2d'),
        ),
        _case(
            'eval2d_4field_strided',
            kernels.bench_2d_eval,
            (two.index, two.weight, two.fp_strided),
            ('core', 'numba', 'numba.scalar', 'ebl', '2d'),
        ),
        _case(
            'eval2d_4field_tuple_c',
            kernels.bench_2d_eval_tuple,
            (two.index, two.weight, two.fp),
            ('numba', 'numba.scalar', 'ebl', '2d'),
        ),
        _case(
            'locate2d_local',
            kernels.bench_2d_locate,
            (two.x0_local, two.x1_local, two.xp0, two.xp1),
            ('core', 'numba', 'numba.search', '2d'),
        ),
        _case(
            'locate2d_random',
            kernels.bench_2d_locate,
            (two.x0_random, two.x1_random, two.xp0, two.xp1),
            ('numba', 'numba.search', '2d'),
        ),
        _case(
            'interp2d_local',
            kernels.bench_2d_interp,
            (
                two.x0_local,
                two.x1_local,
                two.xp0,
                two.xp1,
                two.fp[0],
                two.local_hints,
            ),
            ('core', 'numba', 'numba.scalar', '2d'),
        ),
        _case(
            'interp2d_random',
            kernels.bench_2d_interp,
            (
                two.x0_random,
                two.x1_random,
                two.xp0,
                two.xp1,
                two.fp[0],
                two.random_hints,
            ),
            ('core', 'numba', 'numba.scalar', '2d'),
        ),
        _case(
            'interp2d_local_strided',
            kernels.bench_2d_interp,
            (
                two.x0_local,
                two.x1_local,
                two.xp0,
                two.xp1,
                two.fp_strided[0],
                two.local_hints,
            ),
            ('numba', 'numba.scalar', '2d'),
        ),
        _case(
            'array2d_local',
            kernels.bench_2d_array_reuse,
            (
                two.x0_local,
                two.x1_local,
                two.xp0,
                two.xp1,
                two.fp[0],
                np.zeros(2, dtype=np.int64),
                np.empty_like(two.x0_local),
            ),
            ('core', 'numba', 'numba.array', '2d'),
        ),
        _case(
            'array2d_random',
            kernels.bench_2d_array_reuse,
            (
                two.x0_random,
                two.x1_random,
                two.xp0,
                two.xp1,
                two.fp[0],
                np.zeros(2, dtype=np.int64),
                np.empty_like(two.x0_random),
            ),
            ('numba', 'numba.array', '2d'),
        ),
        _case(
            'array2d_extrapolate',
            kernels.bench_2d_array_reuse,
            (
                two.x0_extrapolated,
                two.x1_extrapolated,
                two.xp0,
                two.xp1,
                two.fp[0],
                np.zeros(2, dtype=np.int64),
                np.empty_like(two.x0_extrapolated),
            ),
            ('numba', 'numba.array', '2d'),
        ),
        _case(
            'array2d_allocate',
            kernels.bench_2d_array_allocate,
            (two.x0_local, two.x1_local, two.xp0, two.xp1, two.fp[0]),
            ('numba', 'numba.array', '2d'),
        ),
        _case(
            'eval3d_4field_c',
            kernels.bench_3d_eval,
            (three.index, three.weight, three.fp),
            ('core', 'numba', 'numba.scalar', 'ebl', '3d'),
        ),
        _case(
            'eval3d_4field_strided',
            kernels.bench_3d_eval,
            (three.index, three.weight, three.fp_strided),
            ('core', 'numba', 'numba.scalar', 'ebl', '3d'),
        ),
        _case(
            'eval3d_4field_tuple_c',
            kernels.bench_3d_eval_tuple,
            (three.index, three.weight, three.fp),
            ('numba', 'numba.scalar', 'ebl', '3d'),
        ),
        _case(
            'locate3d_local',
            kernels.bench_3d_locate,
            (
                three.x0_local,
                three.x1_local,
                three.x2_local,
                three.xp0,
                three.xp1,
                three.xp2,
            ),
            ('core', 'numba', 'numba.search', '3d'),
        ),
        _case(
            'locate3d_random',
            kernels.bench_3d_locate,
            (
                three.x0_random,
                three.x1_random,
                three.x2_random,
                three.xp0,
                three.xp1,
                three.xp2,
            ),
            ('numba', 'numba.search', '3d'),
        ),
        _case(
            'interp3d_local',
            kernels.bench_3d_interp,
            (
                three.x0_local,
                three.x1_local,
                three.x2_local,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                three.local_hints,
            ),
            ('core', 'numba', 'numba.scalar', '3d'),
        ),
        _case(
            'interp3d_random',
            kernels.bench_3d_interp,
            (
                three.x0_random,
                three.x1_random,
                three.x2_random,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                three.random_hints,
            ),
            ('core', 'numba', 'numba.scalar', '3d'),
        ),
        _case(
            'interp3d_local_strided',
            kernels.bench_3d_interp,
            (
                three.x0_local,
                three.x1_local,
                three.x2_local,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp_strided[0],
                three.local_hints,
            ),
            ('numba', 'numba.scalar', '3d'),
        ),
        _case(
            'array3d_local',
            kernels.bench_3d_array_reuse,
            (
                three.x0_local,
                three.x1_local,
                three.x2_local,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                np.zeros(3, dtype=np.int64),
                np.empty_like(three.x0_local),
            ),
            ('core', 'numba', 'numba.array', '3d'),
        ),
        _case(
            'array3d_random',
            kernels.bench_3d_array_reuse,
            (
                three.x0_random,
                three.x1_random,
                three.x2_random,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                np.zeros(3, dtype=np.int64),
                np.empty_like(three.x0_random),
            ),
            ('numba', 'numba.array', '3d'),
        ),
        _case(
            'array3d_extrapolate',
            kernels.bench_3d_array_reuse,
            (
                three.x0_extrapolated,
                three.x1_extrapolated,
                three.x2_extrapolated,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                np.zeros(3, dtype=np.int64),
                np.empty_like(three.x0_extrapolated),
            ),
            ('numba', 'numba.array', '3d'),
        ),
        _case(
            'array3d_allocate',
            kernels.bench_3d_array_allocate,
            (
                three.x0_local,
                three.x1_local,
                three.x2_local,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
            ),
            ('numba', 'numba.array', '3d'),
        ),
        _case(
            'python1d_array',
            kernels.bench_python_1d_array,
            (
                one.local.reshape(64, 32),
                one.xp,
                one.fp[0],
                np.empty((64, 32), dtype=np.float64),
            ),
            ('python', 'python.array', '1d'),
        ),
        _case(
            'python2d_broadcast',
            kernels.bench_python_2d_broadcast,
            (
                two.x0_broadcast,
                two.x1_broadcast,
                two.xp0,
                two.xp1,
                two.fp[0],
                np.empty((64, 32), dtype=np.float64),
            ),
            ('python', 'python.array', '2d'),
        ),
        _case(
            'python3d_broadcast',
            kernels.bench_python_3d_broadcast,
            (
                three.x0_broadcast,
                three.x1_broadcast,
                three.x2_broadcast,
                three.xp0,
                three.xp1,
                three.xp2,
                three.fp[0],
                np.empty((8, 16, 16), dtype=np.float64),
            ),
            ('python', 'python.array', '3d'),
        ),
    ]

    return cases
