"""Run repository benchmark suites on one fixed logical CPU.

Author: Richard Foltyn
"""

import os

_THREAD_ENV_VARS = (
    'MKL_NUM_THREADS',
    'NUMBA_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)


def _configure_runtime() -> int:
    """Configure a single-threaded runtime and pin it to one logical CPU."""
    for name in _THREAD_ENV_VARS:
        os.environ[name] = '1'

    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        msg = 'benchmarks require Linux CPU-affinity support'
        raise RuntimeError(msg)

    cpu_count = os.cpu_count() or 1
    cpu = min(7, cpu_count - 1)
    allowed = os.sched_getaffinity(0)
    if cpu not in allowed:
        msg = f'benchmark CPU {cpu} is unavailable; allowed CPUs: {sorted(allowed)}'
        raise RuntimeError(msg)

    os.sched_setaffinity(0, {cpu})
    affinity = os.sched_getaffinity(0)
    if affinity != {cpu}:
        msg = f'failed to pin benchmark to CPU {cpu}; affinity is {sorted(affinity)}'
        raise RuntimeError(msg)

    return cpu


def main() -> None:
    """Configure the process and dispatch to the benchmark runner."""
    try:
        cpu = _configure_runtime()
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    from benchmarks._runner import run

    run(cpu)


if __name__ == '__main__':
    main()
