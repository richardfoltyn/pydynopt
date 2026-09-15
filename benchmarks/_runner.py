"""Shared deterministic benchmark measurement and reporting.

Author: Richard Foltyn
"""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean, median, pstdev
import subprocess
from time import perf_counter_ns
from typing import Any

COMPONENTS = ('interpolate',)
MAX_ROUNDS = 100_000
MIN_PROBE_NS = 5_000_000
NOISY_CV_PCT = 2.0
SAMPLES = 11
SCHEMA_VERSION = 1
SUITE_VERSION = 1
TARGET_SAMPLE_NS = 100_000_000
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkCase:
    """Describe one benchmark workload."""

    name: str
    func: Callable[..., float]
    args: tuple[object, ...]
    operations_per_round: int
    groups: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Store the measurements for one benchmark workload."""

    name: str
    median_ns: float
    relative_mad_pct: float
    cv_pct: float
    rounds: int
    samples_ns: tuple[float, ...]
    groups: tuple[str, ...]


def _check_affinity(cpu: int) -> None:
    """Verify that the process remains pinned to the selected CPU."""
    affinity = os.sched_getaffinity(0)
    if affinity != {cpu}:
        msg = f'benchmark affinity changed: expected [{cpu}], got {sorted(affinity)}'
        raise RuntimeError(msg)


def _run_case(case: BenchmarkCase, rounds: int) -> float:
    """Execute a case and validate its checksum."""
    result = float(case.func(*case.args, rounds))
    if not math.isfinite(result):
        msg = f'{case.name} produced a non-finite checksum'
        raise RuntimeError(msg)

    return result


def _warm_cases(cases: Sequence[BenchmarkCase], cpu: int) -> None:
    """Compile and warm every case before measuring any case."""
    for case in cases:
        _check_affinity(cpu)
        _run_case(case, 1)


def _calibrate(case: BenchmarkCase, cpu: int) -> int:
    """Choose enough workload rounds to reach the sample-time target."""
    probe_rounds = 1
    elapsed = 0
    while True:
        _check_affinity(cpu)
        start = perf_counter_ns()
        _run_case(case, probe_rounds)
        elapsed = perf_counter_ns() - start
        if elapsed >= MIN_PROBE_NS or probe_rounds >= MAX_ROUNDS:
            break

        scale = max(2, math.ceil(MIN_PROBE_NS / max(elapsed, 1)))
        probe_rounds = min(MAX_ROUNDS, probe_rounds * scale)

    rounds = math.ceil(TARGET_SAMPLE_NS * probe_rounds / max(elapsed, 1))
    result = min(MAX_ROUNDS, max(1, rounds))

    return result


def _time_case(case: BenchmarkCase, cpu: int) -> BenchmarkResult:
    """Measure one case and return robust summary statistics."""
    rounds = _calibrate(case, cpu)
    _run_case(case, max(1, rounds // 4))

    samples: list[float] = []
    for _ in range(SAMPLES):
        _check_affinity(cpu)
        start = perf_counter_ns()
        _run_case(case, rounds)
        duration = perf_counter_ns() - start
        operations = rounds * case.operations_per_round
        samples.append(duration / operations)

    value = median(samples)
    mad = median(abs(sample - value) for sample in samples)
    relative_mad_pct = 100.0 * mad / value
    cv_pct = 100.0 * pstdev(samples) / fmean(samples)
    result = BenchmarkResult(
        name=case.name,
        median_ns=value,
        relative_mad_pct=relative_mad_pct,
        cv_pct=cv_pct,
        rounds=rounds,
        samples_ns=tuple(samples),
        groups=case.groups,
    )

    return result


def _geomean(values: Sequence[float]) -> float:
    """Return the geometric mean of positive values."""
    result = math.exp(fmean(math.log(value) for value in values))

    return result


def _load_cases(component: str) -> list[BenchmarkCase]:
    """Load one component lazily after CPU affinity is configured."""
    if component == 'interpolate':
        from benchmarks.interpolate import make_cases

        return make_cases()

    msg = f'unknown benchmark component: {component}'
    raise ValueError(msg)


def _package_version(name: str) -> str:
    """Return an installed package version or an unavailable marker."""
    try:
        result = version(name)
    except PackageNotFoundError:
        result = 'unavailable'

    return result


def _git_output(*args: str) -> str:
    """Run a read-only Git query for benchmark metadata."""
    try:
        result = subprocess.run(
            ('git', *args),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        result = 'unavailable'

    return result


def _cpu_model() -> str:
    """Return the Linux CPU model description."""
    try:
        lines = Path('/proc/cpuinfo').read_text().splitlines()
        result = next(
            line.split(':', maxsplit=1)[1].strip()
            for line in lines
            if line.startswith('model name')
        )
    except (OSError, StopIteration):
        result = platform.processor() or 'unavailable'

    return result


def _metadata(cpu: int) -> dict[str, Any]:
    """Collect environment metadata required for valid comparisons."""
    status = _git_output('status', '--porcelain')
    metadata = {
        'affinity': sorted(os.sched_getaffinity(0)),
        'commit': _git_output('rev-parse', 'HEAD'),
        'cpu': cpu,
        'cpu_model': _cpu_model(),
        'dirty': bool(status and status != 'unavailable'),
        'numba': _package_version('numba'),
        'numpy': _package_version('numpy'),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'timestamp_utc': datetime.now(UTC).isoformat(),
    }

    return metadata


def _case_payload(result: BenchmarkResult) -> dict[str, Any]:
    """Convert one result to its stable JSON representation."""
    payload = asdict(result)
    payload['samples_ns'] = list(result.samples_ns)

    return payload


def _run_component(
    component: str,
    cpu: int,
) -> tuple[list[BenchmarkResult], dict[str, float]]:
    """Warm and measure all cases for one component."""
    cases = _load_cases(component)
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        msg = f'{component} contains duplicate benchmark names'
        raise RuntimeError(msg)

    print(f'Warming {component}: {len(cases)} cases')
    _warm_cases(cases, cpu)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        results = []
        for case in cases:
            result = _time_case(case, cpu)
            results.append(result)
            marker = ' NOISY' if result.cv_pct > NOISY_CV_PCT else ''
            print(
                f'  {result.name:<29} {result.median_ns:>10.3f} ns/op '
                f'MAD {result.relative_mad_pct:>5.2f}% '
                f'CV {result.cv_pct:>5.2f}%{marker}'
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        for group in result.groups:
            grouped[group].append(result.median_ns)
    groups = {name: _geomean(values) for name, values in sorted(grouped.items())}

    print('Groups:')
    for name, value in groups.items():
        print(f'  {name:<29} {value:>10.3f} ns/op')

    return results, groups


def _parse_args(argv: Sequence[str] | None) -> Namespace:
    """Parse the benchmark command line."""
    parser = ArgumentParser(description='Run deterministic pydynopt benchmarks')
    parser.add_argument('component', choices=(*COMPONENTS, 'all'))
    parser.add_argument('--baseline', type=Path, help='JSON result to compare against')
    parser.add_argument('--output', type=Path, help='write JSON results to this path')

    return parser.parse_args(argv)


def _read_baseline(path: Path | None) -> dict[str, Any] | None:
    """Read an optional baseline result."""
    if path is None:
        return None

    result = json.loads(path.read_text())

    return result


def _validate_baseline(baseline: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Reject comparisons made across incompatible environments."""
    if baseline.get('schema_version') != SCHEMA_VERSION:
        msg = 'baseline schema version does not match'
        raise RuntimeError(msg)
    if baseline.get('suite_version') != SUITE_VERSION:
        msg = 'baseline benchmark suite version does not match'
        raise RuntimeError(msg)

    expected = baseline['metadata']
    keys = ('cpu', 'cpu_model', 'python', 'numpy', 'numba')
    mismatches = [
        f'{key}: {expected.get(key)!r} != {metadata.get(key)!r}'
        for key in keys
        if expected.get(key) != metadata.get(key)
    ]
    if mismatches:
        detail = '; '.join(mismatches)
        msg = f'baseline environment does not match: {detail}'
        raise RuntimeError(msg)


def _print_comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Print per-case and per-group changes from a baseline."""
    print('Comparison with baseline:')
    for component, values in current['components'].items():
        before = baseline.get('components', {}).get(component)
        if before is None:
            print(f'  {component}: absent from baseline')
            continue

        for section, key in (('cases', 'median_ns'), ('groups', None)):
            print(f'  {component} {section}:')
            old_values = before[section]
            for name, value in values[section].items():
                current_value = value if key is None else value[key]
                old = old_values.get(name)
                if old is None:
                    print(f'    {name:<27} new')
                    continue
                old_value = old if key is None else old[key]
                change = 100.0 * (current_value / old_value - 1.0)
                noisy = ''
                if key is not None and (
                    value['cv_pct'] > NOISY_CV_PCT or old['cv_pct'] > NOISY_CV_PCT
                ):
                    noisy = ' NOISY'
                print(f'    {name:<27} {change:>+8.2f}%{noisy}')


def run(cpu: int, argv: Sequence[str] | None = None) -> None:
    """Run selected benchmark components and report their results."""
    args = _parse_args(argv)
    metadata = _metadata(cpu)
    baseline = _read_baseline(args.baseline)
    if baseline is not None:
        _validate_baseline(baseline, metadata)

    print(
        f'CPU {cpu}: {metadata["cpu_model"]}; '
        f'Python {metadata["python"]}; NumPy {metadata["numpy"]}; '
        f'Numba {metadata["numba"]}'
    )
    print(
        f'{SAMPLES} samples/case; target {TARGET_SAMPLE_NS / 1.0e6:.0f} ms/sample; '
        f'noisy CV > {NOISY_CV_PCT:.1f}%'
    )

    selected = COMPONENTS if args.component == 'all' else (args.component,)
    components: dict[str, Any] = {}
    for component in selected:
        results, groups = _run_component(component, cpu)
        components[component] = {
            'cases': {result.name: _case_payload(result) for result in results},
            'groups': groups,
        }

    payload = {
        'schema_version': SCHEMA_VERSION,
        'suite_version': SUITE_VERSION,
        'samples': SAMPLES,
        'target_sample_ns': TARGET_SAMPLE_NS,
        'metadata': metadata,
        'components': components,
    }
    if baseline is not None:
        _print_comparison(payload, baseline)
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        print(f'Wrote {args.output}')
