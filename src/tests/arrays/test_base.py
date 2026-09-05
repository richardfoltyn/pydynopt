"""Test probability clipping and array grid creation."""

from typing import Any

import numpy as np
import pytest

from pydynopt import arrays
from pydynopt.arrays import clip_prob, logspace, powerspace

_clip_prob_any: Any = clip_prob
_powerspace_any: Any = powerspace


def test_package_root_exports_remain_stable() -> None:
    assert arrays.__all__ == [
        'clip_prob',
        'ind2sub',
        'logspace',
        'powerspace',
        'sub2ind',
    ]


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (0.05, 0.0),
        (0.1, 0.1),
        (0.5, 0.5),
        (0.9, 0.9),
        (0.95, 1.0),
        (np.float32(0.05), 0.0),
        (np.int64(1), 1.0),
    ],
)
def test_clip_prob_scalars(value: Any, expected: float) -> None:
    result = clip_prob(value, 0.1)
    assert type(result) is float
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    'value',
    [
        [0.05, 0.1, 0.5, 0.9, 0.95],
        (0.05, 0.1, 0.5, 0.9, 0.95),
        np.array([0.05, 0.1, 0.5, 0.9, 0.95], dtype=np.float32),
        np.array([0, 1]),
    ],
)
def test_clip_prob_arrays_and_sequences(value: Any) -> None:
    expected = np.asarray(value, dtype=np.float64)
    expected[expected < 0.1] = 0.0
    expected[expected > 0.9] = 1.0

    result = clip_prob(value, 0.1)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_clip_prob_preserves_multidimensional_and_zero_dimensional_shapes() -> None:
    value = np.array([[0.05, 0.5], [0.9, 0.95]])
    result = clip_prob(value, 0.1)
    assert result.shape == value.shape
    np.testing.assert_allclose(result, [[0.0, 0.5], [0.9, 1.0]])

    scalar_array = np.array(0.05)
    scalar_result = clip_prob(scalar_array, 0.1)
    assert isinstance(scalar_result, np.ndarray)
    assert scalar_result.shape == ()
    assert scalar_result.item() == 0.0


def test_clip_prob_output_is_fully_written_and_returned() -> None:
    value = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
    out = np.full(value.shape, -99.0)
    result = clip_prob(value, 0.1, out=out)

    assert result is out
    np.testing.assert_allclose(out, [0.0, 0.1, 0.5, 0.9, 1.0])


@pytest.mark.parametrize('tol', [-0.1, 0.5001, np.nan, np.inf])
def test_clip_prob_rejects_invalid_tolerance(tol: float) -> None:
    with pytest.raises(ValueError, match=r'0 <= tol <= 0\.5'):
        clip_prob([0.5], tol)


@pytest.mark.parametrize(
    'out',
    [
        np.empty(3, dtype=np.float64),
        np.empty(2, dtype=np.float32),
        np.empty(2, dtype=np.int64),
    ],
)
def test_clip_prob_rejects_invalid_output_buffers(out: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        clip_prob([0.25, 0.75], 0.1, out=out)


def test_clip_prob_rejects_read_only_and_scalar_outputs() -> None:
    out = np.empty(2)
    out.flags.writeable = False
    with pytest.raises(ValueError, match='writable'):
        clip_prob([0.25, 0.75], 0.1, out=out)

    with pytest.raises(TypeError, match='do not accept'):
        _clip_prob_any(0.25, 0.1, out=np.empty(1))


@pytest.mark.parametrize('value', [True, 1 + 2j, ['x'], np.array([True])])
def test_clip_prob_rejects_non_real_inputs(value: Any) -> None:
    with pytest.raises(TypeError):
        _clip_prob_any(value, 0.1)


@pytest.mark.parametrize('exponent', [0.5, 1.0, 2.0, np.float32(3.0)])
def test_powerspace_matches_reference(exponent: Any) -> None:
    result = powerspace(1.0, 5.0, 7, exponent)
    expected = 1.0 + 4.0 * np.linspace(0.0, 1.0, 7) ** float(exponent)
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64
    assert result[0] == 1.0
    assert result[-1] == 5.0


def test_powerspace_preserves_flipped_boundary_behavior() -> None:
    result = powerspace(5.0, 1.0, 5, 2.0)
    expected = (5.0 - 4.0 * np.linspace(0.0, 1.0, 5) ** 2.0)[::-1]
    np.testing.assert_allclose(result, expected)
    assert result[0] == 1.0
    assert result[-1] == 5.0


def test_powerspace_one_point_grid_preserves_existing_endpoint_choice() -> None:
    np.testing.assert_array_equal(powerspace(1.0, 5.0, 1, 2.0), [5.0])
    np.testing.assert_array_equal(powerspace(5.0, 1.0, 1, 2.0), [5.0])


@pytest.mark.parametrize('n', [0, -1])
def test_powerspace_rejects_invalid_size(n: int) -> None:
    with pytest.raises(ValueError, match='at least'):
        powerspace(0.0, 1.0, n, 1.0)


@pytest.mark.parametrize('n', [1.5, True])
def test_powerspace_rejects_non_integer_size(n: Any) -> None:
    with pytest.raises(TypeError, match='integer'):
        _powerspace_any(0.0, 1.0, n, 1.0)


@pytest.mark.parametrize('exponent', [0.0, -1.0, np.nan, np.inf])
def test_powerspace_rejects_invalid_exponent(exponent: float) -> None:
    with pytest.raises(ValueError, match='strictly positive'):
        powerspace(0.0, 1.0, 3, exponent)


def test_logspace_default_and_explicit_shift() -> None:
    result = logspace(1.0, 100.0, 5)
    np.testing.assert_allclose(result, np.geomspace(1.0, 100.0, 5))

    shifted = logspace(0.0, 8.0, 5, log_shift=1.0)
    expected = np.exp(np.linspace(np.log(1.0), np.log(9.0), 5)) - 1.0
    expected[[0, -1]] = [0.0, 8.0]
    np.testing.assert_allclose(shifted, expected)


def test_logspace_reference_fraction_places_requested_grid_point() -> None:
    result = logspace(1.0, 101.0, 11, x0=11.0, frac_at_x0=0.7)
    assert result[0] == 1.0
    assert result[-1] == 101.0
    assert result[7] == pytest.approx(11.0)


def test_logspace_inserted_values_preserve_order_length_and_endpoints() -> None:
    result = logspace(1.0, 100.0, 6, insert_vals=[20.0, 3.0])
    assert len(result) == 6
    assert result[0] == 1.0
    assert result[-1] == 100.0
    assert np.all(result[1:] >= result[:-1])
    assert 3.0 in result
    assert 20.0 in result


@pytest.mark.parametrize('frac', [0.0, 1.0, -0.1, 1.1, np.nan])
def test_logspace_rejects_invalid_fractions(frac: float) -> None:
    with pytest.raises(ValueError):
        logspace(1.0, 100.0, 5, frac_at_x0=frac)


@pytest.mark.parametrize('x0', [1.0, 101.0, 0.0, np.inf])
def test_logspace_rejects_invalid_reference_points(x0: float) -> None:
    with pytest.raises(ValueError, match='x0'):
        logspace(1.0, 101.0, 5, x0=x0, frac_at_x0=0.8)


@pytest.mark.parametrize(
    ('start', 'stop', 'shift'),
    [
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (np.nan, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 2.0, -1.0),
    ],
)
def test_logspace_rejects_invalid_domains(
    start: float, stop: float, shift: float
) -> None:
    with pytest.raises(ValueError):
        logspace(start, stop, 5, log_shift=shift)


@pytest.mark.parametrize('inserted', [[2.0, 3.0, 4.0], [0.5], [11.0]])
def test_logspace_rejects_invalid_insertions(inserted: list[float]) -> None:
    with pytest.raises(ValueError):
        logspace(1.0, 10.0, 4, insert_vals=inserted)
