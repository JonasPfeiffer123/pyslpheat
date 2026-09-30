"""
The public helper functions must return exactly what the reference
implementations of v0.3.0 return (see ``reference_impl.py``).
"""

import os

import numpy as np
import pandas as pd
import pytest

import pyslpheat
from pyslpheat import bdew, vdi4655

import reference_impl as ref
from golden_cases import write_leap_try

TYPE_DAYS = ["WWH", "WWB", "WSH", "WSB", "ÜWH", "ÜWB", "ÜSH", "ÜSB", "SWX", "SSX"]

BUNDLED_TRY = [
    pyslpheat.TRY_BAUTZEN_2015, pyslpheat.TRY_BAUTZEN_2015_WINTER,
    pyslpheat.TRY_BAUTZEN_2015_SUMMER, pyslpheat.TRY_BAUTZEN_2045,
    pyslpheat.TRY_BAUTZEN_2045_WINTER, pyslpheat.TRY_BAUTZEN_2045_SUMMER,
]


@pytest.fixture(scope="module")
def try_files(tmp_path_factory):
    return BUNDLED_TRY + [write_leap_try(str(tmp_path_factory.mktemp("try")))]


@pytest.fixture(scope="module")
def daily_data():
    return pd.read_csv(os.path.join(bdew._HERE, "daily_coefficients.csv"), delimiter=";")


def _same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_bdew_import_try(try_files):
    for path in try_files:
        actual = bdew.import_TRY(path)
        expected = ref.bdew_import_TRY(path)
        _same(actual[0], expected[0])
        assert actual[1:] == (None, None, None, None)


def test_vdi_import_try(try_files):
    for path in try_files:
        actual = vdi4655.import_TRY(path)
        expected = ref.vdi_import_TRY(path)
        assert len(actual) == 5
        for a, e in zip(actual, expected):
            _same(a, e)


def test_import_try_missing_file(tmp_path):
    missing = str(tmp_path / "missing.dat")
    for func in (bdew.import_TRY, vdi4655.import_TRY):
        with pytest.raises(FileNotFoundError) as info:
            func(missing)
        with pytest.raises(FileNotFoundError) as expected:
            open(missing, "r", encoding="latin-1")
        assert str(info.value) == str(expected.value)


def test_allocation_temperature():
    rng = np.random.default_rng(0)
    inputs = [rng.normal(8.0, 9.0, size=n) for n in (0, 1, 2, 3, 4, 365, 366)]
    inputs.append(np.round(bdew.calculate_daily_averages(bdew.import_TRY(BUNDLED_TRY[0])[0]), 1))
    for values in inputs:
        _same(bdew.calculate_allocation_temperature(values),
              ref.bdew_calculate_allocation_temperature(values))


@pytest.mark.parametrize("profile", ["HEF03", "GKO03", "GBD03", "GHA34", "GMF05", "GHD33"])
@pytest.mark.parametrize("year", [2023, 2024])
def test_weekday_factor(profile, year, daily_data):
    weekdays = bdew.generate_year_months_days_weekdays(year)[3]
    _same(bdew.get_weekday_factor(weekdays, profile[:3], profile[3:], daily_data),
          ref.bdew_get_weekday_factor(weekdays, profile[:3], profile[3:], daily_data))


def test_weekday_factor_errors(daily_data):
    weekdays = bdew.generate_year_months_days_weekdays(2023)[3]
    with pytest.raises(ValueError, match="Profile 'XXX99' not found"):
        bdew.get_weekday_factor(weekdays, "XXX", "99", daily_data)
    with pytest.raises(KeyError, match="Missing weekday column in BDEW data"):
        bdew.get_weekday_factor(np.array([1, 2, 8]), "GBD", "03", daily_data)


@pytest.mark.parametrize("building_type", ["EFH", "MFH", "XYZ"])
@pytest.mark.parametrize("year", [2023, 2024])
def test_standardized_quarter_hourly_profile(building_type, year):
    days_of_year = vdi4655.generate_year_months_days_weekdays(year)[0]
    rng = np.random.default_rng(year)
    type_days = np.array(TYPE_DAYS)[rng.integers(0, len(TYPE_DAYS), size=len(days_of_year))]
    actual = vdi4655.standardized_quarter_hourly_profile(
        year, building_type, days_of_year, type_days)
    expected = ref.vdi_standardized_quarter_hourly_profile(
        year, building_type, days_of_year, type_days)
    assert len(actual) == 4
    for a, e in zip(actual, expected):
        _same(a, e)


def test_standardized_quarter_hourly_profile_short_type_days():
    # Fewer type days than days in the year: the reference wraps around
    days_of_year = vdi4655.generate_year_months_days_weekdays(2023)[0]
    type_days = np.array(TYPE_DAYS)
    actual = vdi4655.standardized_quarter_hourly_profile(2023, "MFH", days_of_year, type_days)
    expected = ref.vdi_standardized_quarter_hourly_profile(2023, "MFH", days_of_year, type_days)
    for a, e in zip(actual, expected):
        _same(a, e)


def _series(index, seed=0):
    values = np.random.default_rng(seed).gamma(2.0, 3.0, size=len(index))
    return pd.Series(values, index=index)


def _stochastic_indexes():
    full_year = pd.DatetimeIndex(bdew.calculate_hourly_intervals(2023).astype("datetime64[s]"))
    leap_year = pd.DatetimeIndex(bdew.calculate_hourly_intervals(2024).astype("datetime64[s]"))
    # Incomplete first and last day
    partial = pd.date_range("2023-03-01 05:00", periods=24 * 9 + 7, freq="h")
    # More than 24 values per day, most of them not on the full hour
    half_hourly = pd.date_range("2023-06-01", periods=48 * 12, freq="30min")
    # Same day not contiguous
    shuffled = full_year[:24 * 30][np.random.default_rng(5).permutation(24 * 30)]
    return {"full_year": full_year, "leap_year": leap_year, "partial": partial,
            "half_hourly": half_hourly, "shuffled": shuffled}


@pytest.mark.parametrize("name", ["full_year", "leap_year", "partial", "half_hourly", "shuffled"])
@pytest.mark.parametrize("max_shift", [0, 1, 3])
def test_peak_jitter(name, max_shift):
    series = _series(_stochastic_indexes()[name])
    rng_actual, rng_expected = np.random.default_rng(42), np.random.default_rng(42)
    actual = bdew._apply_peak_jitter(series, max_shift, rng_actual)
    expected = ref.bdew_apply_peak_jitter(series, max_shift, rng_expected)
    assert actual.index.equals(expected.index)
    _same(actual.values, expected.values)
    # The random stream must be left in the same state
    assert rng_actual.random() == rng_expected.random()


@pytest.mark.parametrize("name", ["full_year", "leap_year", "partial", "half_hourly", "shuffled"])
@pytest.mark.parametrize("draws_per_day", [0.0, 4.0, 9.5])
def test_dhw_draw_events(name, draws_per_day):
    series = _series(_stochastic_indexes()[name], seed=1)
    actual = bdew._apply_dhw_draw_events(series, draws_per_day, 7)
    expected = ref.bdew_apply_dhw_draw_events(series, draws_per_day, 7)
    assert actual.index.equals(expected.index)
    _same(actual.values, expected.values)
