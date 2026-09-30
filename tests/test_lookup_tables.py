"""
The lookup tables built from the bundled CSV files must answer exactly like
the pandas merges and row filters they replace.
"""

import os

import numpy as np
import pandas as pd
import pytest

from pyslpheat import bdew

HOURLY_CSV = os.path.join(bdew._HERE, "hourly_coefficients.csv")
PROFILE_TYPES = ["GBA", "GBD", "GBH", "GGA", "GGB", "GHA", "GHD", "GKO", "GMF", "GMK",
                 "GPD", "GWA", "HEF", "HMF"]


@pytest.fixture(scope="module")
def hourly_data():
    return pd.read_csv(HOURLY_CSV, delimiter=";")


def _merge_reference(hourly_data, profile_type, daily_weekdays, temperatures):
    """Hourly factor lookup as implemented in v0.3.0."""
    conditions = pd.DataFrame({
        "Wochentag": np.repeat(daily_weekdays, 24),
        "T": temperatures,
        "Stunde": np.tile(np.arange(24), len(daily_weekdays)),
    })
    merged = pd.merge(
        conditions, hourly_data[hourly_data["Typ"] == profile_type], how="left",
        left_on=["Wochentag", "T", "Stunde"], right_on=["Wochentag", "Temperatur", "Stunde"])
    return merged["Stundenfaktor"].values.astype(float)


def _conditions(temperatures, weekdays=range(1, 8), days=200, seed=0):
    """Random weekday per day and random temperature class per hour."""
    rng = np.random.default_rng(seed)
    return (rng.choice(np.asarray(list(weekdays)), size=days),
            rng.choice(np.asarray(temperatures, dtype=float), size=days * 24))


def test_hourly_table_keys_are_unique(hourly_data):
    assert not hourly_data.duplicated(["Typ", "Wochentag", "Temperatur", "Stunde"]).any()
    assert sorted(hourly_data["Typ"].unique()) == PROFILE_TYPES


@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_hourly_factors_match_merge(profile_type, hourly_data):
    # Tabulated classes, classes outside the table, an untabulated value and NaN;
    # weekday 8 is not tabulated either
    temperatures = list(np.arange(-22.5, 35.0, 5.0)) + [3.0, np.nan]
    weekdays, temperature_class = _conditions(temperatures, weekdays=range(1, 9))
    try:
        expected = _merge_reference(hourly_data, profile_type, weekdays, temperature_class)
    except ValueError as exc:
        with pytest.raises(ValueError) as info:
            bdew._hourly_factors(profile_type, weekdays, temperature_class)
        assert str(info.value) == str(exc)
        return
    actual = bdew._hourly_factors(profile_type, weekdays, temperature_class)
    assert actual.dtype == expected.dtype
    assert np.isnan(expected).any() and not np.isnan(expected).all()
    assert np.array_equal(actual, expected, equal_nan=True)


def test_hourly_factors_with_unparsable_entries(hourly_data):
    # GMF holds three entries with a decimal comma at 22.5 °C. They must fail
    # only when selected, as with the merge.
    weekdays, temperature_class = _conditions([-17.5, 2.5, 17.5])
    expected = _merge_reference(hourly_data, "GMF", weekdays, temperature_class)
    actual = bdew._hourly_factors("GMF", weekdays, temperature_class)
    assert np.array_equal(actual, expected)

    weekdays, temperature_class = _conditions([22.5])
    with pytest.raises(ValueError, match="could not convert string to float: '5,18'"):
        bdew._hourly_factors("GMF", weekdays, temperature_class)


def test_hourly_factors_unknown_profile_type(hourly_data):
    weekdays, temperature_class = _conditions([-17.5, 2.5])
    expected = _merge_reference(hourly_data, "XXX", weekdays, temperature_class)
    actual = bdew._hourly_factors("XXX", weekdays, temperature_class)
    assert np.isnan(expected).all()
    assert np.array_equal(actual, expected, equal_nan=True)


def test_profile_parameters_match_daily_table():
    daily_data = pd.read_csv(os.path.join(bdew._HERE, "daily_coefficients.csv"), delimiter=";")
    assert daily_data["Standardlastprofil"].is_unique
    for profile in daily_data["Standardlastprofil"]:
        coefficients, weekday_factors = bdew._profile_parameters(profile)
        assert coefficients == bdew.get_coefficients(profile[:3], profile[3:], daily_data)
        row = daily_data[daily_data["Standardlastprofil"] == profile].iloc[0]
        assert weekday_factors.tolist() == [float(row[str(day)]) for day in range(1, 8)]
        assert not weekday_factors.flags.writeable
