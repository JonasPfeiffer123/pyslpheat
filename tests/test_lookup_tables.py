"""
The lookup tables built from the bundled CSV files must answer exactly like
the pandas merges and row filters they replace.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
import pytest

import pyslpheat
from pyslpheat import bdew, vdi4655

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


def test_hourly_factors_are_numbers(hourly_data):
    # Up to 0.4.0 three GMF factors were written with a decimal comma ('5,18'),
    # which made the whole column text and every GMF calculation fail.
    factors = hourly_data["Stundenfaktor"]
    assert factors.dtype == np.float64
    assert np.isfinite(factors).all()


def test_hourly_factors_of_a_day_sum_to_100_percent(hourly_data):
    sums = hourly_data.groupby(["Typ", "Wochentag", "Temperatur"])["Stundenfaktor"].sum()
    assert len(sums) == len(PROFILE_TYPES) * 7 * 10
    # The published tables are rounded to two decimals per hour
    assert (sums - 100.0).abs().max() < 0.1


@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_hourly_factors_match_merge(profile_type, hourly_data):
    # Tabulated classes, classes outside the table, an untabulated value and NaN;
    # weekday 8 is not tabulated either
    temperatures = list(np.arange(-22.5, 35.0, 5.0)) + [3.0, np.nan]
    weekdays, temperature_class = _conditions(temperatures, weekdays=range(1, 9))
    expected = _merge_reference(hourly_data, profile_type, weekdays, temperature_class)
    actual = bdew._hourly_factors(profile_type, weekdays, temperature_class)
    assert actual.dtype == expected.dtype
    assert np.isnan(expected).any() and not np.isnan(expected).all()
    assert np.array_equal(actual, expected, equal_nan=True)


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


# ── VDI 4655 ─────────────────────────────────────────────────────────────────

def test_daily_factors_match_factor_table():
    factor_data = pd.read_csv(os.path.join(vdi4655._VDI4655_DATA_DIR, "Faktoren.csv"), sep=";")
    listed = factor_data[factor_data["Profiltag"].notna()]
    assert listed["Profiltag"].is_unique
    factors = vdi4655._daily_factors()
    assert set(factors) == set(listed["Profiltag"])
    for tag in listed["Profiltag"]:
        # Row selection as implemented in v0.3.0
        index = factor_data[factor_data["Profiltag"] == tag].index[0]
        expected = tuple(factor_data.loc[index, c] for c in ("Fheiz,TT", "Fel,TT", "FTWW,TT"))
        assert factors[tag] == expected


def test_unknown_climate_zone_warns_once_per_day(caplog):
    holidays = np.array([], dtype="datetime64[D]")
    with caplog.at_level(logging.WARNING, logger="pyslpheat.vdi4655"):
        vdi4655.calculate(15_000.0, 3_000.0, 3_500.0, "EFH", 2, 2023, "99",
                          pyslpheat.TRY_BAUTZEN_2015, holidays)
    messages = [r.getMessage() for r in caplog.records if "No factors found" in r.getMessage()]
    assert len(messages) == 365
    assert messages[0].startswith("No factors found for profile day EFH99W")


def test_profile_files_list_every_interval_once():
    files = sorted(glob.glob(os.path.join(vdi4655._VDI4655_DATA_DIR, "load_profiles", "*.csv")))
    assert len(files) == 20
    for path in files:
        times = pd.read_csv(path, sep=";")["Zeit"].dropna()
        assert times.tolist() == vdi4655._PROFILE_TIMES, os.path.basename(path)


def test_loaded_profiles_are_read_only():
    profile = vdi4655._load_profile("EFHWWH")
    assert profile.shape == (3, 96)
    assert not profile.flags.writeable
