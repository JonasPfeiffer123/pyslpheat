"""
Leap years: a calculation for a leap year covers all 366 days.

DWD TRY files hold 8760 hours. For a leap year the weather of 28 February is
repeated for 29 February; weekdays and holidays follow the real calendar. A
file that already holds 8784 hours is used as it is.

Non-leap years must stay bit for bit as in 0.4.1; that is covered by the
golden-master tests (``test_golden_master.py``), whose non-leap cases were
recorded before leap-year support existed.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

import pyslpheat
from pyslpheat import bdew, vdi4655

from golden_cases import write_leap_try

TRY = pyslpheat.TRY_BAUTZEN_2015
LEAP_YEARS = [2024, 2028]
BDEW_PROFILES = [("HEF", "03"), ("HMF", "03"), ("GKO", "03")]
VDI_TYPES = ["EFH", "MFH"]
RTOL = 1e-9

BDEW_OPTIONS = {
    "plain": {},
    "dhw_share": dict(dhw_share=0.2),
    "limit_exponent": dict(dhw_share=0.3, heating_limit_temp=15.0, heating_exponent=1.4),
    "dhw_flat": dict(dhw_share=0.25, dhw_flat=True),
    "stochastic": dict(dhw_share=0.2, stochastic=True, stochastic_seed=7),
    "draw_events": dict(dhw_share=0.2, dhw_draw_events=True),
}


@pytest.fixture(scope="module")
def leap_try(tmp_path_factory):
    """TRY file with 8784 hours: the bundled 2015 file with 28 February repeated."""
    return write_leap_try(str(tmp_path_factory.mktemp("leap")))


def _bdew(year, profile=("HMF", "03"), try_path=TRY, **kwargs):
    params = dict(annual_heat_kWh=20_000.0, profile_type=profile[0], subtype=profile[1],
                  TRY_file_path=try_path, year=year)
    params.update(kwargs)
    return bdew.calculate(**params)


def _vdi(year, building_type="EFH", try_path=TRY, **kwargs):
    params = dict(annual_heating_kWh=16_000.0, annual_dhw_kWh=4_000.0,
                  annual_electricity_kWh=3_000.0, building_type=building_type,
                  number_people_household=2, year=year, climate_zone="9", TRY=try_path,
                  holidays=np.array(sorted(bdew.compute_holidays(year)), dtype="datetime64[D]"))
    params.update(kwargs)
    return vdi4655.calculate(**params)


def _file_temperature():
    return bdew.import_TRY(TRY)[0]


def _assert_weather_of_leap_year(hourly_temperature, year):
    """Hourly temperature of a leap year built from the 365-day file."""
    source = _file_temperature()
    feb28, mar1 = 58 * 24, 59 * 24
    assert len(hourly_temperature) == 8784
    np.testing.assert_array_equal(hourly_temperature[:mar1], source[:mar1])
    np.testing.assert_array_equal(hourly_temperature[mar1:mar1 + 24], source[feb28:mar1])
    np.testing.assert_array_equal(hourly_temperature[mar1 + 24:], source[mar1:])


def _assert_frames_identical(actual, expected):
    assert list(actual.columns) == list(expected.columns)
    assert actual.index.equals(expected.index)
    for column in expected.columns:
        assert actual[column].values.tobytes() == expected[column].values.tobytes(), column


# ── Calendar and weather ─────────────────────────────────────────────────────

@pytest.mark.parametrize("year", LEAP_YEARS)
@pytest.mark.parametrize("profile", BDEW_PROFILES)
def test_bdew_covers_every_hour_of_the_leap_year(year, profile):
    df = _bdew(year, profile)

    assert len(df) == 8784
    assert (df.index == pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")).all()
    feb28, feb29 = df.loc[f"{year}-02-28"], df.loc[f"{year}-02-29"]
    assert len(feb29) == 24
    np.testing.assert_array_equal(feb29["temperature_C"].values, feb28["temperature_C"].values)
    _assert_weather_of_leap_year(df["temperature_C"].values, year)


@pytest.mark.parametrize("year", LEAP_YEARS)
@pytest.mark.parametrize("building_type", VDI_TYPES)
def test_vdi_covers_every_interval_of_the_leap_year(year, building_type):
    df = _vdi(year, building_type)

    assert len(df) == 35_136
    assert df.index.equals(pd.date_range(f"{year}-01-01", periods=35_136, freq="15min"))
    feb28, feb29 = df.loc[f"{year}-02-28"], df.loc[f"{year}-02-29"]
    assert len(feb29) == 96
    np.testing.assert_array_equal(feb29["temperature_C"].values, feb28["temperature_C"].values)
    # Each hourly temperature is repeated for the four quarter hours
    hourly = df["temperature_C"].values.reshape(-1, 4)
    assert (hourly == hourly[:, :1]).all()
    _assert_weather_of_leap_year(hourly[:, 0], year)
    # Summing to hourly values works as for normal years
    assert len(df.resample("h").sum()) == 8784


@pytest.mark.parametrize("module", [bdew, vdi4655])
@pytest.mark.parametrize("year", LEAP_YEARS)
def test_calendar_of_the_leap_year(module, year):
    # Weekday codes follow the real calendar, in whatever numbering the module
    # uses for normal years
    days, months, day_of_month, weekdays = module.generate_year_months_days_weekdays(year)
    assert len(days) == 366
    assert days[59] == np.datetime64(f"{year}-02-29")
    assert (months[59], day_of_month[59]) == (2, 29)

    normal_days, _, _, normal_weekdays = module.generate_year_months_days_weekdays(2023)
    code_of_day_name = dict(zip(pd.DatetimeIndex(normal_days).day_name(), normal_weekdays))
    expected = [code_of_day_name[name] for name in pd.DatetimeIndex(days).day_name()]
    np.testing.assert_array_equal(weekdays, expected)


# ── Energy balance ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("year", [2023] + LEAP_YEARS)
@pytest.mark.parametrize("option", sorted(BDEW_OPTIONS))
@pytest.mark.parametrize("profile", BDEW_PROFILES)
def test_bdew_energy_per_component(year, option, profile):
    kwargs = BDEW_OPTIONS[option]
    df = _bdew(year, profile, **kwargs)

    assert df["Q_total_kWh"].sum() == pytest.approx(20_000.0, rel=RTOL)
    assert np.array_equal(df["Q_total_kWh"].values,
                          df["Q_heat_kWh"].values + df["Q_dhw_kWh"].values)
    share = kwargs.get("dhw_share")
    if share is not None:
        assert df["Q_dhw_kWh"].sum() == pytest.approx(share * 20_000.0, rel=RTOL)
        assert df["Q_heat_kWh"].sum() == pytest.approx((1 - share) * 20_000.0, rel=RTOL)


@pytest.mark.parametrize("year", [2023] + LEAP_YEARS)
@pytest.mark.parametrize("building_type", VDI_TYPES)
def test_vdi_energy_per_component(year, building_type):
    df = _vdi(year, building_type)

    assert df["Q_heat_kWh"].sum() == pytest.approx(16_000.0, rel=RTOL)
    assert df["Q_dhw_kWh"].sum() == pytest.approx(4_000.0, rel=RTOL)
    assert df["Q_electricity_kWh"].sum() == pytest.approx(3_000.0, rel=RTOL)
    assert df["Q_total_kWh"].sum() == pytest.approx(20_000.0, rel=RTOL)
    assert np.array_equal(df["Q_total_kWh"].values,
                          df["Q_heat_kWh"].values + df["Q_dhw_kWh"].values)


@pytest.mark.parametrize("year", LEAP_YEARS)
@pytest.mark.parametrize("profile", BDEW_PROFILES)
def test_bdew_design_load_only_in_leap_year(year, profile):
    # Mode B derives the annual energy from the design load. The profile must
    # be the one of Mode A scaled to that annual energy.
    mode_b = _bdew(year, profile, annual_heat_kWh=None, peak_design_kW=12.0,
                   design_temperature=-14.0, dhw_share=0.2)
    annual = mode_b["Q_total_kWh"].sum()
    mode_a = _bdew(year, profile, annual_heat_kWh=annual, dhw_share=0.2)
    assert len(mode_b) == 8784
    for column in ("Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh"):
        np.testing.assert_allclose(mode_b[column].values, mode_a[column].values, rtol=1e-10)


@pytest.mark.parametrize("year", LEAP_YEARS)
@pytest.mark.parametrize("profile", BDEW_PROFILES)
def test_bdew_annual_and_design_load_in_leap_year(year, profile, caplog):
    # Mode C meets the annual energy and the design-day heating power at once
    with caplog.at_level(logging.INFO, logger="pyslpheat.bdew"):
        df = _bdew(year, profile, annual_heat_kWh=20_000.0, peak_design_kW=12.0,
                   design_temperature=-14.0, dhw_share=0.2)
    assert len(df) == 8784
    assert df["Q_total_kWh"].sum() == pytest.approx(20_000.0, rel=RTOL)
    assert df["Q_dhw_kWh"].sum() == pytest.approx(4_000.0, rel=RTOL)
    (record,) = [r for r in caplog.records if r.getMessage().startswith("Mode C")]
    design_day_heating_kW = record.args[2]
    assert design_day_heating_kW == pytest.approx(12.0, rel=RTOL)


# ── Same result as with a weather file that already has 366 days ─────────────

@pytest.mark.parametrize("option", sorted(BDEW_OPTIONS))
@pytest.mark.parametrize("profile", BDEW_PROFILES)
def test_bdew_365_day_file_equals_366_day_file(option, profile, leap_try):
    kwargs = BDEW_OPTIONS[option]
    _assert_frames_identical(_bdew(2024, profile, **kwargs),
                             _bdew(2024, profile, try_path=leap_try, **kwargs))


@pytest.mark.parametrize("building_type", VDI_TYPES)
def test_vdi_365_day_file_equals_366_day_file(building_type, leap_try):
    _assert_frames_identical(_vdi(2024, building_type),
                             _vdi(2024, building_type, try_path=leap_try))


# ── Weather files of the wrong length ────────────────────────────────────────

def _write_try(directory, hours):
    """Bundled 2015 file cut or extended to *hours* data lines."""
    with open(TRY, "r", encoding="latin-1", newline="") as fh:
        lines = fh.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("***")) + 1
    header, data = lines[:start], lines[start:]
    data = (data * 2)[:hours]
    path = os.path.join(directory, f"TRY_{hours}h.dat")
    with open(path, "w", encoding="latin-1", newline="") as fh:
        fh.writelines(header + data)
    return path


@pytest.mark.parametrize("year, hours", [
    (2023, 8000), (2023, 8736), (2023, 8761), (2023, 8784),
    (2024, 8000), (2024, 8761), (2024, 8808),
])
@pytest.mark.parametrize("calculate", [_bdew, _vdi])
def test_wrong_weather_length_raises(year, hours, calculate, tmp_path):
    path = _write_try(str(tmp_path), hours)
    with pytest.raises(ValueError, match=f"{hours} hourly values") as info:
        calculate(year, try_path=path)
    assert str(year) in str(info.value)


# ── Caches hold the weather file as read, independent of the year ────────────

@pytest.mark.parametrize("calculate", [_bdew, _vdi])
def test_leap_year_leaves_cached_weather_untouched(calculate):
    pyslpheat.clear_caches()
    normal = calculate(2023)
    leap = calculate(2024)
    _assert_frames_identical(calculate(2023), normal)
    _assert_frames_identical(calculate(2024), leap)
    assert len(_file_temperature()) == 8760
