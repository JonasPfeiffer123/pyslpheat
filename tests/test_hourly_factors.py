"""
BDEW hourly factors: interpolation between the temperature classes, hours
outside the tabulated range and the daily energy.

The hourly factors are tabulated for the class centres -17.5 to 27.5 °C. Each
hour interpolates linearly between the two centres around its temperature. Up
to 0.4.2 the interpolation had the wrong sign: at the lower centre it gave
2·F(upper) − F(lower), so the factor jumped at every centre and could become
negative.

As every hour uses its own temperature, the 24 interpolated factors of a day
add up to 100 % only at a constant temperature. Since 0.4.4 they are scaled to
100 % per day, so every day keeps its daily demand.

The outer classes are open-ended (BGW P 2006/8, table 10: "Temp. <= -15 °C" and
"Temp. > 25 °C"), so colder and warmer hours take the factors of the outer
centres. Up to 0.4.2 such hours found no factor and got no load at all; their
energy went to the other hours of the year.
"""

import os

import numpy as np
import pandas as pd
import pytest

import pyslpheat
from pyslpheat import bdew

PROFILE_TYPES = ["HEF", "HMF", "GKO", "GHA", "GMK", "GBD", "GBH",
                 "GWA", "GGA", "GBA", "GGB", "GPD", "GMF", "GHD"]
# GMK, GWA, GBA and GGB have hours with a tabulated factor of 0 at 27.5 °C
WITHOUT_ZERO_FACTORS = [p for p in PROFILE_TYPES if p not in ("GMK", "GWA", "GBA", "GGB")]
BUNDLED_TRY = {
    "2015": pyslpheat.TRY_BAUTZEN_2015,
    "2015_winter": pyslpheat.TRY_BAUTZEN_2015_WINTER,
    "2015_summer": pyslpheat.TRY_BAUTZEN_2015_SUMMER,
    "2045": pyslpheat.TRY_BAUTZEN_2045,
    "2045_winter": pyslpheat.TRY_BAUTZEN_2045_WINTER,
    "2045_summer": pyslpheat.TRY_BAUTZEN_2045_SUMMER,
}
HOURLY_COEFFICIENTS = os.path.join(os.path.dirname(bdew.__file__),
                                   "data", "bdew", "hourly_coefficients.csv")
# 9 to 15 January 2023: one week without public holidays
WEEK = slice(8, 15)
ANNUAL_kWh = 10_000.0
RTOL = 1e-9


@pytest.fixture(scope="module")
def write_try(tmp_path_factory):
    """Return a function writing a copy of a TRY file with other hourly temperatures."""
    directory = tmp_path_factory.mktemp("try")

    def write(name, temperatures, source=pyslpheat.TRY_BAUTZEN_2015):
        path = directory / f"{name}.dat"
        if path.exists():
            return str(path)
        with open(source, "r", encoding="latin-1", newline="") as fh:
            lines = fh.readlines()
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("***")) + 1
        assert len(lines) - start == len(temperatures)
        data = []
        for line, temperature in zip(lines[start:], temperatures):
            parts = line.split()
            parts[5] = f"{temperature:.1f}"
            data.append(" ".join(parts) + "\n")
        with open(path, "w", encoding="latin-1", newline="") as fh:
            fh.writelines(lines[:start] + data)
        return str(path)

    return write


@pytest.fixture(scope="module")
def constant_try(write_try):
    """Return a function writing a TRY file with one temperature in every hour."""
    return lambda temperature: write_try(f"TRY_constant_{temperature:+.1f}",
                                         np.full(8760, temperature))


@pytest.fixture(scope="module")
def tabulated():
    """Return a function giving the 24 tabulated factors of one class, read from the CSV."""
    table = pd.read_csv(HOURLY_COEFFICIENTS, delimiter=";")

    def get(profile_type, weekday, temperature):
        rows = table[(table["Typ"] == profile_type) & (table["Wochentag"] == weekday)
                     & (table["Temperatur"] == temperature)].sort_values("Stunde")
        assert list(rows["Stunde"]) == list(range(24))
        return rows["Stundenfaktor"].values.astype(float)

    return get


def _bdew(profile_type, try_path):
    return bdew.calculate(ANNUAL_kWh, profile_type, "03", try_path, 2023)


# ── Interpolation between the classes ────────────────────────────────────────

@pytest.mark.parametrize("temperature", [-17.5, -16.0, -12.5, 0.0, 1.5, 12.5, 24.0, 27.5])
@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_constant_weather_interpolates_between_classes(
        profile_type, temperature, constant_try, tabulated):
    # At a constant temperature all hours of a day use the same interpolation
    # weight, so the day's load is split like the interpolated factors
    df = _bdew(profile_type, constant_try(temperature))
    lower = min(np.floor((temperature + 17.5) / 5) * 5 - 17.5, 22.5)
    weight = (temperature - lower) / 5
    weekdays = bdew.generate_year_months_days_weekdays(2023)[3]
    load = df["Q_total_kWh"].values.reshape(-1, 24)

    for day in range(len(load))[WEEK]:
        factors = ((1 - weight) * tabulated(profile_type, weekdays[day], lower)
                   + weight * tabulated(profile_type, weekdays[day], lower + 5))
        np.testing.assert_allclose(load[day] / load[day].sum(), factors / factors.sum(),
                                   rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("try_file", sorted(BUNDLED_TRY))
@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_bundled_weather_has_no_negative_load(profile_type, try_file):
    # Up to 0.4.2 HEF03 had 13 hours with negative load with the 2015 year file
    df = _bdew(profile_type, BUNDLED_TRY[try_file])

    assert (df["Q_heat_kWh"] >= 0).all()
    assert (df["Q_dhw_kWh"] >= 0).all()


# ── Hours outside the table ──────────────────────────────────────────────────


@pytest.mark.parametrize("temperature", [-30.0, -20.0, -17.6, 27.5, 27.6, 30.0, 35.0])
@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_constant_weather_beyond_the_table_keeps_the_annual_demand(
        profile_type, temperature, constant_try):
    # Up to 0.4.2 every hour got no load: at a constant -20 °C the whole demand was 0
    df = _bdew(profile_type, constant_try(temperature))

    assert np.isfinite(df.values).all()
    assert df["Q_heat_kWh"].sum() > 0
    assert df["Q_dhw_kWh"].sum() > 0
    assert df["Q_total_kWh"].sum() == pytest.approx(ANNUAL_kWh, rel=RTOL)


@pytest.mark.parametrize("temperature, outer_centre",
                         [(-30.0, -17.5), (-20.0, -17.5), (-17.6, -17.5),
                          (27.6, 27.5), (30.0, 27.5), (35.0, 27.5)])
@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_constant_weather_beyond_the_table_uses_outer_class(
        profile_type, temperature, outer_centre, constant_try):
    # At a constant temperature every day has the same daily heat demand, so the
    # total load differs only through the hourly factors
    beyond = _bdew(profile_type, constant_try(temperature))
    at_centre = _bdew(profile_type, constant_try(outer_centre))

    np.testing.assert_allclose(beyond["Q_total_kWh"].values,
                               at_centre["Q_total_kWh"].values, rtol=1e-12)


@pytest.mark.parametrize("profile_type", WITHOUT_ZERO_FACTORS)
def test_bundled_weather_hours_beyond_the_table_have_load(profile_type):
    # The 2015 winter file has hours below -17.5 °C and at or above 27.5 °C
    df = _bdew(profile_type, pyslpheat.TRY_BAUTZEN_2015_WINTER)
    temperature = df["temperature_C"].values
    beyond = (temperature < -17.5) | (temperature >= 27.5)
    assert (temperature < -17.5).any() and (temperature >= 27.5).any()

    assert (df["Q_heat_kWh"].values[beyond] > 0).all()
    assert (df["Q_dhw_kWh"].values[beyond] > 0).all()
    assert df["Q_total_kWh"].sum() == pytest.approx(ANNUAL_kWh, rel=RTOL)


# ── Daily energy ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("try_file", ["2015", "2045_summer"])
@pytest.mark.parametrize("profile_type", PROFILE_TYPES)
def test_daily_energy_depends_only_on_the_daily_mean(profile_type, try_file, write_try):
    # The daily demand follows from the daily mean temperatures. Up to 0.4.3 the
    # hourly split changed it: the 24 interpolated factors of a day did not add
    # up to 100 % when the temperature changed during the day
    hourly = bdew.import_TRY(BUNDLED_TRY[try_file])[0]
    daily_mean = np.round(bdew.calculate_daily_averages(hourly), 1)
    flat = write_try(f"TRY_daily_mean_{try_file}", np.repeat(daily_mean, 24),
                     source=BUNDLED_TRY[try_file])

    varying = _bdew(profile_type, BUNDLED_TRY[try_file])
    constant_per_day = _bdew(profile_type, flat)

    for column in ("Q_heat_kWh", "Q_dhw_kWh"):
        np.testing.assert_allclose(varying[column].values.reshape(-1, 24).sum(axis=1),
                                   constant_per_day[column].values.reshape(-1, 24).sum(axis=1),
                                   rtol=1e-12, err_msg=column)
