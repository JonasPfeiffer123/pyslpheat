"""
BDEW hourly factors outside the tabulated temperature range.

The hourly factors are tabulated for the class centres -17.5 to 27.5 °C. The
outer classes are open-ended (BGW P 2006/8, table 10: "Temp. <= -15 °C" and
"Temp. > 25 °C"), so colder and warmer hours take the factors of the outer
centres. Up to 0.4.2 such hours found no factor and got no load at all; their
energy went to the other hours of the year.
"""

import numpy as np
import pytest

import pyslpheat
from pyslpheat import bdew

PROFILE_TYPES = ["HEF", "HMF", "GKO", "GHA", "GMK", "GBD", "GBH",
                 "GWA", "GGA", "GBA", "GGB", "GPD", "GMF", "GHD"]
# GMK, GWA, GBA and GGB have hours with a tabulated factor of 0 at 27.5 °C
WITHOUT_ZERO_FACTORS = [p for p in PROFILE_TYPES if p not in ("GMK", "GWA", "GBA", "GGB")]
ANNUAL_kWh = 10_000.0
RTOL = 1e-9


@pytest.fixture(scope="module")
def constant_try(tmp_path_factory):
    """Return a function writing a TRY file with one temperature in every hour."""
    with open(pyslpheat.TRY_BAUTZEN_2015, "r", encoding="latin-1", newline="") as fh:
        lines = fh.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("***")) + 1
    directory = tmp_path_factory.mktemp("try")

    def write(temperature):
        data = []
        for line in lines[start:]:
            parts = line.split()
            parts[5] = f"{temperature:.1f}"
            data.append(" ".join(parts) + "\n")
        path = directory / f"TRY_constant_{temperature:+.1f}.dat"
        with open(path, "w", encoding="latin-1", newline="") as fh:
            fh.writelines(lines[:start] + data)
        return str(path)

    return write


def _bdew(profile_type, try_path):
    return bdew.calculate(ANNUAL_kWh, profile_type, "03", try_path, 2023)


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
