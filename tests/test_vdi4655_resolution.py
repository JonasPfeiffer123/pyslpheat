"""
VDI 4655 results in hourly resolution (``resolution="1h"``).

The default stays quarter-hourly as defined by VDI 4655; the golden-master
tests pin it bit for bit. With ``"1h"`` the four quarter-hours of each hour are
added up, so the result has the same time steps as ``bdew.calculate()`` and
both can be stacked without resampling (issue #2).
"""

import numpy as np
import pandas as pd
import pytest

import pyslpheat
from pyslpheat import bdew, vdi4655
from pyslpheat.bdew import compute_holidays

TRY = pyslpheat.TRY_BAUTZEN_2015
ENERGY_COLUMNS = ["Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh", "Q_electricity_kWh"]
RTOL = 1e-12


def _vdi(year=2023, building_type="EFH", **kwargs):
    holidays = np.array(sorted(compute_holidays(year)), dtype="datetime64[D]")
    return vdi4655.calculate(
        annual_heating_kWh=12_000, annual_dhw_kWh=2_000, annual_electricity_kWh=3_500,
        building_type=building_type, number_people_household=3, year=year,
        climate_zone="9", TRY=TRY, holidays=holidays, **kwargs,
    )


def test_default_is_quarter_hourly():
    default = _vdi()
    pd.testing.assert_frame_equal(default, _vdi(resolution="15min"), check_exact=True)
    assert len(default) == 35_040
    assert default.index.freq == pd.offsets.Minute(15)


@pytest.mark.parametrize("building_type", ["EFH", "MFH"])
@pytest.mark.parametrize("year, hours", [(2023, 8760), (2024, 8784)])
def test_hourly_adds_up_quarter_hours(year, hours, building_type):
    quarter = _vdi(year, building_type)
    hourly = _vdi(year, building_type, resolution="1h")

    assert len(hourly) == hours == len(quarter) // 4
    assert list(hourly.columns) == list(quarter.columns)
    assert hourly.index[0] == pd.Timestamp(f"{year}-01-01")
    assert hourly.index.freq == pd.offsets.Hour()
    assert np.array_equal(hourly.index.values, quarter.index.values[::4])

    for column in ENERGY_COLUMNS:
        expected = quarter[column].values.reshape(-1, 4).sum(axis=1)
        np.testing.assert_allclose(hourly[column].values, expected, rtol=RTOL, atol=0.0,
                                   err_msg=column)
        assert hourly[column].sum() == pytest.approx(quarter[column].sum(), rel=RTOL)

    # As in the quarter-hourly result, the total is exactly heating plus DHW
    assert np.array_equal(hourly["Q_total_kWh"].values,
                          (hourly["Q_heat_kWh"] + hourly["Q_dhw_kWh"]).values)
    # The weather file is hourly, so these are its temperatures
    assert np.array_equal(hourly["temperature_C"].values, quarter["temperature_C"].values[::4])


@pytest.mark.parametrize("year", [2023, 2024])
def test_hourly_matches_bdew_time_steps(year):
    vdi = _vdi(year, resolution="1h")
    bdew_result = bdew.calculate(annual_heat_kWh=20_000, profile_type="HMF", subtype="03",
                                 TRY_file_path=TRY, year=year)

    assert np.array_equal(vdi.index.values, bdew_result.index.values)
    assert np.array_equal(vdi["temperature_C"].values, bdew_result["temperature_C"].values)
    stacked = np.vstack([vdi["Q_total_kWh"].values, bdew_result["Q_total_kWh"].values])
    assert stacked.shape == (2, len(bdew_result))


@pytest.mark.parametrize("resolution", ["h", "H", "60min", "15T", "hourly", "", None])
def test_unknown_resolution_raises(resolution):
    with pytest.raises(ValueError, match="Resolution must be one of"):
        _vdi(resolution=resolution)
