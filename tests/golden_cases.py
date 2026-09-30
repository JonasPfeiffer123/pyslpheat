"""
Case definitions shared by the golden-master generator and the tests.

Each case is one call of ``bdew.calculate`` or ``vdi4655.calculate``. The
recorded outputs live in ``tests/golden/`` and were captured from the
unoptimised reference implementation (see ``generate_golden.py``).
"""

import os
from typing import Any, Dict, Tuple

import numpy as np

import pyslpheat
from pyslpheat import bdew, vdi4655

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
GOLDEN_NPZ = os.path.join(GOLDEN_DIR, "golden_master.npz")
GOLDEN_META = os.path.join(GOLDEN_DIR, "golden_master.json")

# Key of the synthetic 8784-hour TRY file, see write_leap_try()
LEAP_TRY = "leap"

TRY_FILES = {
    "2015": pyslpheat.TRY_BAUTZEN_2015,
    "2045_winter": pyslpheat.TRY_BAUTZEN_2045_WINTER,
    "2045_summer": pyslpheat.TRY_BAUTZEN_2045_SUMMER,
}

# Nationwide German public holidays, written out so the cases do not depend on
# the code under test.
HOLIDAYS = {
    2021: ["2021-01-01", "2021-04-02", "2021-04-05", "2021-05-01", "2021-05-13",
           "2021-05-24", "2021-10-03", "2021-12-25", "2021-12-26"],
    2023: ["2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-18",
           "2023-05-29", "2023-10-03", "2023-12-25", "2023-12-26"],
    2024: ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09",
           "2024-05-20", "2024-10-03", "2024-12-25", "2024-12-26"],
}


def _bdew(try_file: str = "2015", **kwargs: Any) -> Dict[str, Any]:
    return {"module": "bdew", "try_file": try_file, "kwargs": kwargs}


def _vdi(try_file: str = "2015", holidays: bool = True, **kwargs: Any) -> Dict[str, Any]:
    return {"module": "vdi4655", "try_file": try_file, "holidays": holidays, "kwargs": kwargs}


CASES: Dict[str, Dict[str, Any]] = {
    # ── BDEW: profile types / subtypes, Mode A ───────────────────────────────
    "bdew_GBD03_2023": _bdew(
        annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2023),
    "bdew_HMF03_2023_dhw_share": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023, dhw_share=0.25),
    "bdew_GKO03_2023": _bdew(
        annual_heat_kWh=80_000.0, profile_type="GKO", subtype="03", year=2023),
    "bdew_HEF03_2021": _bdew(
        annual_heat_kWh=18_000.0, profile_type="HEF", subtype="03", year=2021),
    # SigLinDe subtypes with non-zero linear terms mH/bH/mW/bW
    "bdew_HEF33_2023": _bdew(
        annual_heat_kWh=18_000.0, profile_type="HEF", subtype="33", year=2023),
    "bdew_GKO34_2023_dhw_share": _bdew(
        annual_heat_kWh=80_000.0, profile_type="GKO", subtype="34", year=2023, dhw_share=0.1),
    # GMF: hourly_coefficients.csv holds three '5,18' entries (decimal comma)
    "bdew_GMF03_2023": _bdew(
        annual_heat_kWh=60_000.0, profile_type="GMF", subtype="03", year=2023),
    # ── BDEW: call exactly as issued by DistrictHeatingSim ───────────────────
    "bdew_GBD03_2023_dhs_call": _bdew(
        annual_heat_kWh=np.float64(50_000.0), profile_type="GBD", subtype="03", year=2023,
        dhw_share=0.15, heating_limit_temp=15.0, heating_exponent=1.0, peak_design_kW=40.0),
    "bdew_HMF03_2023_dhs_call_defaults": _bdew(
        annual_heat_kWh=np.float64(120_000.0), profile_type="HMF", subtype="03", year=2023,
        dhw_share=0.2, heating_limit_temp=None, heating_exponent=1.0, peak_design_kW=None),
    # ── BDEW: optional shaping parameters ────────────────────────────────────
    "bdew_GBD03_2023_heating_limit": _bdew(
        annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2023,
        heating_limit_temp=12.0),
    "bdew_HMF03_2023_heating_exponent": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023,
        heating_exponent=1.5),
    "bdew_HMF03_2023_exponent_limit_share": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023,
        heating_exponent=0.7, heating_limit_temp=15.0, dhw_share=0.3),
    "bdew_HMF03_2023_dhw_flat": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023, dhw_flat=True),
    "bdew_HMF03_2023_dhw_share_invalid": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023, dhw_share=1.5),
    "bdew_HMF03_2023_dhw_share_nan": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023,
        dhw_share=float("nan")),
    # ── BDEW: design-load scaling ────────────────────────────────────────────
    "bdew_GKO03_2023_mode_b": _bdew(
        annual_heat_kWh=None, profile_type="GKO", subtype="03", year=2023,
        peak_design_kW=45.0, design_temperature=-14.0),
    "bdew_GKO03_2023_mode_c": _bdew(
        annual_heat_kWh=80_000.0, profile_type="GKO", subtype="03", year=2023,
        peak_design_kW=40.0, design_temperature=-14.0),
    "bdew_HEF33_2023_mode_c_limit_share": _bdew(
        annual_heat_kWh=18_000.0, profile_type="HEF", subtype="33", year=2023,
        peak_design_kW=9.0, design_temperature=-12.4, heating_limit_temp=15.0, dhw_share=0.2),
    # ── BDEW: stochastic post-processing (fixed seeds) ───────────────────────
    "bdew_HMF03_2023_stochastic": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023,
        dhw_share=0.25, stochastic=True, stochastic_seed=42),
    "bdew_GBD03_2023_stochastic_custom": _bdew(
        annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2023,
        stochastic=True, stochastic_seed=7, stochastic_sigma_sh=0.2,
        stochastic_sigma_dhw=0.35, stochastic_max_shift_sh=2, stochastic_max_shift_dhw=3),
    "bdew_HMF03_2023_draw_events": _bdew(
        annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2023,
        dhw_share=0.25, dhw_draw_events=True),
    "bdew_HEF03_2023_stochastic_draw_events": _bdew(
        annual_heat_kWh=18_000.0, profile_type="HEF", subtype="03", year=2023,
        stochastic=True, stochastic_seed=3, dhw_draw_events=True,
        dhw_draws_per_day=6.5, dhw_draw_seed=11),
    # ── BDEW: other weather file, leap year ──────────────────────────────────
    "bdew_GBD03_2023_try2045_winter": _bdew(
        "2045_winter", annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2023),
    # 8760-hour TRY in a leap year: the reference implementation raises
    "bdew_GBD03_2024_try8760": _bdew(
        annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2024),
    "bdew_GBD03_2024_leap": _bdew(
        LEAP_TRY, annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03", year=2024,
        dhw_share=0.15, heating_limit_temp=15.0),
    "bdew_HMF03_2024_leap_stochastic_draw_events": _bdew(
        LEAP_TRY, annual_heat_kWh=120_000.0, profile_type="HMF", subtype="03", year=2024,
        stochastic=True, stochastic_seed=42, dhw_draw_events=True),
    # ── BDEW: error paths ────────────────────────────────────────────────────
    "bdew_unknown_profile": _bdew(
        annual_heat_kWh=50_000.0, profile_type="XXX", subtype="99", year=2023),
    "bdew_no_scaling_input": _bdew(
        annual_heat_kWh=None, profile_type="GBD", subtype="03", year=2023),

    # ── VDI 4655 ─────────────────────────────────────────────────────────────
    # annual_electricity_kWh=1 is the placeholder DistrictHeatingSim passes
    "vdi_EFH_2023_holidays": _vdi(
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=1,
        building_type="EFH", number_people_household=2, year=2023, climate_zone="9"),
    "vdi_MFH_2023_holidays": _vdi(
        annual_heating_kWh=60_000.0, annual_dhw_kWh=12_000.0, annual_electricity_kWh=1,
        building_type="MFH", number_people_household=2, year=2023, climate_zone="9"),
    "vdi_EFH_2023_no_holidays": _vdi(
        holidays=False,
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=1,
        building_type="EFH", number_people_household=2, year=2023, climate_zone="9"),
    "vdi_MFH_2021_holidays": _vdi(
        annual_heating_kWh=60_000.0, annual_dhw_kWh=12_000.0, annual_electricity_kWh=9_000.0,
        building_type="MFH", number_people_household=4, year=2021, climate_zone="9"),
    "vdi_EFH_2023_zone3": _vdi(
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=3_500.0,
        building_type="EFH", number_people_household=3, year=2023, climate_zone="3"),
    "vdi_MFH_2023_try2045_summer": _vdi(
        "2045_summer",
        annual_heating_kWh=60_000.0, annual_dhw_kWh=12_000.0, annual_electricity_kWh=1,
        building_type="MFH", number_people_household=2, year=2023, climate_zone="9"),
    # Climate zone without factors: reference implementation falls back to defaults
    "vdi_EFH_2023_unknown_zone": _vdi(
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=3_500.0,
        building_type="EFH", number_people_household=2, year=2023, climate_zone="99"),
    "vdi_EFH_2024_try8760": _vdi(
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=1,
        building_type="EFH", number_people_household=2, year=2024, climate_zone="9"),
    "vdi_EFH_2024_leap_holidays": _vdi(
        LEAP_TRY,
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=1,
        building_type="EFH", number_people_household=2, year=2024, climate_zone="9"),
    "vdi_MFH_2024_leap_no_holidays": _vdi(
        LEAP_TRY, holidays=False,
        annual_heating_kWh=60_000.0, annual_dhw_kWh=12_000.0, annual_electricity_kWh=9_000.0,
        building_type="MFH", number_people_household=4, year=2024, climate_zone="9"),
    "vdi_invalid_building_type": _vdi(
        annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0, annual_electricity_kWh=1,
        building_type="B", number_people_household=2, year=2023, climate_zone="9"),
}


def write_leap_try(directory: str) -> str:
    """
    Write a synthetic 8784-hour TRY file and return its path.

    The bundled TRY files cover 8760 hours. For leap-year cases the 24 lines of
    28 February are repeated once, so the file has 366 complete days. Only the
    number and order of data lines matters; the date columns are not parsed.

    :param directory: Existing directory to write into
    :type directory: str
    :return: Path of the written file
    :rtype: str
    """
    with open(TRY_FILES["2015"], "r", encoding="latin-1", newline="") as fh:
        lines = fh.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("***")) + 1
    feb28 = start + (31 + 27) * 24
    leap_lines = lines[:feb28 + 24] + lines[feb28:feb28 + 24] + lines[feb28 + 24:]
    path = os.path.join(directory, "TRY2015_leap_8784h.dat")
    with open(path, "w", encoding="latin-1", newline="") as fh:
        fh.writelines(leap_lines)
    return path


def run_case(name: str, leap_try_path: str) -> Tuple[str, Any]:
    """
    Execute one case.

    :param name: Key into :data:`CASES`
    :type name: str
    :param leap_try_path: Path returned by :func:`write_leap_try`
    :type leap_try_path: str
    :return: ``("ok", DataFrame)`` or ``("error", exception)``
    :rtype: Tuple[str, Any]
    """
    case = CASES[name]
    try_path = leap_try_path if case["try_file"] == LEAP_TRY else TRY_FILES[case["try_file"]]
    kwargs = dict(case["kwargs"])
    try:
        if case["module"] == "bdew":
            return "ok", bdew.calculate(TRY_file_path=try_path, **kwargs)
        holidays = np.array(HOLIDAYS[kwargs["year"]] if case["holidays"] else [],
                            dtype="datetime64[D]")
        return "ok", vdi4655.calculate(TRY=try_path, holidays=holidays, **kwargs)
    except Exception as exc:  # noqa: BLE001 - error type and message are part of the golden master
        return "error", exc


def expected_index(spec: Dict[str, Any]) -> np.ndarray:
    """
    Rebuild a recorded DatetimeIndex from its start/step/periods description.

    :param spec: ``index`` entry of a case in the golden-master metadata
    :type spec: Dict[str, Any]
    :return: datetime64 array in the recorded dtype
    :rtype: np.ndarray
    """
    steps = np.arange(spec["periods"]) * np.timedelta64(spec["step_seconds"], "s")
    return (np.datetime64(spec["start"], "s") + steps).astype(spec["dtype"])
