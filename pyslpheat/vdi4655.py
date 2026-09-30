"""
VDI 4655 heat and electricity demand profile calculation module.

Implements the VDI 4655 day-type methodology for residential buildings
with quarter-hourly profiles based on day-type classification and
DWD TRY meteorological data.

References
----------
VDI 4655 (July 2021):
    *Referenzlastprofile von Wohngebäuden für Strom, Heizung und Trinkwarmwasser sowie Referenzerzeugungsprofile für Fotovoltaikanlagen* Verein Deutscher Ingenieure.
    Day-type profiles in ``data/vdi4655/load_profiles/``;
    daily energy factors in ``data/vdi4655/Faktoren.csv``.

DWD / BBSR (July 2017):
    Handbuch - Ortsgenaue Testreferenzjahre von Deutschland für mittlere, extreme und zukünftige Witterungsverhältnisse
    TRY weather files provide hourly temperature and cloud cover for
    day-type classification (season × weekday × cloud cover).

:author: Dipl.-Ing. (FH) Jonas Pfeiffer
"""

import logging
import os as _os

import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    from ._cache import FileCache, cached_data, read_only
    from ._weather import fit_to_year
except ImportError:  # executed as a script: python pyslpheat/vdi4655.py
    from _cache import FileCache, cached_data, read_only
    from _weather import fit_to_year

_log = logging.getLogger(__name__)


# ── Local helper implementations (no external package dependency) ─────────────

def _read_TRY(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a TRY weather file into read-only meteorological arrays.

    :return: (temperature, windspeed, direct_rad, global_rad, cloud_cover)
    """
    temps, winds, dirs, diffs, clouds = [], [], [], [], []
    past_header = False
    with open(filename, "r", encoding="latin-1") as fh:
        for line in fh:
            if not past_header:
                if line.strip().startswith("***"):
                    past_header = True
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                temps.append(float(parts[5]))   # t
                winds.append(float(parts[8]))   # WG
                clouds.append(float(parts[9]))  # N (oktas)
                dirs.append(float(parts[12]))   # B (direct)
                diffs.append(float(parts[13]))  # D (diffuse)
            except ValueError:
                continue
    temperature      = np.array(temps,  dtype=float)
    windspeed        = np.array(winds,  dtype=float)
    direct_radiation = np.array(dirs,   dtype=float)
    diffuse_radiation = np.array(diffs, dtype=float)
    global_radiation = direct_radiation + diffuse_radiation
    cloud_cover      = np.array(clouds, dtype=float)
    return tuple(read_only(a) for a in (
        temperature, windspeed, direct_radiation, global_radiation, cloud_cover))


_TRY_CACHE = FileCache(_read_TRY)


def import_TRY(filename: str):
    """
    Read TRY weather file and return meteorological arrays.

    Searches for the '***' separator line robustly (compatible with all DWD TRY
    formats). Columns follow the standard TRY definition:
      RW HW MM DD HH t p WR WG N x RF B D A E IL
      index: 0  1  2  3  4 5 6  7  8 9 ...    12 13

    :return: (temperature [°C], windspeed [m/s], direct_rad [W/m²],
              global_rad [W/m²], cloud_cover [oktas 0-8])

    .. note::
        The parsed file is cached per process and read again when its
        modification time or size changes. The returned arrays are copies
        and may be modified freely.
    """
    return tuple(a.copy() for a in _TRY_CACHE.get(filename))


_VDI4655_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "vdi4655")


def get_resource_path(relative_path: str) -> str:
    """
    Resolve a resource path to an absolute path inside data/vdi4655/.
    Accepts both forward and backward slashes and strips any legacy path prefixes.
    """
    norm = relative_path.replace("/", _os.sep).replace("\\", _os.sep)
    # Strip legacy prefixes from older directory layout
    for prefix in (
        "data" + _os.sep + "VDI 4655 profiles" + _os.sep,
        "VDI 4655 profiles" + _os.sep,
    ):
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
            break
    # Map old sub-folder names to new layout
    norm = norm.replace("VDI 4655 data" + _os.sep, "")
    norm = norm.replace("VDI 4655 load profiles" + _os.sep, "load_profiles" + _os.sep)
    return _os.path.join(_VDI4655_DATA_DIR, norm)

def generate_year_months_days_weekdays(year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate temporal arrays for VDI 4655 day-type classification.

    :param year: Target year
    :type year: int
    :return: Tuple of (days_of_year, months, days, weekdays) with ISO weekdays 1=Monday, 7=Sunday
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    
    .. note::
        Used for workday/weekend/holiday classification in VDI 4655.
    """
    start_date = np.datetime64(f'{year}-01-01')
    
    # Determine number of days (handle leap years)
    end_date = np.datetime64(f'{year}-12-31')
    num_days = (end_date - start_date).astype(int) + 1
    
    # Generate day-of-year array
    days_of_year = np.arange(start_date, start_date + np.timedelta64(num_days, 'D'), dtype='datetime64[D]')
    
    # Extract month numbers (1-12)
    months = days_of_year.astype('datetime64[M]').astype(int) % 12 + 1
    
    # Extract day numbers within month (1-31)
    month_start = days_of_year.astype('datetime64[M]')
    days = (days_of_year - month_start).astype(int) + 1
    
    # Calculate weekday numbers (1=Monday, 7=Sunday)
    # NumPy weekday: 0=Monday, 6=Sunday, convert to 1=Monday, 7=Sunday
    weekdays = ((days_of_year.astype('datetime64[D]').astype(int) + 4) % 7) + 1
    
    return days_of_year, months, days, weekdays

def calculate_daily_averages(temperature: np.ndarray, cloud_cover: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate daily averages from hourly meteorological data for VDI 4655 day-type classification.

    :param temperature: Hourly temperature [°C] (8760 or 8784 hours)
    :type temperature: np.ndarray
    :param cloud_cover: Hourly cloud cover [0-8 oktas] (8760 or 8784 hours)
    :type cloud_cover: np.ndarray
    :return: Tuple of (daily_avg_temperature, daily_avg_cloud_cover)
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If arrays incomplete or mismatched lengths
    :raises IndexError: If reshaping fails
    
    .. note::
        Seasons: W (<5°C), Ü (5-15°C), S (>15°C). Cloud: H (<4 oktas), B (≥4 oktas), X (summer).
    """
    num_hours = temperature.size
    num_days = num_hours // 24
    
    # Validate complete daily blocks
    if num_hours % 24 != 0:
        raise ValueError(f"Incomplete hourly data: {num_hours} hours not divisible by 24")
    
    if temperature.size != cloud_cover.size:
        raise ValueError("Temperature and cloud cover arrays must have same length")
    
    # Reshape to daily blocks and calculate averages
    daily_temperature = temperature[:num_days*24].reshape((num_days, 24))
    daily_cloud_cover = cloud_cover[:num_days*24].reshape((num_days, 24))
    
    daily_avg_temperature = np.mean(daily_temperature, axis=1)
    daily_avg_cloud_cover = np.mean(daily_cloud_cover, axis=1)
    
    return daily_avg_temperature, daily_avg_cloud_cover

def calculate_quarter_hourly_intervals(year: int) -> np.ndarray:
    """
    Generate quarter-hourly datetime intervals for VDI 4655 load profiles.

    :param year: Target year
    :type year: int
    :return: Quarter-hourly datetime64[15m] intervals (35,040 or 35,136 for leap year)
    :rtype: np.ndarray
    
    .. note::
        15-minute resolution matches VDI 4655 standard for district heating analysis.
    """
    start_date = np.datetime64(f'{year}-01-01')
    
    # Determine number of days (handle leap years)
    end_date = np.datetime64(f'{year}-12-31')
    num_days = (end_date - start_date).astype(int) + 1
    
    # Calculate total number of quarter-hourly intervals
    num_quarter_hours = num_days * 24 * 4
    
    # Generate quarter-hourly interval array
    intervals = np.arange(
        start_date, 
        start_date + np.timedelta64(num_quarter_hours, '15m'), 
        dtype='datetime64[15m]'
    )
    
    return intervals

def quarter_hourly_data(data: np.ndarray) -> np.ndarray:
    """
    Expand daily data to quarter-hourly resolution.

    :param data: Daily values to expand
    :type data: np.ndarray
    :return: Quarter-hourly array with each daily value replicated 96 times
    :rtype: np.ndarray
    
    .. note::
        Each daily value → 96 quarter-hourly intervals (24h × 4 quarters/h).
    """
    num_quarter_hours_per_day = 24 * 4  # 96 intervals per day
    return np.repeat(data, num_quarter_hours_per_day)

_INTERVALS_PER_DAY = 24 * 4

# Column names are fixed by the VDI 4655 CSV schema (German headers in source data)
_PROFILE_COLUMNS = ('Strombedarf normiert', 'Heizwärme normiert', 'Warmwasser normiert')
_PROFILE_TIMES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]

@cached_data
def _load_profile(profile_day: str) -> np.ndarray:
    """
    Normalised quarter-hourly profiles of one type day, read once per process.

    :param profile_day: Building type and type day, e.g. ``'EFHWWH'``
    :type profile_day: str
    :return: Read-only array of shape (3, 96): electricity, space heating and
        DHW for the intervals starting at 00:00, 00:15, … 23:45
    :rtype: np.ndarray
    :raises FileNotFoundError: If there is no profile file for this type day
    """
    file_path = get_resource_path(f'data\\VDI 4655 profiles\\VDI 4655 load profiles\\{profile_day}.csv')
    profile_data = pd.read_csv(file_path, sep=';')
    by_time = profile_data[profile_data['Zeit'].notna()].set_index('Zeit').reindex(_PROFILE_TIMES)
    values = by_time[list(_PROFILE_COLUMNS)].to_numpy(dtype=float).T
    # Handle any missing values (fill with average)
    return read_only(np.nan_to_num(values, nan=1.0))

def standardized_quarter_hourly_profile(year: int, 
                                      building_type: str, 
                                      days_of_year: np.ndarray, 
                                      type_days: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate standardized VDI 4655 quarter-hourly load profiles.

    :param year: Target year
    :type year: int
    :param building_type: VDI 4655 type (EFH, MFH)
    :type building_type: str
    :param days_of_year: Daily datetime64 array
    :type days_of_year: np.ndarray
    :param type_days: Day-type classifications (e.g., WWH, SWX)
    :type type_days: np.ndarray
    :return: Tuple of (intervals, electricity, heating, hot_water) normalized profiles [0-2]
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :raises FileNotFoundError: If VDI 4655 profile data missing
    :raises KeyError: If building/day-type combination unavailable
    :raises ValueError: If temporal arrays inconsistent
    
    .. note::
        Day-type format: {Season}{DayType}{Cloud} - W/Ü/S + W/S + H/B/X.
    """
    # Generate quarter-hourly time intervals
    quarter_hourly_intervals = calculate_quarter_hourly_intervals(year)

    # Map each day of the year to its type day; all 96 intervals of a day share it
    year_days = quarter_hourly_intervals[::_INTERVALS_PER_DAY].astype('datetime64[D]')
    indices = np.searchsorted(days_of_year, year_days)
    daily_type_days = type_days[indices % len(type_days)]

    # Load VDI 4655 profile data for all required day types (each file is read
    # once per process)
    all_type_days, day_index = np.unique(daily_type_days, return_inverse=True)
    profiles = np.empty((len(all_type_days), len(_PROFILE_COLUMNS), _INTERVALS_PER_DAY))

    for i, type_day in enumerate(all_type_days):
        try:
            profiles[i] = _load_profile(f"{building_type}{type_day}")
        except FileNotFoundError:
            _log.warning("Profile file not found: %s", f"{building_type}{type_day}.csv")
            # Use a flat dummy profile if file missing
            profiles[i] = 1.0

    if len(days_of_year) * _INTERVALS_PER_DAY != len(quarter_hourly_intervals):
        raise ValueError("All arrays must be of the same length")

    # Select the profile of each day and string the days together
    electricity_demand, heating_demand, hot_water_demand = (
        profiles[day_index, column].reshape(-1) for column in range(len(_PROFILE_COLUMNS)))

    return quarter_hourly_intervals, electricity_demand, heating_demand, hot_water_demand

@cached_data
def _daily_factors() -> Dict[str, Tuple[float, float, float]]:
    """
    VDI 4655 daily energy factors, read once per process.

    :return: Profile day (e.g. ``'EFH9WWH'``) → (Fheiz,TT, Fel,TT, FTWW,TT);
        shared, must not be modified
    :rtype: Dict[str, Tuple[float, float, float]]
    :raises FileNotFoundError: If the factor file is missing
    """
    factors_file = get_resource_path('data\\VDI 4655 profiles\\VDI 4655 data\\Faktoren.csv')

    try:
        factor_data = pd.read_csv(factors_file, sep=';')
    except FileNotFoundError:
        raise FileNotFoundError(f"VDI 4655 factor data not found: {factors_file}")

    factor_data = factor_data[factor_data['Profiltag'].notna()]
    factors: Dict[str, Tuple[float, float, float]] = {}
    for tag, heating, electricity, hot_water in zip(
            factor_data['Profiltag'], factor_data['Fheiz,TT'],
            factor_data['Fel,TT'], factor_data['FTWW,TT']):
        # If a profile day is listed more than once, the first row counts
        factors.setdefault(tag, (heating, electricity, hot_water))
    return factors

def calculation_load_profile(TRY: str,
                           building_type: str, 
                           number_people_household: int, 
                           annual_electricity_kWh: float, 
                           annual_heating_kWh: float, 
                           annual_dhw_kWh: float, 
                           holidays: np.ndarray, 
                           climate_zone: str = "9", 
                           year: int = 2019) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate comprehensive VDI 4655 load profiles.

    :param TRY: Path to Test Reference Year weather data with 8760 hourly
        values; for a leap year 8784, or 8760 and 29 February repeats 28 February
    :type TRY: str
    :param building_type: VDI 4655 type (EFH, MFH)
    :type building_type: str
    :param number_people_household: Number of occupants
    :type number_people_household: int
    :param annual_electricity_kWh: Annual electricity [kWh/a]
    :type annual_electricity_kWh: float
    :param annual_heating_kWh: Annual heating [kWh/a]
    :type annual_heating_kWh: float
    :param annual_dhw_kWh: Annual DHW [kWh/a]
    :type annual_dhw_kWh: float
    :param holidays: Holiday dates array
    :type holidays: np.ndarray
    :param climate_zone: German climate zone 1-15 (default "9")
    :type climate_zone: str
    :param year: Target year (default 2019)
    :type year: int
    :return: Tuple of (intervals, electricity, heating, dhw, temperature) in kWh per 15min
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :raises FileNotFoundError: If TRY or factor data missing
    :raises ValueError: If parameters invalid or the weather data does not fit the year
    :raises KeyError: If VDI 4655 data incomplete
    
    .. note::
        Implements complete VDI 4655 workflow with day-type classification and energy balance normalization.
    """
    # Load VDI 4655 scaling factors (file is read once per process)
    factors = _daily_factors()

    # Generate temporal arrays
    days_of_year, months, days, weekdays = generate_year_months_days_weekdays(year)
    
    # Import and process weather data
    temperature, _, _, _, degree_of_coverage = import_TRY(TRY)
    # Leap year with 365 days of weather data: 29 February repeats 28 February
    temperature, degree_of_coverage = fit_to_year(year, temperature, degree_of_coverage)
    daily_avg_temperature, daily_avg_degree_of_coverage = calculate_daily_averages(temperature, degree_of_coverage)
    
    # VDI 4655 day-type classification
    # Season codes (per VDI 4655 standard): W=winter (<5°C), Ü=transition (5–15°C), S=summer (>15°C)
    season = np.where(daily_avg_temperature < 5, "W",
                     np.where((daily_avg_temperature >= 5) & (daily_avg_temperature <= 15), "Ü", "S"))
    
    # Day type classification (workday vs weekend/holiday)
    day_type = np.where((weekdays == 7) | np.isin(days_of_year, holidays), "S", "W")  # Sunday=7 or holiday
    
    # Cloud cover classification (only for non-summer days)
    cloud_classification = np.where(season == "S", "X", 
                                  np.where((daily_avg_degree_of_coverage >= 0) & (daily_avg_degree_of_coverage < 4), "H", "B"))
    
    # Combine classifications into day-type codes
    type_day = np.char.add(np.char.add(season, day_type), cloud_classification)
    profile_day = np.char.add((building_type + climate_zone), type_day)

    # Extract scaling factors for each day: look up each distinct profile day
    # once (a year has at most ten) and expand to all days
    tags, day_index = np.unique(profile_day, return_inverse=True)
    missing = [tag for tag in tags if tag not in factors]
    if missing:
        for tag in profile_day[np.isin(profile_day, missing)]:
            _log.warning("No factors found for profile day %s — using defaults", tag)
    tag_factors = np.array(
        [factors.get(tag, (1.0, 0.0, 0.0)) for tag in tags], dtype=float).reshape(-1, 3)
    f_heating_tt = tag_factors[day_index, 0]
    f_el_tt = tag_factors[day_index, 1]
    f_hotwater_tt = tag_factors[day_index, 2]

    # Calculate daily energy consumption using VDI 4655 formulas; the standard's
    # 1/365 is the share of one day, i.e. 1/366 in a leap year
    days_per_year = len(days_of_year)
    daily_electricity = annual_electricity_kWh * ((1/days_per_year) + (number_people_household * f_el_tt))
    daily_heating = annual_heating_kWh * f_heating_tt
    daily_hot_water = annual_dhw_kWh * ((1/days_per_year) + (number_people_household * f_hotwater_tt))

    # Generate standardized quarter-hourly profiles
    quarter_hourly_intervals, electricity_profile, heating_profile, hot_water_profile = \
        standardized_quarter_hourly_profile(year, building_type, days_of_year, type_day)

    # Expand daily factors to quarter-hourly resolution
    quarter_hourly_daily_electricity = quarter_hourly_data(daily_electricity)
    quarter_hourly_daily_heating = quarter_hourly_data(daily_heating)
    quarter_hourly_daily_hot_water = quarter_hourly_data(daily_hot_water)

    # Apply daily factors to standardized profiles
    electricity_scaled = electricity_profile * quarter_hourly_daily_electricity
    heating_scaled = heating_profile * quarter_hourly_daily_heating
    hot_water_scaled = hot_water_profile * quarter_hourly_daily_hot_water

    # Energy balance correction to match annual targets
    electricity_corrected = electricity_scaled / np.sum(electricity_scaled) * annual_electricity_kWh
    heating_corrected = heating_scaled / np.sum(heating_scaled) * annual_heating_kWh
    hot_water_corrected = hot_water_scaled / np.sum(hot_water_scaled) * annual_dhw_kWh

    return quarter_hourly_intervals, electricity_corrected, heating_corrected, hot_water_corrected, temperature

def calculate(annual_heating_kWh: float,
             annual_dhw_kWh: float,
             annual_electricity_kWh: float,
             building_type: str,
             number_people_household: int,
             year: int,
             climate_zone: str,
             TRY: str,
             holidays: np.ndarray) -> pd.DataFrame:
    """
    Calculate VDI 4655 building energy demand profiles.

    :param annual_heating_kWh: Annual heating [kWh/a]
    :type annual_heating_kWh: float
    :param annual_dhw_kWh: Annual DHW [kWh/a]
    :type annual_dhw_kWh: float
    :param annual_electricity_kWh: Annual electricity [kWh/a]
    :type annual_electricity_kWh: float
    :param building_type: VDI 4655 type (EFH, MFH)
    :type building_type: str
    :param number_people_household: Number of occupants
    :type number_people_household: int
    :param year: Target year
    :type year: int
    :param climate_zone: German climate zone 1-15
    :type climate_zone: str
    :param TRY: Path to Test Reference Year weather data with 8760 hourly
        values; for a leap year 8784, or 8760 and 29 February repeats 28 February
    :type TRY: str
    :param holidays: Holiday dates array
    :type holidays: np.ndarray
    :return: DataFrame with 15-min DatetimeIndex and columns
        ``Q_heat_kWh``, ``Q_dhw_kWh``, ``Q_total_kWh``, ``Q_electricity_kWh``,
        ``temperature_C``. Values are energy per 15-min interval [kWh].
    :rtype: pd.DataFrame
    :raises ValueError: If parameters invalid or negative, or the weather data does
        not fit the year
    :raises FileNotFoundError: If TRY or VDI 4655 data missing
    :raises RuntimeError: If calculation fails
    """
    # Input validation
    if annual_heating_kWh <= 0 or annual_dhw_kWh <= 0 or annual_electricity_kWh <= 0:
        raise ValueError("Annual energy consumption values must be positive")
    
    if number_people_household <= 0:
        raise ValueError("Number of people in household must be positive")
    
    valid_building_types = ["EFH", "MFH"]
    if building_type not in valid_building_types:
        raise ValueError(f"Building type must be one of {valid_building_types}")
    
    # Execute VDI 4655 calculation
    time_15min, electricity_kWh_15min, heating_kWh_15min, hot_water_kWh_15min, temperature = \
        calculation_load_profile(
            TRY=TRY,
            building_type=building_type,
            number_people_household=number_people_household,
            annual_electricity_kWh=annual_electricity_kWh,
            annual_heating_kWh=annual_heating_kWh,
            annual_dhw_kWh=annual_dhw_kWh,
            holidays=holidays,
            climate_zone=climate_zone,
            year=year
        )
    
    # Calculate total heat demand
    total_heat_kWh_15min = heating_kWh_15min + hot_water_kWh_15min
    
    # Validation of energy balance
    tolerance = 0.01  # 1% tolerance
    if abs(heating_kWh_15min.sum() - annual_heating_kWh) / annual_heating_kWh > tolerance:
        _log.warning("Heating energy balance error: %.0f vs %.0f kWh",
                     heating_kWh_15min.sum(), annual_heating_kWh)
    if abs(hot_water_kWh_15min.sum() - annual_dhw_kWh) / annual_dhw_kWh > tolerance:
        _log.warning("DHW energy balance error: %.0f vs %.0f kWh",
                     hot_water_kWh_15min.sum(), annual_dhw_kWh)
    if abs(electricity_kWh_15min.sum() - annual_electricity_kWh) / annual_electricity_kWh > tolerance:
        _log.warning("Electricity energy balance error: %.0f vs %.0f kWh",
                     electricity_kWh_15min.sum(), annual_electricity_kWh)

    # numpy's datetime64[15m] is not a standard unit — build the index with
    # pandas to guarantee correct 15-minute spacing.
    n_intervals = len(time_15min)
    idx = pd.date_range(f"{year}-01-01", periods=n_intervals, freq="15min")
    # Expand hourly temperature to 15-min resolution for the output column
    temp_15min = np.repeat(temperature, 4)[:n_intervals]

    return pd.DataFrame({
        "Q_heat_kWh":        heating_kWh_15min,
        "Q_dhw_kWh":         hot_water_kWh_15min,
        "Q_total_kWh":       total_heat_kWh_15min,
        "Q_electricity_kWh": electricity_kWh_15min,
        "temperature_C":     temp_15min,
    }, index=idx)


if __name__ == "__main__":
    # Smoke test – verify energy balance is maintained.
    # For a full parameter comparison run:  python examples/vdi4655_demo.py
    try:
        from pyslpheat import TRY_BAUTZEN_2015
    except ImportError:
        TRY_BAUTZEN_2015 = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "data", "try", "TRY2015_511676144222_Jahr.dat"
        )
    _holidays = np.array(["2026-01-01", "2026-04-03", "2026-04-06",
                           "2026-05-01", "2026-05-14", "2026-05-25",
                           "2026-10-03", "2026-12-25", "2026-12-26"],
                          dtype="datetime64[D]")
    _df = calculate(annual_heating_kWh=15_000, annual_dhw_kWh=5_000,
                    annual_electricity_kWh=4_000, building_type="MFH",
                    number_people_household=4, year=2026, climate_zone="9",
                    TRY=TRY_BAUTZEN_2015, holidays=_holidays)
    _heat  = _df["Q_heat_kWh"].sum()
    _dhw   = _df["Q_dhw_kWh"].sum()
    assert abs(_heat  - 15_000) / 15_000 < 0.01, f"Heating balance error: {_heat:.0f}"
    assert abs(_dhw   -  5_000) /  5_000 < 0.01, f"DHW balance error: {_dhw:.0f}"
    print(f"vdi4655.py OK  MFH  heat={_heat:.0f} kWh  dhw={_dhw:.0f} kWh  "
          f"peak={_df['Q_total_kWh'].max():.2f} kWh/15min")