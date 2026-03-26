"""
BDEW Standard Load Profile heat demand calculation module.

Implements the BDEW SigLinDe methodology for residential and commercial/public
buildings with temperature-dependent sigmoid + linear daily heat demand and
weekday-dependent hourly shape factors.

References
----------
BDEW/VKU/GEODE (2025):
    *Leitfaden — Abwicklung von Standardlastprofilen Gas*, Berlin, 28.10.2025.
    Coefficient tables extracted into ``data/bdew/daily_coefficients.csv``.

FfE München (Juli 2015):
    *Weiterentwicklung des Standardlastprofilverfahrens Gas*
    Underlying method for subtypes 33/34.

BGW/DVGW (August 2006):
    *Anwendung von Standardlastprofilen zur Belieferung nichtleistungsgemessener
    Kunden.* 
    Hourly shape factors in ``data/bdew/hourly_coefficients.csv``.

:author: Dipl.-Ing. (FH) Jonas Pfeiffer
"""

import logging
import pandas as pd
import numpy as np
import math
import os
from typing import Optional, Tuple

_log = logging.getLogger(__name__)
from datetime import date as _date, timedelta as _timedelta

# Data directory for BDEW CSV files
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bdew")


def import_TRY(filepath: str) -> Tuple[np.ndarray, None, None, None, None]:
    """
    Read hourly temperatures from a DWD TRY .dat file.

    The fixed-width header ends with a line starting with '***'.
    Data columns (whitespace-separated): RW HW MM DD HH t p WR WG N x RF B D A E IL
    Column index 5 is the air temperature t [°C].

    :param filepath: Path to TRY .dat file
    :type filepath: str
    :return: (hourly_temperature, None, None, None, None)
    :rtype: Tuple[np.ndarray, None, None, None, None]
    """
    temperatures: list = []
    past_header = False
    with open(filepath, 'r', encoding='latin-1') as fh:
        for line in fh:
            if not past_header:
                if line.strip().startswith('***'):
                    past_header = True
                continue
            parts = line.split()
            if len(parts) >= 6:
                temperatures.append(float(parts[5]))
    return np.array(temperatures, dtype=float), None, None, None, None

def generate_year_months_days_weekdays(year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate temporal arrays for BDEW calculations.

    :param year: Target year
    :type year: int
    :return: Tuple of (days_of_year, months, days, weekdays) with weekday 1=Monday, 7=Sunday
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    
    .. note::
        Handles leap years automatically. Weekdays use ISO numbering.
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
    weekdays = ((days_of_year.astype('datetime64[D]').astype(int) + 4) % 7) + 1
    
    return days_of_year, months, days, weekdays

def calculate_daily_averages(temperature: np.ndarray) -> np.ndarray:
    """
    Calculate daily average temperatures from hourly data.

    :param temperature: Hourly temperature [°C] for complete year (8760/8784 hours)
    :type temperature: np.ndarray
    :return: Daily average temperatures [°C]
    :rtype: np.ndarray
    :raises ValueError: If temperature array not divisible by 24
    
    .. note::
        Used for BDEW sigmoid function temperature dependencies.
    """
    num_hours = temperature.size
    num_days = num_hours // 24
    
    # Validate complete daily blocks
    if num_hours % 24 != 0:
        raise ValueError(f"Temperature data incomplete: {num_hours} hours not divisible by 24")
    
    # Reshape to daily blocks and calculate averages
    daily_temperature = temperature[:num_days*24].reshape((num_days, 24))
    daily_avg_temperature = np.mean(daily_temperature, axis=1)
    
    return daily_avg_temperature

def calculate_allocation_temperature(daily_avg_temperature: np.ndarray) -> np.ndarray:
    """
    Calculate the BDEW allocation temperature (weighted moving average of daily temperatures).

    The allocation temperature smooths out short-term fluctuations and is used as the
    input to the SigLinDe sigmoid function.

    :param daily_avg_temperature: Daily mean temperatures [°C]
    :type daily_avg_temperature: np.ndarray
    :return: Allocation temperature for each day [°C]
    :rtype: np.ndarray

    .. note::
        Formula (BDEW guideline p. 43-44):
        T_allo(D) = (T_D·8 + T_{D-1}·4 + T_{D-2}·2 + T_{D-3}·1) / 15
    """
    weights = np.array([8.0, 4.0, 2.0, 1.0]) / 15.0
    n = len(daily_avg_temperature)
    result = np.empty(n)
    for i in range(n):
        result[i] = (
            weights[0] * daily_avg_temperature[i]
            + weights[1] * daily_avg_temperature[max(i - 1, 0)]
            + weights[2] * daily_avg_temperature[max(i - 2, 0)]
            + weights[3] * daily_avg_temperature[max(i - 3, 0)]
        )
    return result

def compute_holidays(year: int) -> set:
    """
    Compute statutory German public holidays per BDEW guideline §6.1.1.

    :param year: Target year
    :type year: int
    :return: Set of holiday dates
    :rtype: set

    .. note::
        Holidays are treated as Sundays (weekday 7) in the hourly-factor lookup.
        Nationwide holidays: New Year's Day, Good Friday, Easter Monday, Labour Day,
        Ascension Day, Whit Monday, German Unity Day, Christmas Day (×2).
    """
    a = year % 19
    b = year % 4
    c = year % 7
    k = year // 100
    p = (13 + 8 * k) // 25
    q = k // 4
    M = (15 - p + k - q) % 30
    N = (4 + k - q) % 7
    d = (19 * a + M) % 30
    e = (2 * b + 4 * c + 6 * d + N) % 7
    if d == 29 and e == 6:
        easter_offset = 50
    elif d == 28 and e == 6 and (11 * M + 11) % 30 < 19:
        easter_offset = 49
    else:
        easter_offset = d + e + 22
    easter = _date(year, 3, 1) + _timedelta(days=easter_offset - 1)
    return {
        _date(year, 1, 1),                       # New Year's Day
        easter + _timedelta(days=-2),             # Good Friday
        easter + _timedelta(days=1),              # Easter Monday
        _date(year, 5, 1),                        # Labour Day
        easter + _timedelta(days=39),             # Ascension Day
        easter + _timedelta(days=50),             # Whit Monday
        _date(year, 10, 3),                       # German Unity Day
        _date(year, 12, 25),                      # Christmas Day
        _date(year, 12, 26),                      # Boxing Day
    }

def calculate_hourly_intervals(year: int) -> np.ndarray:
    """
    Generate hourly datetime intervals for full year.

    :param year: Target year
    :type year: int
    :return: Hourly datetime64 intervals (8760 or 8784 for leap year)
    :rtype: np.ndarray
    """
    start_date = np.datetime64(f'{year}-01-01', 'h')
    
    # Determine number of days (handle leap years) - use datetime64[D] for both dates
    end_date_day = np.datetime64(f'{year}-12-31', 'D')
    start_date_day = np.datetime64(f'{year}-01-01', 'D')
    num_days = (end_date_day - start_date_day).astype(int) + 1
    
    # Calculate total number of hourly intervals
    num_hours = num_days * 24
    
    # Generate hourly interval array
    intervals = np.arange(
        start_date, 
        start_date + np.timedelta64(num_hours, 'h'), 
        dtype='datetime64[h]'
    )
    
    return intervals

def get_coefficients(profile_type: str, 
                    subtype: str, 
                    daily_data: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Extract BDEW profile coefficients for load calculation.

    :param profile_type: BDEW building type (GKO, GHA, GMK, GBD, GBH, GWA, GGA, GBA, GGB, GPD, GMF, GHD)
    :type profile_type: str
    :param subtype: Building subtype for detailed classification
    :type subtype: str
    :param daily_data: BDEW daily coefficients DataFrame
    :type daily_data: pd.DataFrame
    :return: Tuple of (A, B, C, D, mH, bH, mW, bW) sigmoid and linear coefficients
    :rtype: Tuple[float, float, float, float, float, float, float, float]
    :raises ValueError: If profile not found in data
    :raises KeyError: If coefficient columns missing
    
    .. note::
        Sigmoid: h_T = A/(1+(B/(T-40))^C) + mH*T + bH. DHW: mW*T + bW + D
    """
    # Combine profile type and subtype for lookup
    profile = profile_type + subtype
    
    # Find matching profile row in coefficient data
    profile_row = daily_data[daily_data['Standardlastprofil'] == profile]
    
    if profile_row.empty:
        raise ValueError(f"Profile '{profile}' not found in BDEW coefficient data")
    
    # Extract coefficients from first matching row
    row = profile_row.iloc[0]
    
    try:
        A = float(row['A'])
        B = float(row['B']) 
        C = float(row['C'])
        D = float(row['D'])
        mH = float(row['mH'])
        bH = float(row['bH'])
        mW = float(row['mW'])
        bW = float(row['bW'])
    except KeyError as e:
        raise KeyError(f"Missing coefficient column in BDEW data: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid coefficient value in BDEW data: {e}") from e
    
    return A, B, C, D, mH, bH, mW, bW

def get_weekday_factor(daily_weekdays: np.ndarray, 
                      profile_type: str, 
                      subtype: str, 
                      daily_data: pd.DataFrame) -> np.ndarray:
    """
    Extract weekday-specific load factors from BDEW data.

    :param daily_weekdays: Weekday numbers (1=Monday to 7=Sunday) for each day
    :type daily_weekdays: np.ndarray
    :param profile_type: BDEW building type
    :type profile_type: str
    :param subtype: Building subtype
    :type subtype: str
    :param daily_data: BDEW coefficients DataFrame with weekday columns '1'-'7'
    :type daily_data: pd.DataFrame
    :return: Weekday factors for each day (typically 0.5-1.5)
    :rtype: np.ndarray
    :raises ValueError: If profile not found
    :raises KeyError: If weekday columns missing
    
    .. note::
        Accounts for different operation patterns: offices high Mon-Fri, schools minimal weekends.
    """
    # Combine profile type and subtype for lookup
    profile = profile_type + subtype
    
    # Find matching profile row in coefficient data
    profile_row = daily_data[daily_data['Standardlastprofil'] == profile]
    
    if profile_row.empty:
        raise ValueError(f"Profile '{profile}' not found in BDEW coefficient data")
    
    # Extract weekday factors for each day
    try:
        weekday_factors = np.array([
            profile_row.iloc[0][str(day)] for day in daily_weekdays
        ]).astype(float)
    except KeyError as e:
        raise KeyError(f"Missing weekday column in BDEW data: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid weekday factor value: {e}") from e
    
    return weekday_factors

def _apply_peak_jitter(series: pd.Series, max_shift: int, rng: np.random.Generator) -> pd.Series:
    """
    Shift the hourly profile per day by ±max_shift hours (circular, within day).
    Daily sum is preserved.
    """
    result = series.copy()
    for day in series.index.normalize().unique():
        mask = series.index.normalize() == day
        vals = series[mask].values
        if len(vals) < 24:
            continue
        shift = int(rng.integers(-max_shift, max_shift + 1))
        result[mask] = np.roll(vals, shift)
    return result


def _apply_lognormal_noise(series: pd.Series, sigma: float, rng: np.random.Generator) -> pd.Series:
    """
    Multiply each hour by a log-normal factor (μ=0 in log-space), then
    renormalise to the original annual sum so the energy balance is exact.
    Zero values (e.g. summer nights without heating) are preserved as zero.
    """
    original_sum = series.sum()
    noise = rng.lognormal(mean=0.0, sigma=sigma, size=len(series))
    noisy = np.where(series.values <= 0.0, 0.0, series.values * noise)
    noisy_sum = noisy.sum()
    if noisy_sum > 0:
        noisy = noisy * original_sum / noisy_sum
    return pd.Series(noisy, index=series.index)


def _apply_dhw_draw_events(
    dhw: pd.Series,
    draws_per_day: float,
    seed: int,
) -> pd.Series:
    """
    Replace the smooth BDEW DHW timeseries with clustered stochastic draw events.

    Draw model:

    - Draws per day  ~ Poisson(draws_per_day)
    - Draw start hour ~ bimodal: 60 % morning (05–09 h), 40 % evening (17–22 h)
    - Draw duration  ~ Uniform(1, 3 h) contiguous block, same amplitude per block
    - Block amplitude ~ log-normal(mu=0, sigma=0.4), cap 2.0
    - Hours between draws stay at zero → realistic zero-load gaps

    Total DHW energy is preserved by normalisation.

    :param dhw: Hourly DHW series with DatetimeIndex
    :param draws_per_day: Expected number of draw events per day (Poisson rate)
    :param seed: RNG seed for reproducibility
    :return: Replaced DHW series with identical annual sum
    """
    rng = np.random.default_rng(seed)
    original_dhw = dhw.sum()

    new_dhw = np.zeros(len(dhw))
    days = pd.Series(dhw.index.date).unique()
    hour_index = {ts: i for i, ts in enumerate(dhw.index)}

    for day in days:
        n_draws = rng.poisson(draws_per_day)
        for _ in range(n_draws):
            if rng.random() < 0.60:
                start_h = int(rng.uniform(5, 9))    # morning window
            else:
                start_h = int(rng.uniform(17, 22))  # evening window

            duration = int(rng.uniform(1, 4))            # 1–3 h contiguous block
            amp = min(rng.lognormal(0.0, 0.4), 2.0)     # same amplitude across block

            for dh in range(duration):
                h  = (start_h + dh) % 24
                ts = pd.Timestamp(year=day.year, month=day.month,
                                  day=day.day, hour=h)
                if ts in hour_index:
                    new_dhw[hour_index[ts]] += amp

    total = new_dhw.sum()
    if total > 0:
        new_dhw *= original_dhw / total

    return pd.Series(new_dhw, index=dhw.index, dtype=float)


def calculate(annual_heat_kWh: Optional[float],
             profile_type: str,
             subtype: str,
             TRY_file_path: str,
             year: int,
             dhw_share: Optional[float] = None,
             heating_limit_temp: Optional[float] = None,
             heating_exponent: float = 1.0,
             dhw_flat: bool = False,
             peak_design_kW: Optional[float] = None,
             design_temperature: Optional[float] = None,
             stochastic: bool = False,
             stochastic_seed: Optional[int] = 42,
             stochastic_sigma_sh: float = 0.12,
             stochastic_sigma_dhw: float = 0.20,
             stochastic_max_shift_sh: int = 1,
             stochastic_max_shift_dhw: int = 2,
             dhw_draw_events: bool = False,
             dhw_draws_per_day: float = 4.0,
             dhw_draw_seed: int = 42) -> pd.DataFrame:
    """
    Calculate heat demand profiles using BDEW Standard Load Profile methodology.

    Scaling can be specified in three ways:

    1. ``annual_heat_kWh`` only – annual heat demand [kWh/a], original BDEW approach.
       Profile shape is determined purely by the sigmoid coefficients.

    2. ``peak_design_kW`` + ``design_temperature`` only – design heating power
       [kW] at design outside temperature [°C].  KW is set so that the daily-mean
       heating power at *design_temperature* equals *peak_design_kW*.
       Annual energy is a *derived* output.

    3. **Both** ``annual_heat_kWh`` AND ``peak_design_kW`` + ``design_temperature`` –
       both constraints are satisfied simultaneously.  The BDEW heating shape
       h_T_heating(T) is linearly rescaled by a slope factor α so that:

         KW · α · h_T_heating(T_design) = peak_design_kW  (daily mean)
         KW · [α · Σh_heat·F_D + Σh_dhw·F_D] = annual_heat_kWh

       Solving yields unique KW and α.  A steeper building (low annual demand, high peak)
       gives α > 1 (higher temperature sensitivity); a flat building gives α < 1.
       The temperature distribution over the year changes; the annual total and
       design-day peak are both met exactly.

    :param annual_heat_kWh: Annual heat demand [kWh/a], or None when using design-load
        scaling only
    :type annual_heat_kWh: Optional[float]
    :param profile_type: BDEW building type (GKO, GHA, GMK, etc.)
    :type profile_type: str
    :param subtype: Building subtype
    :type subtype: str
    :param TRY_file_path: Path to Test Reference Year weather data
    :type TRY_file_path: str
    :param year: Calculation year
    :type year: int
    :param dhw_share: Optional DHW share override (0-1)
    :type dhw_share: Optional[float]
    :param heating_limit_temp: Heating cut-off temperature [°C] – space heating is set
        to zero on days where the allocation temperature exceeds this threshold; DHW
        continues unaffected.
    :type heating_limit_temp: Optional[float]
    :param heating_exponent: Power-law exponent applied to the daily heating
        distribution after all other shaping (>1 → sharper peaks, <1 → flatter);
        annual heating sum is preserved
    :type heating_exponent: float
    :param dhw_flat: If True, DHW is distributed uniformly (1/24) within each day
    :type dhw_flat: bool
    :param peak_design_kW: Design heating power [kW] at *design_temperature*
        (space heating only, excluding DHW).  Requires *design_temperature*.
    :type peak_design_kW: Optional[float]
    :param design_temperature: Design outside temperature [°C], e.g. -12.4
    :type design_temperature: Optional[float]
    :param stochastic: Apply stochastic post-processing (peak jitter + log-normal noise).
        Applied before returning; energy balance is preserved by renormalisation.
    :type stochastic: bool
    :param stochastic_seed: Random seed for reproducibility (None = random).
    :type stochastic_seed: Optional[int]
    :param stochastic_sigma_sh: Log-normal σ for space heating (~12% default).
    :type stochastic_sigma_sh: float
    :param stochastic_sigma_dhw: Log-normal σ for DHW (~20% default).
    :type stochastic_sigma_dhw: float
    :param stochastic_max_shift_sh: Max daily peak shift [h] for space heating.
    :type stochastic_max_shift_sh: int
    :param stochastic_max_shift_dhw: Max daily peak shift [h] for DHW.
    :type stochastic_max_shift_dhw: int
    :param dhw_draw_events: Replace the smooth BDEW DHW timeseries with stochastic
        discrete draw events (bimodal morning/evening peaks, Poisson draw count per day).
        Annual DHW energy is preserved by renormalisation.  Moves further away from the
        BDEW average profile towards realistic individual-building behaviour.
    :type dhw_draw_events: bool
    :param dhw_draws_per_day: Expected number of draw events per day (Poisson rate).
        Typical range: 3–8 for residential buildings.
    :type dhw_draws_per_day: float
    :param dhw_draw_seed: RNG seed for DHW draw events (independent of stochastic_seed).
    :type dhw_draw_seed: int
    :return: DataFrame with DatetimeIndex and columns
        ``Q_heat_kWh``, ``Q_dhw_kWh``, ``Q_total_kWh``, ``temperature_C``.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If TRY or BDEW data files not found
    :raises ValueError: If profile not found or invalid parameters
    """
    # Input validation
    have_design = peak_design_kW is not None and design_temperature is not None
    have_jwb    = annual_heat_kWh is not None and annual_heat_kWh > 0
    if have_design and peak_design_kW <= 0:
        raise ValueError("peak_design_kW must be positive")
    if not have_design and not have_jwb:
        raise ValueError("Provide annual_heat_kWh > 0, or peak_design_kW + design_temperature, or both")

    if not isinstance(year, int) or year < 1900 or year > 2100:
        raise ValueError("Year must be valid integer between 1900 and 2100")
    
    # Generate temporal arrays for calculation
    days_of_year, months, days, daily_weekdays = generate_year_months_days_weekdays(year)

    # Import and process meteorological data
    hourly_temperature, _, _, _, _ = import_TRY(TRY_file_path)
    daily_avg_temperature = np.round(calculate_daily_averages(hourly_temperature), 1)

    # Allocation temperature (BDEW guideline p. 43-44):
    # T_allo(D) = (T_D*8 + T_{D-1}*4 + T_{D-2}*2 + T_{D-3}*1) / 15
    daily_alloc_temp = calculate_allocation_temperature(daily_avg_temperature)

    # Override weekday to 7 (Sunday) for statutory holidays (BDEW guideline §6.1.1)
    holiday_dates = compute_holidays(year)
    for i, d in enumerate(days_of_year):
        if _date.fromisoformat(str(d)) in holiday_dates:
            daily_weekdays[i] = 7

    # Load BDEW coefficient data
    daily_data = pd.read_csv(os.path.join(_HERE, 'daily_coefficients.csv'), delimiter=';')
    
    # Extract building-specific coefficients
    h_A, h_B, h_C, h_D, mH, bH, mW, bW = get_coefficients(profile_type, subtype, daily_data)
    
    # Linear temperature corrections based on allocation temperature (BDEW guideline p. 41-42)
    lin_H = (np.nan_to_num(mH * daily_alloc_temp + bH)
             if mH != 0 or bH != 0 else np.zeros(len(daily_alloc_temp)))
    lin_W = (np.nan_to_num(mW * daily_alloc_temp + bW)
             if mW != 0 or bW != 0 else np.zeros(len(daily_alloc_temp)))

    # SigLinDe: h_total = sigmoid + D + max(mH*T+bH, mW*T+bW)
    # DHW component:     h_dhw  = D + mW*T + bW  (warm water baseline)
    # Heating component: sigmoid + max(lin_H, lin_W) - lin_W
    h_T_sigmoid = h_A / (1 + (h_B / (daily_alloc_temp - 40)) ** h_C)
    h_T_dhw     = h_D + lin_W
    h_T_heating = h_T_sigmoid + np.maximum(lin_H, lin_W) - lin_W

    # Heating cut-off: no space heating above this threshold (DHW still flows)
    if heating_limit_temp is not None:
        h_T_heating = np.where(daily_alloc_temp > heating_limit_temp, 0.0, h_T_heating)

    h_T_total = h_T_heating + h_T_dhw

    # Apply weekday factors
    F_D = get_weekday_factor(daily_weekdays, profile_type, subtype, daily_data)

    # ── Normalization: KW and optional heating slope factor α ────────────────
    #
    # h_T_heating is the raw BDEW shape. After multiplying by α (slope factor)
    # and KW the daily heating energy = KW · α · h_T_heating(T) · F_D.
    #
    # Three modes:
    #
    # A  AHD only:      α = 1,  KW = AHD / Σ(h_T_total · F_D)
    # ── Normalization: KW and optional profile reshaping ─────────────────────
    #
    # The daily heating demand is KW · h_T_heating(T) · F_D [kWh/day].
    # The shape vector h_T_heating is fixed by BDEW coefficients (Mode A/B).
    # In Mode C both annual energy AND design-day peak are prescribed.  A
    # *linear* rescaling of h (h → α·h) cannot change their ratio because
    # KW absorbs α exactly.  Instead, a power-law h → h^β genuinely alters the
    # temperature sensitivity (β>1 → sharper winter peaks, β<1 → flatter).
    #
    # Mode A  annual energy only: β=1, KW = AHD / Σ(h_total·F_D)
    # Mode B  design only:       β=1, KW = peak·24 / h_des  → AHD derived
    # Mode C  both given:        find β such that
    #           KW(β) · h_des^β  = peak·24       (design-day condition)
    #           KW(β) · (Σh^β·F_D + S_dhw) = AHD (annual condition)
    #         with KW(β) = AHD / (Σh^β·F_D + S_dhw).
    #         Since h_des = max(h_T_heating), the ratio h_des^β / Σh^β·F_D is
    #         strictly monotone increasing in β  → unique root via bisection.

    sum_h_dhw_F = np.sum(h_T_dhw * F_D)

    if have_design:
        # h_T_heating evaluated at design temperature (scalar, β=1)
        _lin_H_des = (mH * design_temperature + bH) if mH != 0 or bH != 0 else 0.0
        _lin_W_des = (mW * design_temperature + bW) if mW != 0 or bW != 0 else 0.0
        _sig_des    = h_A / (1 + (h_B / (design_temperature - 40)) ** h_C)
        _h_heat_des = max(_sig_des + max(_lin_H_des, _lin_W_des) - _lin_W_des, 0.0)
        if heating_limit_temp is not None and design_temperature > heating_limit_temp:
            _h_heat_des = 0.0
        if _h_heat_des <= 0:
            raise ValueError(
                f"h_T_heating at design_temperature={design_temperature}°C is zero; "
                "check profile coefficients or heating_limit_temp setting."
            )
        design_day_energy = peak_design_kW * 24.0

        if have_jwb:
            # Mode C – bisection on β
            # Keep zeros as zero; guard positive values against underflow in h^β
            h0 = np.where(h_T_heating > 0, np.maximum(h_T_heating, 1e-15), 0.0)

            def _peak_residual(beta: float) -> float:
                shaped = np.where(h0 > 0, h0 ** beta, 0.0)
                S = float(np.dot(shaped, F_D))
                KW = annual_heat_kWh / (S + sum_h_dhw_F) if (S + sum_h_dhw_F) > 0 else 0.0
                return KW * (_h_heat_des ** beta) - design_day_energy

            b_lo, b_hi = 0.001, 50.0
            r_lo = _peak_residual(b_lo)
            r_hi = _peak_residual(b_hi)
            if r_lo * r_hi > 0:
                if r_lo > 0:
                    raise ValueError(
                        f"peak_design_kW={peak_design_kW:.1f} kW is below the minimum "
                        f"achievable peak for annual_heat_kWh={annual_heat_kWh:.0f} kWh/a with this profile."
                    )
                else:
                    raise ValueError(
                        f"peak_design_kW={peak_design_kW:.1f} kW exceeds the maximum "
                        f"achievable peak for annual_heat_kWh={annual_heat_kWh:.0f} kWh/a (β≤50)."
                    )
            for _ in range(60):
                b_mid = (b_lo + b_hi) / 2.0
                r_mid = _peak_residual(b_mid)
                if r_mid * r_lo <= 0:
                    b_hi = b_mid
                else:
                    b_lo = b_mid
                    r_lo = r_mid
            beta_solved = (b_lo + b_hi) / 2.0
            shaped = np.where(h0 > 0, h0 ** beta_solved, 0.0)
            sum_h_heat_shaped_F = float(np.dot(shaped, F_D))
            KW_kWh = annual_heat_kWh / (sum_h_heat_shaped_F + sum_h_dhw_F)
            h_T_heating = shaped
            _log.info("Mode C: β=%.4f  KW=%.4f  design-day avg heating = %.2f kW",
                      beta_solved, KW_kWh, KW_kWh * (_h_heat_des ** beta_solved) / 24)
        else:
            # Mode B – KW set directly, AHD derived
            sum_h_heat_F_orig = np.sum(h_T_heating * F_D)
            KW_kWh  = design_day_energy / _h_heat_des
            annual_heat_kWh = KW_kWh * (sum_h_heat_F_orig + sum_h_dhw_F)
            _log.info("Mode B: KW=%.4f  → annual_heat_kWh=%.0f kWh/a  (h_heat@%.1f°C=%.4f)",
                      KW_kWh, annual_heat_kWh, design_temperature, _h_heat_des)
    else:
        # Mode A – annual energy only, no shape change
        sum_h_heat_F_orig = np.sum(h_T_heating * F_D)
        KW_kWh = annual_heat_kWh / (sum_h_heat_F_orig + sum_h_dhw_F) if (sum_h_heat_F_orig + sum_h_dhw_F) != 0 else 0

    # Daily demands - proportional split, sum = annual_heat_kWh exactly
    daily_heat_demand_heating = h_T_heating * F_D * KW_kWh
    daily_heat_demand_dhw     = h_T_dhw * F_D * KW_kWh

    # Heating shape exponent: power-law reshaping of the daily heating distribution
    # exponent > 1  →  higher peaks and longer near-zero periods (same annual total)
    # exponent < 1  →  flatter profile
    if heating_exponent != 1.0:
        h_sum = daily_heat_demand_heating.sum()
        if h_sum > 0:
            shaped = np.maximum(daily_heat_demand_heating, 0.0) ** heating_exponent
            daily_heat_demand_heating = shaped * (h_sum / shaped.sum())

    # Process hourly temperature data for interpolation
    hourly_reference_temperature = np.round((hourly_temperature + 2.5) * 2, -1) / 2 - 2.5

    # Calculate temperature bounds for interpolation
    hourly_reference_temperature_2 = np.where(
        hourly_reference_temperature > hourly_temperature,
        hourly_reference_temperature - 5,
        np.where(
            hourly_reference_temperature > 27.5,
            27.5,
            hourly_reference_temperature + 5
        )
    )

    # Determine upper and lower temperature limits for interpolation
    upper_limit = np.where(
        hourly_reference_temperature_2 > hourly_reference_temperature,
        hourly_reference_temperature_2,
        hourly_reference_temperature
    )
    lower_limit = np.where(
        hourly_reference_temperature_2 > hourly_reference_temperature,
        hourly_reference_temperature,
        hourly_reference_temperature_2
    )

    # Expand daily data to hourly resolution
    daily_hours = np.tile(np.arange(24), len(days_of_year))
    hourly_weekdays = np.repeat(daily_weekdays, 24)
    hourly_daily_heat_demand_heating = np.repeat(daily_heat_demand_heating, 24)
    hourly_daily_heat_demand_dhw     = np.repeat(daily_heat_demand_dhw, 24)

    # Load BDEW hourly coefficient data
    hourly_data = pd.read_csv(os.path.join(_HERE, 'hourly_coefficients.csv'), delimiter=';')
    filtered_hourly_data = hourly_data[hourly_data["Typ"] == profile_type]

    # Create conditions dataframe for coefficient lookup
    # Note: column names ('Wochentag', 'Stunde', 'Temperatur') are fixed by the CSV schema
    hourly_conditions = pd.DataFrame({
        'Wochentag': hourly_weekdays,
        'TemperaturLower': lower_limit,
        'TemperaturUpper': upper_limit,
        'Stunde': daily_hours
    })

    # Merge hourly conditions with coefficient data for interpolation bounds
    merged_data_T1 = pd.merge(
        hourly_conditions,
        filtered_hourly_data,
        how='left',
        left_on=['Wochentag', 'TemperaturLower', 'Stunde'],
        right_on=['Wochentag', 'Temperatur', 'Stunde']
    )

    merged_data_T2 = pd.merge(
        hourly_conditions,
        filtered_hourly_data,
        how='left',
        left_on=['Wochentag', 'TemperaturUpper', 'Stunde'],
        right_on=['Wochentag', 'Temperatur', 'Stunde']
    )

    # Extract hourly factors for interpolation
    hour_factor_T1 = merged_data_T1["Stundenfaktor"].values.astype(float)
    hour_factor_T2 = merged_data_T2["Stundenfaktor"].values.astype(float)

    # Perform linear interpolation between temperature bounds
    hour_factor_interpolation = hour_factor_T2 + (hour_factor_T1 - hour_factor_T2) * (
        (hourly_temperature - upper_limit) / 5
    )

    # Calculate hourly heat demands
    hourly_heat_demand_heating = np.nan_to_num(
        (hourly_daily_heat_demand_heating * hour_factor_interpolation) / 100
    ).astype(float)

    if dhw_flat:
        # Uniform 1/24 intraday DHW distribution — temperature/weekday independent
        hourly_heat_demand_dhw = np.repeat(daily_heat_demand_dhw / 24.0, 24).astype(float)
    else:
        hourly_heat_demand_dhw = np.nan_to_num(
            (hourly_daily_heat_demand_dhw * hour_factor_interpolation) / 100
        ).astype(float)

    total = hourly_heat_demand_heating + hourly_heat_demand_dhw
    scale_factor = annual_heat_kWh / np.sum(total) if np.sum(total) > 0 else 1
    hourly_heat_demand_heating_normed = hourly_heat_demand_heating * scale_factor
    hourly_heat_demand_dhw_normed     = hourly_heat_demand_dhw * scale_factor

    # Calculate initial DHW share and apply correction if dhw_share is specified
    initial_dhw_share = (
        np.sum(hourly_heat_demand_dhw_normed) /
        (np.sum(hourly_heat_demand_heating_normed) + np.sum(hourly_heat_demand_dhw_normed))
    )

    if dhw_share is not None and not math.isnan(dhw_share):
        if 0 <= dhw_share <= 1:
            dhw_correction_factor     = dhw_share / initial_dhw_share if initial_dhw_share > 0 else 1
            heating_correction_factor = (1 - dhw_share) / (1 - initial_dhw_share) if initial_dhw_share < 1 else 1
            hourly_heat_demand_dhw_normed     *= dhw_correction_factor
            hourly_heat_demand_heating_normed *= heating_correction_factor
            total_demand = hourly_heat_demand_heating_normed + hourly_heat_demand_dhw_normed
            scale_factor = annual_heat_kWh / np.sum(total_demand) if np.sum(total_demand) > 0 else 1
            hourly_heat_demand_heating_normed *= scale_factor
            hourly_heat_demand_dhw_normed     *= scale_factor
        else:
            _log.warning("Invalid DHW share %s — using calculated value %.3f", dhw_share, initial_dhw_share)

    # Build DataFrame with DatetimeIndex
    hourly_intervals = calculate_hourly_intervals(year)
    idx = pd.DatetimeIndex(hourly_intervals)

    sh  = pd.Series(hourly_heat_demand_heating_normed.astype(float),   index=idx)
    dhw = pd.Series(hourly_heat_demand_dhw_normed.astype(float), index=idx)

    # ── Optional stochastic post-processing ───────────────────────────────────
    # Applied to the normalised shape so energy targets are preserved.
    # Step 1b: peak jitter (daily circular shift), Step 1a: log-normal noise.
    if stochastic:
        rng = np.random.default_rng(stochastic_seed)
        sh  = _apply_peak_jitter(sh,  stochastic_max_shift_sh,  rng)
        dhw = _apply_peak_jitter(dhw, stochastic_max_shift_dhw, rng)
        sh  = _apply_lognormal_noise(sh,  stochastic_sigma_sh,  rng)
        dhw = _apply_lognormal_noise(dhw, stochastic_sigma_dhw, rng)

    # ── Optional discrete DHW draw events ─────────────────────────────────────
    # Replaces the smooth DHW timeseries with clustered stochastic events.
    # Applied after stochastic so that the annual energy target is always met.
    if dhw_draw_events:
        dhw = _apply_dhw_draw_events(dhw, dhw_draws_per_day, dhw_draw_seed)

    return pd.DataFrame({
        "Q_heat_kWh":    sh,
        "Q_dhw_kWh":     dhw,
        "Q_total_kWh":   sh + dhw,
        "temperature_C": hourly_temperature,
    }, index=idx)


if __name__ == "__main__":
    # Smoke test – verify energy balance is maintained.
    # For a full parameter comparison run:  python examples/bdew_demo.py
    try:
        from pyslpheat import TRY_BAUTZEN_2015
    except ImportError:
        TRY_BAUTZEN_2015 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "try", "TRY2015_511676144222_Jahr.dat"
        )
    _df = calculate(20_000, "HMF", "03", TRY_BAUTZEN_2015, 2026, dhw_share=0.25)
    _total = _df["Q_total_kWh"].sum()
    assert abs(_total - 20_000) / 20_000 < 0.001, \
        f"Energy balance error: {_total:.0f} kWh (expected 20000)"
    print(f"bdew.py OK  HMF03  total={_total:.0f} kWh  "
          f"peak={_df['Q_total_kWh'].max():.2f} kWh/h")