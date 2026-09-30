"""
Verbatim copies of helper functions as of commit 2751480 (v0.3.0).

They serve as the reference in ``test_helpers_reference.py``: the public
helpers in ``pyslpheat.bdew`` and ``pyslpheat.vdi4655`` must keep returning
exactly what these slow implementations return.
"""

import numpy as np
import pandas as pd

from pyslpheat.vdi4655 import calculate_quarter_hourly_intervals, get_resource_path


def bdew_import_TRY(filepath):
    temperatures = []
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


def bdew_calculate_allocation_temperature(daily_avg_temperature):
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


def bdew_get_weekday_factor(daily_weekdays, profile_type, subtype, daily_data):
    profile = profile_type + subtype
    profile_row = daily_data[daily_data['Standardlastprofil'] == profile]
    if profile_row.empty:
        raise ValueError(f"Profile '{profile}' not found in BDEW coefficient data")
    try:
        weekday_factors = np.array([
            profile_row.iloc[0][str(day)] for day in daily_weekdays
        ]).astype(float)
    except KeyError as e:
        raise KeyError(f"Missing weekday column in BDEW data: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid weekday factor value: {e}") from e
    return weekday_factors


def vdi_import_TRY(filename):
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
    temperature = np.array(temps, dtype=float)
    windspeed = np.array(winds, dtype=float)
    direct_radiation = np.array(dirs, dtype=float)
    diffuse_radiation = np.array(diffs, dtype=float)
    global_radiation = direct_radiation + diffuse_radiation
    cloud_cover = np.array(clouds, dtype=float)
    return temperature, windspeed, direct_radiation, global_radiation, cloud_cover


def vdi_standardized_quarter_hourly_profile(year, building_type, days_of_year, type_days):
    quarter_hourly_intervals = calculate_quarter_hourly_intervals(year)
    daily_dates = np.array([np.datetime64(dt, 'D') for dt in quarter_hourly_intervals])
    indices = np.searchsorted(days_of_year, daily_dates)
    quarterly_type_days = type_days[indices % len(type_days)]

    all_type_days = np.unique(quarterly_type_days)
    all_data = {}
    for type_day in all_type_days:
        profile_filename = f"{building_type}{type_day}.csv"
        file_path = get_resource_path(
            f'data\\VDI 4655 profiles\\VDI 4655 load profiles\\{profile_filename}')
        try:
            profile_data = pd.read_csv(file_path, sep=';')
            all_data[f"{building_type}{type_day}"] = profile_data
        except FileNotFoundError:
            times = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
            dummy_data = pd.DataFrame({
                'Zeit': times,
                'Strombedarf normiert': np.ones(96),
                'Heizwärme normiert': np.ones(96),
                'Warmwasser normiert': np.ones(96),
            })
            all_data[f"{building_type}{type_day}"] = dummy_data

    profile_days = np.char.add(building_type, quarterly_type_days)
    times_str = np.datetime_as_string(quarter_hourly_intervals, unit='m')
    times = np.array([t.split('T')[1] for t in times_str])
    times_profile_df = pd.DataFrame({
        'Datum': np.repeat(days_of_year, 24*4),
        'Zeit': times,
        'ProfileDay': profile_days
    })
    combined_df = pd.concat([
        df.assign(ProfileDay=profile_day)
        for profile_day, df in all_data.items()
    ])
    merged_df = pd.merge(times_profile_df, combined_df, on=['Zeit', 'ProfileDay'], how='left')

    electricity_demand = merged_df['Strombedarf normiert'].values
    heating_demand = merged_df['Heizwärme normiert'].values
    hot_water_demand = merged_df['Warmwasser normiert'].values

    electricity_demand = np.nan_to_num(electricity_demand, nan=1.0)
    heating_demand = np.nan_to_num(heating_demand, nan=1.0)
    hot_water_demand = np.nan_to_num(hot_water_demand, nan=1.0)

    return quarter_hourly_intervals, electricity_demand, heating_demand, hot_water_demand


def bdew_apply_peak_jitter(series, max_shift, rng):
    result = series.copy()
    for day in series.index.normalize().unique():
        mask = series.index.normalize() == day
        vals = series[mask].values
        if len(vals) < 24:
            continue
        shift = int(rng.integers(-max_shift, max_shift + 1))
        result[mask] = np.roll(vals, shift)
    return result


def bdew_apply_dhw_draw_events(dhw, draws_per_day, seed):
    rng = np.random.default_rng(seed)
    original_dhw = dhw.sum()

    new_dhw = np.zeros(len(dhw))
    days = pd.Series(dhw.index.date).unique()
    hour_index = {ts: i for i, ts in enumerate(dhw.index)}

    for day in days:
        n_draws = rng.poisson(draws_per_day)
        for _ in range(n_draws):
            if rng.random() < 0.60:
                start_h = int(rng.uniform(5, 9))
            else:
                start_h = int(rng.uniform(17, 22))

            duration = int(rng.uniform(1, 4))
            amp = min(rng.lognormal(0.0, 0.4), 2.0)

            for dh in range(duration):
                h = (start_h + dh) % 24
                ts = pd.Timestamp(year=day.year, month=day.month,
                                  day=day.day, hour=h)
                if ts in hour_index:
                    new_dhw[hour_index[ts]] += amp

    total = new_dhw.sum()
    if total > 0:
        new_dhw *= original_dhw / total

    return pd.Series(new_dhw, index=dhw.index, dtype=float)
