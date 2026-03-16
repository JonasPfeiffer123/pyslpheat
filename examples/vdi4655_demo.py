#!/usr/bin/env python3
"""
VDI 4655 heat load profile demo.

Calculates 15-minute heat and electricity demand profiles for residential
buildings using the VDI 4655 day-type methodology, then saves results to CSV
and produces four comparison figures.

Method (VDI 4655, May 2008):
  - Day-type classification per day: season (W/Ü/S) × weekday type (W/S)
    × cloud cover (H/B/X) from the DWD TRY weather dataset.
  - Daily energy scaling factors (F_heiz,TT / F_TWW,TT / F_el,TT) from
    Faktoren.csv.
  - Quarter-hourly intraday profiles from the VDI 4655 profile CSV files.
  - Final normalization to annual energy targets.

Note: VDI 4655 produces step-function profiles — all days of the same type
are identical.  For temperature-continuous profiles prefer the BDEW module.

Figures saved to OUTPUT_DIR:
  fig1_annual_{YEAR}.png          – Full-year hourly overview (MFH)
  fig2_seasonal_{YEAR}.png        – Peak-load week + summer week (MFH)
  fig3_ldc_{YEAR}.png             – Load-duration curves (MFH)
  fig4_building_types_{YEAR}.png  – EFH vs MFH comparison (annual + LDC)

Usage::

    python vdi4655_demo.py [--try-file path/to/TRY.dat] [--output ./output]
"""

import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from pyslpheat import TRY_BAUTZEN_2015
    from pyslpheat import vdi4655_calculate as vdi4655
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(_repo_dir, "pyslpheat"))
    from vdi4655 import calculate as vdi4655  # noqa: E402
    TRY_BAUTZEN_2015 = os.path.join(_repo_dir, "pyslpheat", "data", "try",
                                     "TRY2015_511676144222_Jahr.dat")

# ── CLI arguments ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="VDI 4655 heat load profile demo")
ap.add_argument("--try-file", default=None,
                help="Path to DWD TRY .dat file (default: bundled Bautzen 2015)")
ap.add_argument("--output", default=None,
                help="Output directory (default: <repo>/output)")
args = ap.parse_args()

TRY_PATH    = args.try_file if args.try_file else TRY_BAUTZEN_2015
OUTPUT_DIR  = os.path.abspath(args.output) if args.output else \
              os.path.join(_repo_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared parameters ──────────────────────────────────────────────────────────
YEAR               = 2026
CLIMATE_ZONE       = "9"       # DWD climate zone – central Germany (Bautzen)
ANNUAL_HEATING     = 15_000.0  # kWh/a  space heating
ANNUAL_DHW         =  5_000.0  # kWh/a  domestic hot water
ANNUAL_ELECTRICITY =  4_000.0  # kWh/a  (required by VDI 4655)

# German public holidays 2026 (treated as Sundays in day-type classification)
HOLIDAYS = np.array([
    "2026-01-01", "2026-04-03", "2026-04-06",
    "2026-05-01", "2026-05-14", "2026-05-25",
    "2026-10-03", "2026-12-25", "2026-12-26",
], dtype="datetime64[D]")

COLORS = {
    "Q_heat_kWh":        "#1f77b4",
    "Q_dhw_kWh":         "#ff7f0e",
    "Q_total_kWh":       "#2ca02c",
    "Q_electricity_kWh": "#9467bd",
}
LABELS = {
    "Q_heat_kWh":        "Heating",
    "Q_dhw_kWh":         "DHW",
    "Q_total_kWh":       "Total heat",
    "Q_electricity_kWh": "Electricity",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calc(building_type: str, n_people: int, **kwargs) -> pd.DataFrame:
    """Run VDI 4655 calculation and resample 15-min → hourly kWh."""
    df_15min = vdi4655(
        annual_heating_kWh=ANNUAL_HEATING,
        annual_dhw_kWh=ANNUAL_DHW,
        annual_electricity_kWh=ANNUAL_ELECTRICITY,
        building_type=building_type,
        number_people_household=n_people,
        year=YEAR,
        climate_zone=CLIMATE_ZONE,
        TRY=TRY_PATH,
        holidays=HOLIDAYS,
        **kwargs,
    )
    df_h = df_15min[["Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh",
                      "Q_electricity_kWh"]].resample("h").sum()
    df_h = df_h.iloc[:8760].copy()
    df_h.index = pd.date_range(f"{YEAR}-01-01", periods=8760, freq="h")
    df_h.index.name = "timestamp"
    return df_h


def _ldc(series: pd.Series) -> np.ndarray:
    """Return values sorted descending (load-duration curve)."""
    return series.sort_values(ascending=False).values


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _fmt_week_ax(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)


# ── Reference MFH calculation ─────────────────────────────────────────────────
print("Calculating VDI 4655 MFH profile …")
df_mfh = _calc("MFH", 4)

print(f"  Annual heating     : {df_mfh['Q_heat_kWh'].sum():,.0f} kWh  "
      f"(target {ANNUAL_HEATING:,.0f})")
print(f"  Annual DHW         : {df_mfh['Q_dhw_kWh'].sum():,.0f} kWh  "
      f"(target {ANNUAL_DHW:,.0f})")
print(f"  Annual electricity : {df_mfh['Q_electricity_kWh'].sum():,.0f} kWh  "
      f"(target {ANNUAL_ELECTRICITY:,.0f})")
print(f"  Peak total heat    : {df_mfh['Q_total_kWh'].max():.2f} kWh/h")

# CSV export
_csv = os.path.join(OUTPUT_DIR, f"vdi4655_mfh_cz{CLIMATE_ZONE}_{YEAR}.csv")
df_mfh.to_csv(_csv)
print(f"  CSV saved: {_csv}")

# Peak window for weekly detail
_peak_ts  = df_mfh["Q_total_kWh"].idxmax()
_peak_day = _peak_ts.normalize()
_w_start  = _peak_day - pd.Timedelta(days=3)
_w_end    = _peak_day + pd.Timedelta(days=3, hours=23)

# ── Figure 1 – Full-year overview (MFH, hourly) ───────────────────────────────
print("\nFigure 1: full-year overview …")
fig1, axes1 = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig1.suptitle(f"VDI 4655 – MFH  climate zone {CLIMATE_ZONE}  {YEAR}  "
              f"(heating {ANNUAL_HEATING/1000:.0f} MWh/a + "
              f"DHW {ANNUAL_DHW/1000:.0f} MWh/a)", fontsize=12)

for ax, col in zip(axes1, ["Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh"]):
    ax.plot(df_mfh.index, df_mfh[col],
            color=COLORS[col], lw=0.5, alpha=0.85, label=LABELS[col])
    ax.axvline(_peak_ts, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("kWh/h")
    ax.set_title(LABELS[col], fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes1[-1].xaxis.set_major_locator(mdates.MonthLocator())

# Annotate peak on the total panel
_peak_val = df_mfh["Q_total_kWh"].max()
axes1[2].annotate(
    f"Peak {_peak_val:.1f} kWh/h\n{_peak_ts.strftime('%d.%m. %Hh')}",
    xy=(_peak_ts, _peak_val),
    xytext=(18, -28), textcoords="offset points",
    fontsize=8, color=COLORS["Q_total_kWh"],
    arrowprops=dict(arrowstyle="->", color=COLORS["Q_total_kWh"], lw=0.8),
)

plt.tight_layout()
_save(fig1, f"fig1_annual_{YEAR}.png")

# ── Figure 2 – Seasonal detail (peak week + summer week) ─────────────────────
print("Figure 2: seasonal detail …")
PERIODS = {
    f"Peak-load week  ({_w_start.strftime('%d.%m')}–{_w_end.strftime('%d.%m')})":
        (_w_start, _w_end),
    f"Summer week  (1–7 Jul {YEAR})":
        (pd.Timestamp(f"{YEAR}-07-01"), pd.Timestamp(f"{YEAR}-07-07 23:00")),
}

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
fig2.suptitle(f"VDI 4655 – MFH  climate zone {CLIMATE_ZONE}  {YEAR}  "
              "seasonal detail (hourly)", fontsize=12)

for ax, (title, (t0, t1)) in zip(axes2, PERIODS.items()):
    sl = df_mfh.loc[t0:t1]
    for col in ("Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh"):
        ax.plot(sl.index, sl[col],
                label=LABELS[col], color=COLORS[col], lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("kWh/h")
    ax.legend(fontsize=8)
    _fmt_week_ax(ax)

plt.tight_layout()
_save(fig2, f"fig2_seasonal_{YEAR}.png")

# ── Figure 3 – Load-duration curve (MFH) ─────────────────────────────────────
print("Figure 3: load-duration curve …")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(f"VDI 4655 – MFH  climate zone {CLIMATE_ZONE}  {YEAR}  "
              "load-duration curve", fontsize=12)

hours = np.arange(1, 8761)

# Left: heat streams
for col in ("Q_heat_kWh", "Q_dhw_kWh", "Q_total_kWh"):
    ax3a.plot(hours, _ldc(df_mfh[col]),
              label=LABELS[col], color=COLORS[col], lw=1.5)
ax3a.set_title("Heat demand", fontsize=10)
ax3a.set_xlabel("Hours [h/a]")
ax3a.set_ylabel("kWh/h")
ax3a.legend(fontsize=8)
ax3a.grid(True, alpha=0.3)

# Right: electricity
ax3b.plot(hours, _ldc(df_mfh["Q_electricity_kWh"]),
          label=LABELS["Q_electricity_kWh"],
          color=COLORS["Q_electricity_kWh"], lw=1.5)
ax3b.set_title("Electricity demand", fontsize=10)
ax3b.set_xlabel("Hours [h/a]")
ax3b.set_ylabel("kWh/h")
ax3b.legend(fontsize=8)
ax3b.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig3, f"fig3_ldc_{YEAR}.png")

# ── Figure 4 – Building-type comparison: EFH (3 occ.) vs MFH (4 dw.) ─────────
print("Figure 4: building-type comparison …")

BUILDINGS = {
    "EFH  3 occupants": ("EFH", 3),
    "MFH  4 dw. units": ("MFH", 4),
}

results: dict[str, pd.DataFrame] = {}
for label, (btype, n) in BUILDINGS.items():
    print(f"  Calculating {label} …")
    results[label] = _calc(btype, n)

BLD_COLORS = ["#1f77b4", "#d62728"]

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 9))
fig4.suptitle(f"VDI 4655 – EFH vs MFH  climate zone {CLIMATE_ZONE}  {YEAR}  "
              "(same annual energy targets)", fontsize=12)

ax_year_heat, ax_year_dhw = axes4[0]
ax_ldc_heat,  ax_ldc_dhw  = axes4[1]

for (label, df_b), col in zip(results.items(), BLD_COLORS):
    # Annual time series (total heat)
    ax_year_heat.plot(df_b.index, df_b["Q_total_kWh"],
                      label=label, color=col, lw=0.5, alpha=0.8)
    ax_year_dhw.plot(df_b.index, df_b["Q_dhw_kWh"],
                     label=label, color=col, lw=0.5, alpha=0.8)
    # LDC
    ax_ldc_heat.plot(hours, _ldc(df_b["Q_total_kWh"]),
                     label=label, color=col, lw=1.5)
    ax_ldc_dhw.plot(hours, _ldc(df_b["Q_dhw_kWh"]),
                    label=label, color=col, lw=1.5)

for ax, title in zip([ax_year_heat, ax_year_dhw],
                     ["Total heat – full year", "DHW – full year"]):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("kWh/h")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

for ax, title in zip([ax_ldc_heat, ax_ldc_dhw],
                     ["Total heat – LDC", "DHW – LDC"]):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Hours [h/a]")
    ax.set_ylabel("kWh/h")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig4, f"fig4_building_types_{YEAR}.png")

print("\nDone.  All figures and CSV saved to:", OUTPUT_DIR)
