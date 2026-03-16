#!/usr/bin/env python3
"""
BDEW heat load profile demo – comprehensive parameter comparison.

Generates six comparison figures covering all configuration options:
  1. HMF subtype comparison (03 / 04 / 05 / 33 / 34)
  2. Building-type comparison (HEF, HMF, GKO, GBD, GHD)
  3. Heating configuration (heating_limit_temp, heating_exponent)
  4. DHW configuration (dhw_share, dhw_flat)
  5. Scaling modes (A – annual only, B – design only, C – both)
  6. Stochastic post-processing (deterministic vs multiple realisations)

All figures are saved as PNG files in the output directory.
HMF subtype CSVs are also exported.

Usage::

    python bdew_demo.py [--try-file PATH] [--output ./output]

If --try-file is omitted the bundled Bautzen TRY 2015 is used.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from pyslpheat import bdew, TRY_BAUTZEN_2015
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(_repo_dir, "pyslpheat"))
    import bdew  # noqa: E402
    TRY_BAUTZEN_2015 = os.path.join(_repo_dir, "pyslpheat", "data", "try",
                                     "TRY2015_511676144222_Jahr.dat")

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="BDEW comprehensive demo")
ap.add_argument("--try-file", default=None, metavar="PATH",
                help="DWD TRY .dat file (default: bundled Bautzen 2015)")
ap.add_argument("--output", default=None, metavar="DIR",
                help="Output directory for PNG/CSV files "
                     "(default: <repo>/output)")
args = ap.parse_args()

TRY_PATH   = args.try_file if args.try_file else TRY_BAUTZEN_2015
OUTPUT_DIR = os.path.abspath(args.output) if args.output else \
             os.path.join(_repo_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR        = 2026
ANNUAL_HEAT = 20_000  # kWh/a
DHW_SHARE   = 0.25    # 25 % DHW share of total heat


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calc(**kwargs) -> pd.DataFrame:
    """bdew.calculate with fixed TRY / year / annual energy / DHW share."""
    return bdew.calculate(
        annual_heat_kWh=ANNUAL_HEAT,
        TRY_file_path=TRY_PATH,
        year=YEAR,
        dhw_share=DHW_SHARE,
        **kwargs,
    )


def _ldc(series: pd.Series) -> np.ndarray:
    return series.sort_values(ascending=False).values


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _fmt_week_ax(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)


# ── Reference profile – peak window ───────────────────────────────────────────
print("Calculating reference profile (HMF03) ...")
df_ref   = _calc(profile_type="HMF", subtype="03")
_peak_day = df_ref["Q_total_kWh"].idxmax().normalize()
W_START   = (_peak_day - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
W_END     = (_peak_day + pd.Timedelta(days=3)).strftime("%Y-%m-%d") + " 23:00"
print(f"  Peak window: {W_START} → {W_END}")


# ── Figure 1: HMF subtype comparison ──────────────────────────────────────────
print("\nFigure 1 – HMF subtype comparison ...")
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 5))
fig1.suptitle(f"BDEW – HMF subtypes  ({ANNUAL_HEAT:,} kWh/a, dhw={DHW_SHARE})", fontsize=12)

for st in ["03", "04", "05", "33", "34"]:
    df = _calc(profile_type="HMF", subtype=st)
    peak = df["Q_total_kWh"].max()
    ax1a.plot(df["Q_total_kWh"].loc[W_START:W_END],
              lw=1.2, label=f"HMF{st}  (peak={peak:.1f} kWh/h)")
    ax1b.plot(_ldc(df["Q_total_kWh"]), lw=1.5,
              label=f"HMF{st}  (peak={peak:.1f})")
    df.index.name = "timestamp"
    df.to_csv(os.path.join(OUTPUT_DIR, f"bdew_HMF{st}_{YEAR}.csv"))

ax1a.set_title("Peak-load week – total heat")
ax1a.set_ylabel("kWh/h")
ax1a.legend(fontsize=8)
_fmt_week_ax(ax1a)

ax1b.set_title("Load duration curve – total heat")
ax1b.set_xlabel("Hours [h/a]")
ax1b.set_ylabel("kWh/h")
ax1b.legend(fontsize=8)
ax1b.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig1, f"fig1_hmf_subtypes_{YEAR}.png")


# ── Figure 2: Building type comparison ────────────────────────────────────────
print("Figure 2 – Building type comparison ...")
BTYPES = [
    ("HEF", "03", "Single-family residential (HEF03)"),
    ("HMF", "03", "Multi-family residential (HMF03)"),
    ("GKO", "03", "Small commercial/public (GKO03)"),
    ("GBD", "03", "Office services (GBD03)"),
    ("GHD", "03", "General commercial aggregate (GHD03)"),
]
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle(f"BDEW – building type comparison  ({ANNUAL_HEAT:,} kWh/a, subtype 03)",
              fontsize=12)

for ptype, st, desc in BTYPES:
    df = _calc(profile_type=ptype, subtype=st)
    ax2a.plot(df["Q_total_kWh"].loc[W_START:W_END], lw=1.2, label=desc)
    ax2b.plot(_ldc(df["Q_total_kWh"]), lw=1.5,
              label=f"{ptype}{st}  (peak={df['Q_total_kWh'].max():.1f})")

ax2a.set_title("Peak-load week – total heat")
ax2a.set_ylabel("kWh/h")
ax2a.legend(fontsize=8)
_fmt_week_ax(ax2a)

ax2b.set_title("Load duration curve – total heat")
ax2b.set_xlabel("Hours [h/a]")
ax2b.set_ylabel("kWh/h")
ax2b.legend(fontsize=8)
ax2b.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig2, f"fig2_building_types_{YEAR}.png")


# ── Figure 3: Heating configuration ───────────────────────────────────────────
print("Figure 3 – Heating configuration options ...")

LIMIT_VARS = {
    "no limit (default)": dict(heating_limit_temp=None,  heating_exponent=1.0),
    "T_lim = 15 °C":      dict(heating_limit_temp=15.0,  heating_exponent=1.0),
    "T_lim = 12 °C":      dict(heating_limit_temp=12.0,  heating_exponent=1.0),
}
EXP_VARS = {
    "exp = 0.7 (flat)":       dict(heating_exponent=0.7,  heating_limit_temp=None),
    "exp = 1.0 (default)":    dict(heating_exponent=1.0,  heating_limit_temp=None),
    "exp = 1.3":               dict(heating_exponent=1.3,  heating_limit_temp=None),
    "exp = 1.5 (peaked)":     dict(heating_exponent=1.5,  heating_limit_temp=None),
    "T_lim=12 + exp=1.5":     dict(heating_exponent=1.5,  heating_limit_temp=12.0),
}

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle("BDEW HMF03 – heating configuration options", fontsize=12)
(ax3_lw, ax3_ll), (ax3_ew, ax3_el) = axes3

for label, kw in LIMIT_VARS.items():
    df = _calc(profile_type="HMF", subtype="03", **kw)
    ax3_lw.plot(df["Q_heat_kWh"].loc[W_START:W_END], lw=1.2,
                label=f"{label}  Σ={df['Q_heat_kWh'].sum():.0f} kWh")
    ax3_ll.plot(_ldc(df["Q_heat_kWh"]), lw=1.5,
                label=f"{label}  peak={df['Q_heat_kWh'].max():.1f}")

for label, kw in EXP_VARS.items():
    df = _calc(profile_type="HMF", subtype="03", **kw)
    ax3_ew.plot(df["Q_heat_kWh"].loc[W_START:W_END], lw=1.2,
                label=f"{label}  peak={df['Q_heat_kWh'].max():.1f}")
    ax3_el.plot(_ldc(df["Q_heat_kWh"]), lw=1.5,
                label=f"{label}  peak={df['Q_heat_kWh'].max():.1f}")

for ax, title in [
    (ax3_lw, "heating_limit_temp – peak week (space heating)"),
    (ax3_ll, "heating_limit_temp – LDC (space heating)"),
    (ax3_ew, "heating_exponent – peak week (space heating)"),
    (ax3_el, "heating_exponent – LDC (space heating)"),
]:
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kWh/h")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in (ax3_lw, ax3_ew):
    _fmt_week_ax(ax)

for ax in (ax3_ll, ax3_el):
    ax.set_xlabel("Hours [h/a]")

plt.tight_layout()
_save(fig3, f"fig3_heating_config_{YEAR}.png")


# ── Figure 4: DHW configuration ───────────────────────────────────────────────
print("Figure 4 – DHW configuration options ...")

SHARE_VARS = {
    "dhw_share = 0.15": 0.15,
    "dhw_share = 0.20": 0.20,
    "dhw_share = 0.25": 0.25,
    "dhw_share = 0.30": 0.30,
    "dhw_share = 0.35": 0.35,
}
FLAT_VARS = {
    "dhw_flat = False  (shaped, default)": dict(dhw_flat=False),
    "dhw_flat = True   (uniform 1/24)":    dict(dhw_flat=True),
}

fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
fig4.suptitle("BDEW HMF03 – DHW configuration options", fontsize=12)
(ax4_sw, ax4_sl), (ax4_fw, ax4_fl) = axes4

for label, share in SHARE_VARS.items():
    df = bdew.calculate(ANNUAL_HEAT, "HMF", "03", TRY_PATH, YEAR, dhw_share=share)
    ax4_sw.plot(df["Q_dhw_kWh"].loc[W_START:W_END], lw=1.2,
                label=f"{label}  Σ={df['Q_dhw_kWh'].sum():.0f} kWh")
    ax4_sl.plot(_ldc(df["Q_total_kWh"]), lw=1.5,
                label=f"{label}  peak_tot={df['Q_total_kWh'].max():.1f}")

for label, kw in FLAT_VARS.items():
    df = _calc(profile_type="HMF", subtype="03", **kw)
    ax4_fw.plot(df["Q_dhw_kWh"].loc[W_START:W_END], lw=1.5, label=label)
    ax4_fl.plot(_ldc(df["Q_dhw_kWh"]), lw=1.5, label=label)

for ax, title in [
    (ax4_sw, "dhw_share – peak week (DHW demand)"),
    (ax4_sl, "dhw_share – LDC (total heat)"),
    (ax4_fw, "dhw_flat – peak week (DHW demand)"),
    (ax4_fl, "dhw_flat – LDC (DHW demand)"),
]:
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kWh/h")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in (ax4_sw, ax4_fw):
    _fmt_week_ax(ax)

for ax in (ax4_sl, ax4_fl):
    ax.set_xlabel("Hours [h/a]")

plt.tight_layout()
_save(fig4, f"fig4_dhw_config_{YEAR}.png")


# ── Figure 5: Scaling modes A / B / C ─────────────────────────────────────────
print("Figure 5 – Scaling modes A / B / C ...")
T_DES = -12.4  # °C design outside temperature

MODE_VARS = [
    ("Mode A  – 20 000 kWh/a, no peak", "-",
     dict(annual_heat_kWh=ANNUAL_HEAT)),
    ("Mode B  – peak  8 kW (annual derived)", "--",
     dict(annual_heat_kWh=None, peak_design_kW= 8.0, design_temperature=T_DES)),
    ("Mode B  – peak 13 kW (annual derived)", "--",
     dict(annual_heat_kWh=None, peak_design_kW=13.0, design_temperature=T_DES)),
    ("Mode C  – 20 000 kWh/a + peak  8 kW", "-",
     dict(annual_heat_kWh=ANNUAL_HEAT, peak_design_kW= 8.0, design_temperature=T_DES)),
    ("Mode C  – 20 000 kWh/a + peak 13 kW", "-",
     dict(annual_heat_kWh=ANNUAL_HEAT, peak_design_kW=13.0, design_temperature=T_DES)),
]

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 5))
fig5.suptitle(f"BDEW HMF03 – scaling modes  (T_design = {T_DES} °C, dhw={DHW_SHARE})",
              fontsize=12)

for label, ls, kw in MODE_VARS:
    df = bdew.calculate(
        profile_type="HMF", subtype="03",
        TRY_file_path=TRY_PATH, year=YEAR, dhw_share=DHW_SHARE, **kw,
    )
    ax5a.plot(df["Q_total_kWh"].loc[W_START:W_END], lw=1.3, ls=ls, label=label)
    ax5b.plot(_ldc(df["Q_total_kWh"]), lw=1.5, ls=ls,
              label=f"{label}  Σ={df['Q_total_kWh'].sum():,.0f} peak={df['Q_total_kWh'].max():.1f}")

ax5a.set_title("Peak-load week – total heat")
ax5a.set_ylabel("kWh/h")
ax5a.legend(fontsize=7)
_fmt_week_ax(ax5a)

ax5b.set_title("Load duration curve – total heat")
ax5b.set_xlabel("Hours [h/a]")
ax5b.set_ylabel("kWh/h")
ax5b.legend(fontsize=7)
ax5b.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig5, f"fig5_scaling_modes_{YEAR}.png")


# ── Figure 6: Stochastic post-processing ──────────────────────────────────────
print("Figure 6 – Stochastic post-processing ...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 5))
fig6.suptitle(
    "BDEW HMF03 – stochastic post-processing  (deterministic vs 3 realisations)",
    fontsize=12,
)

df_det = _calc(profile_type="HMF", subtype="03")
ax6a.plot(df_det["Q_total_kWh"].loc[W_START:W_END], lw=2.0, color="steelblue",
          label=f"Deterministic  (peak={df_det['Q_total_kWh'].max():.1f})", zorder=5)
ax6b.plot(_ldc(df_det["Q_total_kWh"]), lw=2.0, color="steelblue",
          label="Deterministic", zorder=5)

for seed, color in [(42, "#e6550d"), (7, "#31a354"), (123, "#756bb1")]:
    df_s = _calc(profile_type="HMF", subtype="03", stochastic=True, stochastic_seed=seed)
    ax6a.plot(df_s["Q_total_kWh"].loc[W_START:W_END], lw=0.9, color=color, alpha=0.85,
              label=f"Stochastic seed={seed}  (peak={df_s['Q_total_kWh'].max():.1f})")
    ax6b.plot(_ldc(df_s["Q_total_kWh"]), lw=1.2, color=color, alpha=0.85,
              label=f"Stochastic seed={seed}")

ax6a.set_title("Peak-load week – total heat")
ax6a.set_ylabel("kWh/h")
ax6a.legend(fontsize=8)
_fmt_week_ax(ax6a)

ax6b.set_title("Load duration curve – total heat")
ax6b.set_xlabel("Hours [h/a]")
ax6b.set_ylabel("kWh/h")
ax6b.legend(fontsize=8)
ax6b.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig6, f"fig6_stochastic_{YEAR}.png")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
